# %%
# %% [cell 1] (markdown)
"""
# Imports
"""

# %% [cell 2] (code)
# %pip install -q datasets sentence-transformers torch

# %% [cell 3] (code)
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from datasets import load_dataset  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies. Create a venv and install requirements, then try again:\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  python -m pip install -U pip\n"
        "  python -m pip install -r requirements.txt\n\n"
        f"Import error: {e}"
    )

# %% [cell 4] (markdown)
"""
# Configs
"""

# %% [cell 5] (code)
@dataclass(frozen=True)
class Config:
    dataset_name: str = "Anthropic/hh-rlhf"  # if this breaks, check the exact HF name
    split: str = "train"
    num_pairs: int = 1_000
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256
    seed: int = 0
    train_frac: float = 0.8
    optimizer_steps: int = 50
    l2_reg: float = 1e-4
    gd_lr: float = 1e-2
    wolfe_alpha_max: float = 50.0
    # Good default conditioning for this assignment:
    normalize_embeddings: bool = True
    xdiff_scale: str = "std"  # one of: none, l2, std
    xdiff_mult: float = 1.0
    # Default outputs for the run (can be overridden via --save-metrics-prefix)
    save_metrics_prefix: str = "bench/run1"
    # Log cadence (set --log-every 0 to silence optimizer-step logs).
    log_every: int = 10


CFG = Config()

# %% [cell 6] (markdown)
"""
# Dataset handlers
Set up:
1. Dataset importing
2. Pair construction
3. Embedding creation
4. Splitting of pairs into training / validation
"""

# %% [cell 7] (code)
# Build preference pairs
def build_pairs(dataset, max_pairs: int=None) -> Tuple[list, list, list]:
  chosen_prompts = []
  rejected_prompts = []
  labels = []

  # iterate through examples
  for example in dataset:
    if len(chosen_prompts) >= max_pairs:
      break

    # get the chosen and rejected
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")
    if chosen == "" or rejected == "":
      continue
    else:
      chosen_prompts.append(chosen)
      rejected_prompts.append(rejected)
      labels.append(1)

  return chosen_prompts, rejected_prompts, labels

# %% [cell 11] (markdown)
"""
### Embeddings
"""

# %% [cell 12] (code)
## EMBEDDINGS

def embed_texts(embed_model, texts, batch_size=256, normalize_embeddings: bool = False):
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = embed_model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings,
        )
        all_emb.append(emb)
    return np.vstack(all_emb)

# %% [cell 13] (markdown)
"""
### Batching
"""

# %% [cell 14] (code)
def train_val_split(A: np.ndarray, B: np.ndarray, y: np.ndarray, train_frac: float, seed: int):
    assert len(A) == len(B) == len(y)
    N = len(A)
    train_N = int(train_frac * N)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    train_idx = perm[:train_N]
    val_idx = perm[train_N:]
    return (
        A[train_idx],
        B[train_idx],
        y[train_idx],
        A[val_idx],
        B[val_idx],
        y[val_idx],
    )

@dataclass
class XDiffScaler:
    method: str = "none"
    scale_scalar: float = 1.0
    scale_vector: Optional[np.ndarray] = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return X
        if self.method == "l2":
            return X / float(self.scale_scalar)
        if self.method == "std":
            assert self.scale_vector is not None
            return X / self.scale_vector
        raise ValueError(f"Unknown xdiff scaling method: {self.method!r}")


def fit_xdiff_scaler(Xdiff_train: np.ndarray, method: str = "none", eps: float = 1e-12) -> XDiffScaler:
    """
    Fit a scaling transform on TRAIN data, then re-use it for both train and val.
    """
    method = (method or "none").lower()
    if method == "none":
        return XDiffScaler(method="none")
    if method == "l2":
        row_norms = np.linalg.norm(Xdiff_train, axis=1)
        scale = float(np.mean(row_norms) + eps)
        return XDiffScaler(method="l2", scale_scalar=scale)
    if method == "std":
        s = np.std(Xdiff_train, axis=0)
        s = np.where(s < eps, 1.0, s)
        return XDiffScaler(method="std", scale_vector=s)
    raise ValueError(f"Unknown xdiff scaling method: {method!r}. Use one of: none, l2, std")


# %% [cell 16] (markdown)
"""
# Utilities
"""

# %% [cell 17] (code)
def sigmoid(z):
    # numerically stable sigmoid
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def bradley_terry_loss(z, y):
    """
    z: logits (N,)
    y: labels (N,) in {0,1}
    the bradley terry loss formula (withorugh normalization): -sum(y * log(p) + (1 - y) * log(1 - p))
    """
    p = sigmoid(z)
    eps = 1e-12
    unnormalized_loss = -1* (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    # normalize
    return np.mean(unnormalized_loss)

def accuracy_from_logits(z, y):
    p = sigmoid(z)
    preds = (p >= 0.5).astype(np.float64)
    return np.mean(preds == y)

# %% [cell 18] (markdown)
"""
# Linear reward model

This is of the form:
- Score: r_\theta(x) = \theta.Tx
- Logit for pair: z_i = \theta.T (x_{A,i} - x_{B, i})
"""

# %% [cell 19] (code)
def linear_objective(theta, Xdiff, y, l2_reg: float = 0.0):
    """
    theta: (d,)
    Xdiff: (N, d)
    y: (N,)
    Returns: loss (scalar), grad (d,), hess (d,d)
    """
    z = Xdiff @ theta        # (N,)
    p = sigmoid(z)           # (N,)
    loss = bradley_terry_loss(z, y)
    if l2_reg:
        loss = loss + 0.5 * float(l2_reg) * float(theta @ theta)

    # gradient: (1/N) * Xdiff^T (p - y)
    N = Xdiff.shape[0]
    residual = (p - y)       # (N,)
    grad = Xdiff.T @ residual / N  # (d,)
    if l2_reg:
        grad = grad + float(l2_reg) * theta

    # Hessian: (1/N) * X^T W X, W = diag(p(1-p))
    w = p * (1 - p)          # (N,)
    # X^T * diag(w) * X = (X^T * (w[:,None] * X))
    Xw = Xdiff * w[:, None]  # (N,d)
    hess = Xdiff.T @ Xw / N  # (d,d)
    if l2_reg:
        hess = hess + float(l2_reg) * np.eye(len(theta))

    return loss, grad, hess

def linear_objective_grad_only(theta, Xdiff, y, l2_reg: float = 0.0):
    loss, grad, _ = linear_objective(theta, Xdiff, y, l2_reg=l2_reg)
    return loss, grad

# %% [cell 25] (markdown)
"""
# Optimizers!
"""

# %% [cell 26] (markdown)
"""
## Step Length (strong wolfe)
"""

# %% [cell 27] (code)
def strong_wolfe_line_search(f, grad_f, theta, p,
                             c1=1e-4, c2=0.9,
                             alpha_max=50.0):
    """
    f: function returning loss
    grad_f: function returning gradient
    theta: current parameters (vector)
    p: descent direction (vector)
    """
    alpha0 = 0.0
    alpha1 = 1.0  # initial guess
    f0 = f(theta)
    g0 = grad_f(theta)
    d0 = np.dot(g0, p)   # directional derivative

    if d0 >= 0:
        raise ValueError("Direction p is not a descent direction.")

    f_prev = f0
    for i in range(20):  # limit iterations
        theta_new = theta + alpha1 * p
        f_new = f(theta_new)

        # Check Armijo condition
        if (f_new > f0 + c1 * alpha1 * d0) or (i > 0 and f_new >= f_prev):
            return zoom(f, grad_f, theta, p, alpha0, alpha1, f0, d0, c1, c2)

        g_new = grad_f(theta_new)
        d_new = np.dot(g_new, p)

        # Check strong Wolfe curvature condition
        if abs(d_new) <= -c2 * d0:
            return alpha1

        if d_new >= 0:
            return zoom(f, grad_f, theta, p, alpha1, alpha0, f0, d0, c1, c2)

        alpha0 = alpha1
        f_prev = f_new
        alpha1 = min(alpha1 * 2.0, alpha_max)  # increase step

    return alpha1

# %% [cell 29] (code)
def zoom(f, grad_f, theta, p, alo, ahi, f0, d0, c1, c2):
    """
    Zoom phase of Strong Wolfe line search.
    """
    for j in range(20):  # limit zoom iterations
        alpha = 0.5 * (alo + ahi)
        theta_new = theta + alpha * p
        f_new = f(theta_new)

        theta_lo = theta + alo * p
        f_lo = f(theta_lo)

        if (f_new > f0 + c1 * alpha * d0) or (f_new >= f_lo):
            ahi = alpha
        else:
            g_new = grad_f(theta_new)
            d_new = np.dot(g_new, p)
            if abs(d_new) <= -c2 * d0:
                return alpha
            if d_new * (ahi - alo) >= 0:
                ahi = alo
            alo = alpha

    return alpha

# %% [cell 30] (markdown)
"""
## Optimizers (linear reward model)

### First-Order
- Gradient Descent (fixed step)
- Gradient Descent + Wolfe Line Search

### Second-Order
- Newton + Line Search
- Trust Region (Cauchy point)

### Quasi-Newton
- BFGS
"""

# %% [cell 31] (code)
def gradient_descent(objective_fn, theta0, args=(), lr=1e-2, num_steps=100, log_every=10, verbose=True):
    theta = theta0.copy()
    history = []

    for k in range(num_steps):
        loss, grad = objective_fn(theta, *args)
        theta -= lr * grad
        history.append(loss)
        if verbose and (log_every > 0) and (k % log_every == 0):
            print(f"[GD step {k}] loss = {loss:.6f}")
    return theta, history

# %% [cell 32] (code)
def newton_method_linear(objective_fn, theta0, args=(), num_steps=20, log_every=1, verbose=True):
    theta = theta0.copy()
    history = []

    for k in range(num_steps):
        loss, grad, hess = objective_fn(theta, *args) # objective_fn should return loss, grad, hess
        history.append(loss)

        # netwon step
        try:
            step = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            print(f"[Newton] Hessian not invertible at step {k}, stopping.")
            break

        theta = theta + step

        if verbose and (log_every > 0) and (k % log_every == 0):
            step_norm = np.linalg.norm(step)
            grad_norm = np.linalg.norm(grad)
            print(f"[Newton step {k}] loss={loss:.6f}, |step|={step_norm:.3e}, |grad|={grad_norm:.3e}")

        # stopping
        if np.linalg.norm(grad) < 1e-6:
            print("Gradient small, stopping.")
            break

    return theta, history

# %% [cell 33] (code)
def bfgs(objective_fn, theta0, args=(), num_steps=100, log_every=10, verbose=True):
    theta = theta0.copy()
    n = len(theta)
    H_inv = np.eye(n)  # initial inverse Hessian approx

    loss, grad = objective_fn(theta, *args)
    history = [loss]

    for k in range(num_steps):
        # search direction
        p = -H_inv @ grad  # (n,)

        # simple backtracking line search
        alpha = 1.0
        c = 1e-4
        rho = 0.5

        while True:
            theta_new = theta + alpha * p
            loss_new, grad_new = objective_fn(theta_new, *args)
            if loss_new <= loss + c * alpha * grad.dot(p):
                break
            alpha *= rho
            if alpha < 1e-8:
                # step too small; give up
                theta_new, grad_new, loss_new = theta, grad, loss
                break

        s = theta_new - theta        # (n,)
        y_vec = grad_new - grad      # (n,)

        ys = y_vec.dot(s)
        if ys <= 1e-12:
            # curvature condition broken; reset H_inv
            H_inv = np.eye(n)
        else:
            rho_k = 1.0 / ys
            eye = np.eye(n)
            V = eye - rho_k * np.outer(s, y_vec)
            H_inv = V @ H_inv @ V.T + rho_k * np.outer(s, s)

        theta, grad, loss = theta_new, grad_new, loss_new
        history.append(loss)

        if verbose and (log_every > 0) and (k % log_every == 0):
            print(f"[BFGS step {k}] loss={loss:.6f}, alpha={alpha:.2e}")

        if np.linalg.norm(grad) < 1e-6:
            print("Gradient small, stopping BFGS.")
            break

    return theta, history

# %% [cell 34] (markdown)
"""
## Strong Wolfe line-search variants
"""

# %% [cell 35] (code)
def gd_strong_wolfe(objective_fn, theta0, args=(), num_steps=50, alpha_max=50.0, log_every=10, verbose=True):
    theta = theta0.copy()
    history = []

    # Wrappers for strong_wolfe_line_search
    def f_wrapper(current_theta):
        loss, _ = objective_fn(current_theta, *args)
        return loss

    def grad_f_wrapper(current_theta):
        _, grad = objective_fn(current_theta, *args)
        return grad

    for k in range(num_steps):
        g = grad_f_wrapper(theta)
        p = -g  # steepest descent direction

        # line search
        alpha = strong_wolfe_line_search(f_wrapper, grad_f_wrapper, theta, p, alpha_max=alpha_max)

        theta = theta + alpha * p
        loss = f_wrapper(theta)
        history.append(loss)

        if verbose and (log_every > 0) and (k % log_every == 0):
            print(f"[GD-Wolfe step {k}] loss={loss:.6f}, alpha={alpha}")

    return theta, history

# %% [cell 36] (code)
def newton_strong_wolfe(objective_fn, theta0, args=(), num_steps=20, alpha_max=50.0, log_every=1, verbose=True):
    theta = theta0.copy()
    history = []

    # Wrappers for strong_wolfe_line_search and internal usage
    def f_wrapper(current_theta):
        loss, _, _ = objective_fn(current_theta, *args)
        return loss

    def grad_f_wrapper(current_theta):
        _, grad, _ = objective_fn(current_theta, *args)
        return grad

    for k in range(num_steps):
        loss, g, H = objective_fn(theta, *args) # objective_fn must return loss, grad, hess
        history.append(loss)

        # Pure Newton direction: solve H p = -g
        try:
            p = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            print(f"[Newton-Wolfe] Hessian not invertible at step {k}, stopping.")
            break

        alpha = strong_wolfe_line_search(f_wrapper, grad_f_wrapper, theta, p, alpha_max=alpha_max)
        theta = theta + alpha * p

        if verbose and (log_every > 0) and (k % log_every == 0):
            print(f"[Newton-Wolfe step {k}] loss={loss:.6f}, alpha={alpha}")

        if np.linalg.norm(g) < 1e-6:
            print("Gradient small, stopping.")
            break

    return theta, history

def trust_region_cauchy(
    objective_fn,
    theta0,
    args=(),
    num_steps=50,
    delta0: float = 1.0,
    delta_max: float = 100.0,
    eta: float = 0.1,
    log_every: int = 10,
    verbose: bool = True,
):
    """
    Trust Region method using the Cauchy point.

    objective_fn must return (loss, grad, hess).
    """
    theta = theta0.copy()
    history = []
    delta = float(delta0)

    for k in range(num_steps):
        f, g, H = objective_fn(theta, *args)
        history.append(f)

        g_norm = float(np.linalg.norm(g))
        if g_norm < 1e-6:
            if verbose:
                print("Gradient small, stopping Trust Region.")
            break

        gHg = float(g @ (H @ g))
        # Cauchy step length along -g
        if gHg <= 0:
            tau = 1.0
        else:
            tau = min((g_norm**3) / (delta * gHg), 1.0)

        p = -(tau * delta / g_norm) * g  # cauchy point

        # Predicted reduction: m(0) - m(p) = -g^T p - 0.5 p^T H p
        pred = float(-(g @ p) - 0.5 * (p @ (H @ p)))
        if pred <= 0:
            # If model is not predicting improvement, fall back to steepest descent at the boundary.
            p = -(delta / g_norm) * g
            pred = float(-(g @ p) - 0.5 * (p @ (H @ p)))

        f_new, _, _ = objective_fn(theta + p, *args)
        ared = float(f - f_new)
        rho = ared / (pred + 1e-12)

        # Update trust region radius
        if rho < 0.25:
            delta = 0.25 * delta
        elif rho > 0.75 and abs(np.linalg.norm(p) - delta) < 1e-8:
            delta = min(2.0 * delta, delta_max)

        # Accept / reject
        if rho > eta:
            theta = theta + p

        if verbose and (log_every > 0) and (k % log_every == 0):
            print(f"[TR-Cauchy step {k}] loss={f:.6f}, rho={rho:.3f}, delta={delta:.3e}, |p|={np.linalg.norm(p):.3e}")

    return theta, history

# %% [cell 38] (markdown)
"""
# Optimizer helpers
"""

# %% [cell 39] (code)
def run_optimizer_record_losses(
    optimizer_fn,
    name,
    theta0,
    objective_fn_for_optimizer,
    objective_args_for_optimizer=(),
    num_steps=50,
    optimizer_kwargs=None,
):
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    # Track compute / "heft"
    counters = {"calls_total": 0, "calls_fg": 0, "calls_fgh": 0}

    def objective_counted(theta, *current_args):
        counters["calls_total"] += 1
        result = objective_fn_for_optimizer(theta, *current_args)
        if isinstance(result, tuple):
            if len(result) == 2:
                counters["calls_fg"] += 1
            elif len(result) == 3:
                counters["calls_fgh"] += 1
        return result

    # IMPORTANT:
    # We record ONE loss per outer optimizer iteration by using the optimizer's returned history.
    # (For methods with line search, the objective may be evaluated many times internally; we do not
    # count those extra evaluations as "steps".)
    t0 = time.perf_counter()
    theta_final, history = optimizer_fn(
        objective_counted,
        theta0.copy(),
        args=objective_args_for_optimizer,
        num_steps=num_steps,
        verbose=True,  # Changed to True for verbose logging
        **optimizer_kwargs,
    )
    elapsed_s = time.perf_counter() - t0

    return theta_final, history, {"time_s": elapsed_s, **counters}

# %% [cell 40] (code)
optimizers = {
    # First-Order
    "Gradient Descent (fixed step)": gradient_descent,
    "Gradient Descent + Wolfe Line Search": gd_strong_wolfe,
    # Second-Order
    "Newton + Line Search": newton_strong_wolfe,
    "Trust Region (Cauchy point)": trust_region_cauchy,
    # Quasi-Newton
    "BFGS": bfgs,
}

def run_linear_suite(cfg: Config, Xdiff_train: np.ndarray, train_labels: np.ndarray, verbose: bool) -> Dict[str, dict]:
    d = Xdiff_train.shape[1]
    theta0_lin = np.zeros(d, dtype=np.float64)

    results: Dict[str, dict] = {}
    for name, optfn in optimizers.items():
        print(f"Running {name}...")
        current_objective_fn = linear_objective_grad_only
        optimizer_kwargs = {}
        if name in ["Newton + Line Search", "Trust Region (Cauchy point)"]:
            current_objective_fn = linear_objective  # returns hessian
            if name == "Newton + Line Search":
                optimizer_kwargs["alpha_max"] = cfg.wolfe_alpha_max
        elif name == "Gradient Descent (fixed step)":
            optimizer_kwargs = {"lr": cfg.gd_lr}
        elif name == "Gradient Descent + Wolfe Line Search":
            optimizer_kwargs = {"alpha_max": cfg.wolfe_alpha_max}
        # logging cadence for all optimizers
        optimizer_kwargs["log_every"] = cfg.log_every

        theta_final, losses, bench = run_optimizer_record_losses(
            optfn,
            name,
            theta0_lin,
            objective_fn_for_optimizer=current_objective_fn,
            objective_args_for_optimizer=(Xdiff_train, train_labels, cfg.l2_reg),
            num_steps=cfg.optimizer_steps,
            optimizer_kwargs=optimizer_kwargs,
        )
        results[name] = {"theta": theta_final, "losses": losses, "bench": bench}

    return results


def eval_linear(theta: np.ndarray, Xdiff: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Returns (loss, accuracy) using the Bradley–Terry logistic loss WITHOUT regularization.
    """
    z = Xdiff @ theta
    return bradley_terry_loss(z, y), accuracy_from_logits(z, y)


def save_benchmarks_and_metrics(
    results: Dict[str, dict],
    Xdiff_train: np.ndarray,
    y_train: np.ndarray,
    Xdiff_val: np.ndarray,
    y_val: np.ndarray,
    prefix: str,
) -> None:
    """
    Saves:
      - {prefix}_summary.csv
      - {prefix}_runtime.png
      - {prefix}_objective_calls.png
      - {prefix}_val_metrics.png
      - {prefix}_scatter_time_vs_val_loss.png
    """
    import importlib

    # Compute final metrics
    rows = []
    for name, data in results.items():
        theta = data["theta"]
        train_loss, train_acc = eval_linear(theta, Xdiff_train, y_train)
        val_loss, val_acc = eval_linear(theta, Xdiff_val, y_val)
        bench = data.get("bench", {})
        rows.append(
            {
                "optimizer": name,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "time_s": float(bench.get("time_s", float("nan"))),
                "objective_calls_total": int(bench.get("calls_total", 0)),
                "objective_calls_fg": int(bench.get("calls_fg", 0)),
                "objective_calls_fgh": int(bench.get("calls_fgh", 0)),
                "steps_recorded": int(len(data.get("losses", []))),
            }
        )

    # Save CSV
    csv_path = f"{prefix}_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        headers = list(rows[0].keys()) if rows else []
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in headers) + "\n")
    print(f"Saved metrics CSV to: {csv_path}")

    # Plotting (optional if matplotlib exists)
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        print("Extra plots skipped: matplotlib is not available.")
        return

    names = [r["optimizer"] for r in rows]
    time_s = [r["time_s"] for r in rows]
    calls_total = [r["objective_calls_total"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    val_acc = [r["val_acc"] for r in rows]

    # Runtime bar chart
    plt.figure(figsize=(12, 5))
    plt.bar(names, time_s)
    plt.ylabel("Seconds")
    plt.title("Optimizer Runtime (wall-clock)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out = f"{prefix}_runtime.png"
    plt.savefig(out, dpi=200)
    print(f"Saved plot to: {out}")

    # Objective call bar chart
    plt.figure(figsize=(12, 5))
    plt.bar(names, calls_total)
    plt.ylabel("# objective function calls (includes line search)")
    plt.title("Compute Heft Proxy: Objective Calls")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out = f"{prefix}_objective_calls.png"
    plt.savefig(out, dpi=200)
    print(f"Saved plot to: {out}")

    # Validation metrics
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(names, val_loss)
    ax[0].set_title("Validation Loss (final)")
    ax[0].tick_params(axis="x", rotation=25)
    ax[1].bar(names, val_acc)
    ax[1].set_title("Validation Accuracy (final)")
    ax[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out = f"{prefix}_val_metrics.png"
    fig.savefig(out, dpi=200)
    print(f"Saved plot to: {out}")

    # Scatter: time vs val loss
    plt.figure(figsize=(7, 5))
    plt.scatter(time_s, val_loss)
    for i, nm in enumerate(names):
        plt.annotate(nm, (time_s[i], val_loss[i]), textcoords="offset points", xytext=(5, 3), fontsize=8)
    plt.xlabel("Time (s)")
    plt.ylabel("Validation loss (final)")
    plt.title("Speed vs Quality")
    plt.tight_layout()
    out = f"{prefix}_scatter_time_vs_val_loss.png"
    plt.savefig(out, dpi=200)
    print(f"Saved plot to: {out}")


def plot_results(results: Dict[str, dict], save_path: Optional[str] = None, show: bool = True, log_y: bool = True):
    import importlib

    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        print("Plotting skipped: matplotlib is not available.")
        print("Install it with: pip install matplotlib")
        if save_path:
            print(f"(Requested save path was {save_path!r})")
        return

    plt.figure(figsize=(12, 8))
    for name, data in results.items():
        plt.plot(data["losses"], label=name)
    plt.xlabel("Optimization Steps")
    plt.ylabel("Loss")
    plt.title("Optimization Loss Curves for Different Algorithms")
    plt.legend()
    plt.grid(True)
    if log_y:
        plt.yscale("log")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preference-model optimization demo (converted from notebook).")
    p.add_argument("--num-pairs", type=int, default=CFG.num_pairs)
    p.add_argument("--optimizer-steps", type=int, default=CFG.optimizer_steps)
    p.add_argument("--save-metrics-prefix", default=CFG.save_metrics_prefix, help="Where to save benchmark outputs (e.g. bench/run2).")
    p.add_argument("--log-every", type=int, default=CFG.log_every, help="Print progress every N optimizer steps (0 disables optimizer-step logging).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # Let users override cache; default to project-local cache for reproducibility.
    os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".hf_cache"))
    cfg = Config(
        num_pairs=args.num_pairs,
        optimizer_steps=args.optimizer_steps,
        save_metrics_prefix=args.save_metrics_prefix,
        log_every=int(args.log_every),
    )
    verbose = True

    # Load dataset split only (less memory than downloading all splits).
    ds = load_dataset(cfg.dataset_name, split=cfg.split)
    if verbose:
        print(ds)
        print(ds[0].keys())
        ex = ds[1]
        print("====Prompt:=====\n", ex.get("prompt", ""))
        print("\n======Chosen:=====\n", ex.get("chosen", ""))
        print("\n=====Rejected:=====\n", ex.get("rejected", ""))

    chosen, rejected, labels = build_pairs(ds, cfg.num_pairs)
    y = np.asarray(labels, dtype=np.float64)
    print(f"Built {len(chosen)} pairs")

    embed_model = SentenceTransformer(cfg.embed_model_name)
    emb_chosen = embed_texts(embed_model, chosen, batch_size=cfg.batch_size, normalize_embeddings=cfg.normalize_embeddings)
    emb_rejected = embed_texts(embed_model, rejected, batch_size=cfg.batch_size, normalize_embeddings=cfg.normalize_embeddings)
    if verbose:
        print("emb_chosen shape:", emb_chosen.shape)
        print("emb_rejected shape:", emb_rejected.shape)

    train_A, train_B, train_y, val_A, val_B, val_y = train_val_split(
        emb_chosen, emb_rejected, y, train_frac=cfg.train_frac, seed=cfg.seed
    )
    print("Train split:", len(train_A))
    print("Validation split:", len(val_A))

    Xdiff_train = train_A - train_B
    Xdiff_val = val_A - val_B
    scaler = fit_xdiff_scaler(Xdiff_train, method=cfg.xdiff_scale)
    Xdiff_train = scaler.transform(Xdiff_train) * float(cfg.xdiff_mult)
    Xdiff_val = scaler.transform(Xdiff_val) * float(cfg.xdiff_mult)

    results = run_linear_suite(cfg, Xdiff_train, train_y, verbose=verbose)
    print("Done.")

    # Save loss curves and benchmarks under the provided prefix.
    # Example prefix: "bench/run2" -> saves "bench/run2_losses.png" and other files.
    prefix = cfg.save_metrics_prefix
    parent = os.path.dirname(prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plot_results(results, save_path=f"{prefix}_losses.png", show=False, log_y=True)
    save_benchmarks_and_metrics(results, Xdiff_train, train_y, Xdiff_val, val_y, prefix=prefix)


if __name__ == "__main__":
    main()

