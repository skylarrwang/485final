# Running `optimization_final.py`

### Setup

```bash
python3.13 -m venv .venv || python3.12 -m venv .venv || python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### Run

```bash
python optimization_final.py --save-metrics-prefix bench/run1
```

### Outputs (for prefix `bench/run1`)

- **Loss curves**: `bench/run1_losses.png`
- **CSV summary**: `bench/run1_summary.csv`
- **Benchmark plots**:
  - `bench/run1_runtime.png`
  - `bench/run1_objective_calls.png`
  - `bench/run1_val_metrics.png`
  - `bench/run1_scatter_time_vs_val_loss.png`

### CLI flags

- **`--num-pairs`** (default `1000`): number of preference pairs to build
- **`--optimizer-steps`** (default `50`): steps per optimizer
- **`--save-metrics-prefix`** (default `bench/run1`): output prefix (also chooses the loss-plot filename)
- **`--log-every`** (default `10`): print every N optimizer steps (`0` = no step logs)

### Examples

Test to make sure everything's working:

```bash
python optimization_final.py --num-pairs 200 --optimizer-steps 20 --save-metrics-prefix bench/lightweight-test
```

Bigger run:

```bash
python optimization_final.py --num-pairs 1000 --optimizer-steps 50 --save-metrics-prefix bench/run2
```

### Notes

- First run downloads the dataset/model (needs internet).
- `Anthropic/hh-rlhf` may require accepting terms / `huggingface-cli login`.


