# Experiments

These are thin experiment entry points; the scientific implementation lives in `src/`.

## Frozen 50k baseline

```bash
python experiments/run_baseline_50k.py
```

The 50k model is the primary baseline; 250k is only a scaling ablation.

## PIML physics-weight sweep

```bash
python experiments/run_piml_sweep.py
```

This records the reported weights: 0, 0.01, 0.1, and 1.0.

## Real events

For explicit checkpoint/model selection:

```bash
python src/evaluate_real.py --event all --model baseline
python src/evaluate_real.py --event all --model piml --lambda-phys 0.1
```

`results/summary/` contains the committed reference outputs. Large synthetic datasets
and trained checkpoints remain excluded from Git.
