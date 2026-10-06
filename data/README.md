# Data

Large data products are intentionally excluded from version control.

## Synthetic data

Generate clean synthetic examples with:

```bash
python src/data/generate_synthetic.py
```

The canonical setup uses 50,000 training, 5,000 validation, and 1,000 test examples on a 64 x 64 grid. The realistic degradation code is:

```bash
python src/data/add_realistic_noise.py
```

Before running these scripts on a new machine, inspect their CLI/path arguments because these files are preserved from the research pipeline rather than rewritten solely for packaging.

## Real data

The study evaluates three earthquakes:

- 2023 Jishishan
- 2019 Albania
- 2023 Morocco

The unified evaluator expects preprocessed 64 x 64 dual-track arrays containing the observations/masks, physical `X_km` and `Y_km` grids, and pixel-wise LOS east/north/up components.

Raw and preprocessed real products are not redistributed by this repository. Obtain the underlying InSAR products from their original providers and respect their licenses/citation requirements.

For the author's local regression test, `verify_repository.py` points to the original research archive under `~/Documents/PIML`. That verification path is intentionally separate from the public repository data placeholders.

## Suggested local layout

```text
data/
|-- synthetic/
|   `-- <generated arrays>
`-- real/
    |-- jishishan/
    |-- albania/
    `-- morocco/
```

Do not commit large `.npz`, `.npy`, GeoTIFF, or checkpoint files to the normal Git history.
