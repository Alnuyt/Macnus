# Macnus: magneto-Coriolis modes in a rotating cylinder

This archive contains one executable notebook and the small Python package it
uses to compute magneto-Coriolis eigenmodes in a finite rotating cylinder.

## Contents

```text
Macnus_Zenodo/
├── Macnus.ipynb
├── mc_cylinder/
├── CITATION.cff
├── environment.yml
└── README.md
```

`mc_cylinder/` must remain beside the notebook: it contains the symbolic
operators, matrix assembly, field reconstruction, analytical benchmarks, and
validation routines imported by `Macnus.ipynb`.

## Run the notebook

From this directory, create the supplied Conda environment:

```bash
conda env create -f environment.yml
conda activate mc-cylinder
```

Register the corresponding Jupyter kernel once:

```bash
python -m ipykernel install --user \
  --name mc-cylinder \
  --display-name "Python (mc-cylinder)"
```

Then start the notebook:

```bash
jupyter lab Macnus.ipynb
```

Select the kernel **Python (mc-cylinder)** and run the cells from top to
bottom. No package installation and no local path modification are required.
The figures produced by the notebook are written to the local `outputs/`
directory.

## Model parameters

- `Gamma = h/a`: cylinder aspect ratio;
- `Le = B0 / (sqrt(rho*mu0) * Omega * a)`: Lehnert number;
- `E = nu / (Omega*a**2)`: Ekman number;
- `Em = eta / (Omega*a**2)`: magnetic Ekman number.

The time convention is `exp(i*m*phi - i*omega*t)`, so a damped mode satisfies
`Im(omega) < 0`.

## Citation

Please cite the archived release using `CITATION.cff` and its Zenodo DOI : https://doi.org/10.5281/zenodo.23191256

## License MIT
