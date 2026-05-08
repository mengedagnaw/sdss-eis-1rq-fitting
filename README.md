# EIS 1RQ Weighting Comparison

Python workflow for fitting electrochemical impedance spectroscopy (EIS) data using a single-time-constant equivalent circuit model.

## Equivalent circuit

```text
Rs - (Q || R)
```

where:

- `Rs` is the solution resistance.
- `R` is the polarization or charge-transfer resistance.
- `Q` is the constant phase element (CPE).
- `(Q || R)` is the parallel CPE–resistance branch.

## Purpose

This repository compares different weighting strategies for fitting EIS spectra of standard super duplex stainless steel 2507 using a 1RQ equivalent circuit model.

The workflow includes:

- Nyquist and Bode visualization
- nonlinear least-squares EIS fitting
- comparison of modulus, separate-component, and unweighted residual schemes
- residual analysis
- CSV export of fitted parameters

## Repository structure

```text
sdss-standard-eis-1rq-fitting/
├── README.md
├── LICENSE
├── scripts/
│   ├── weighting_combined.py
│   └── weighting_combined.ipynb
└── figures/
    ├── ECM_square_clean.png
    ├── AS_Combined_EIS.png
    ├── AS_Residuals.png
    └── ...
```

## Notes

- A good numerical fit does not prove that the selected equivalent circuit is mechanistically unique.
- Fitted parameters should be interpreted together with residual structure and physical plausibility.
- Raw or unpublished EIS data should not be uploaded unless public release is intended.

## License

This repository is licensed under GPL-3.0.
