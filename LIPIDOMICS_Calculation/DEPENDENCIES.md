# Dependencies

Software environment used to produce the plots and statistics in this repository
(`MichaelisMenten_MMCalculation` and `LIPIDOMICS_Calculation`).

The published figures were generated in **MATLAB R2024a**. The version numbers
below are the ones to cite in a methods section.

## Core environment

| Component | Version |
|-----------|---------|
| MATLAB | 24.1.0.2537033 (R2024a) |
| Optimization Toolbox | 24.1 (R2024a) |
| Statistics and Machine Learning Toolbox | 24.1 (R2024a) |

> Since R2023b, MathWorks assigns every product the release-based version number,
> so on any R2024a install all three lines above read `Version 24.1 (R2024a)`.
> The full 4-part build (`24.1.0.2537033`) applies to MATLAB itself and is recorded
> in the `.mlx` file metadata.

## Third-party function (bundled)

| File | Version | Source |
|------|---------|--------|
| `swtest.m` | Rev. 3.0 (18 Jun 2014) | Ahmed Ben Saïda, MATLAB Central File Exchange — Shapiro–Wilk / Shapiro–Francia normality test |

`swtest.m` internally calls `norminv`, `kurtosis`, and `normcdf`, so it also
requires the Statistics and Machine Learning Toolbox.

## Which toolbox each script uses

### MichaelisMenten_MMCalculation
- `MMcalculation.mlx` — reads `MichaelisMenten.xlsx`
  - Optimization Toolbox: `lsqcurvefit`, `optimoptions` (Michaelis–Menten fit)
  - Statistics and Machine Learning Toolbox: `tinv` (95% CIs), `qqplot` (residual diagnostics)
  - base MATLAB: `readmatrix`, `polyfit`, `polyval`, `plot`, `subplot`, `histogram`

### LIPIDOMICS_Calculation
- `LipidmapStatTests*.mlx`, `Ma2025_Lipidomics_script_matlab.mlx`, `script_ZG_W.m`
  - reads `LipidMapsInvData.xlsx`
  - Statistics and Machine Learning Toolbox: `ttest2`, `ranksum`, `vartest2`, `kstest`
  - third-party: `swtest` (Shapiro–Wilk)
  - base MATLAB: `readtable`, `writetable`, `table`, `normalize`, `heatmap`, `parula`

## Notes
- All functions used are available in R2022b–R2024a; the code runs on older
  installs, but the toolbox version numbers differ from those above (pre-R2023b
  releases number each toolbox independently). Report the versions of whichever
  MATLAB actually produced the figures — R2024a for the plots in this repository.
- To confirm on the machine that made the figures, run `ver` in MATLAB.
