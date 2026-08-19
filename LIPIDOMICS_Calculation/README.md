# Lipidomics tissue analysis

This folder contains the complete MATLAB pipeline for tissue lipidomics analysis. No files outside this folder are required.

## Requirements

- MATLAB R2024b or later
- Statistics and Machine Learning Toolbox

`swtest.m` is included for the Shapiro-Wilk normality test.

## Sample order

The `LipidMapsSD data` worksheet in `LipidMapsInvData.xlsx` is interpreted as follows:

- Columns 2-7 (`WT_F1`, `WT_F2`, `WT_F3`, `WT_M1`, `WT_M2`, `WT_M3`): wild type; three female samples followed by three male samples
- Columns 8-13 (`KO_F1`, `KO_F2`, `KO_F3`, `KO_M1`, `KO_M2`, `KO_M3`): knockout; three female samples followed by three male samples

The `Sample_Metadata` worksheet in the same workbook provides the explicit mapping from each data column and original header to sample ID, genotype, sex, and replicate number. This worksheet is the documented source of the sex assignments used by the MATLAB analysis.

## Run order

1. Open and run `LipidmapStatTests.mlx`.
2. Open and run `LipidmapStatTests_Plot3.mlx`.

Open the live scripts from this package folder. If MATLAB executes a Live Editor temporary copy, each script checks the active editor location, the MATLAB Current Folder and its parent folders, and valid package folders below the Current Folder. A folder is accepted only when it contains `LipidMapsInvData.xlsx`, `swtest.m`, and both required workbook worksheets (`LipidMapsSD data` and `Sample_Metadata`). Keep the MATLAB Current Folder at this package folder or at a parent folder such as `D:\Desktop\Matlab YNM`.

The statistical tables remain available in the MATLAB workspace as `results`, `results_with_fold_change`, `sex_results`, and `sample_metadata`. The scripts write the complete historical result set, updated with sex statistics and sample metadata, to `Outputs`. The plotting script writes the PDF and PNG figures to the same folder.

## Analysis

For each lipid, the primary genotype comparison uses Welch's t-test when both groups pass the Shapiro-Wilk normality test; otherwise, the Mann-Whitney U test is used. This is the comparison reported in `LipidStatsResults.xlsx`, `LipidStatsResultsWithShapiroW2.xlsx`, and `LipidStatsResultsMod.xlsx`, and it is the comparison used to select lipids for the heatmap. The first script also reports Kolmogorov-Smirnov Gaussian-distribution p-values and Shapiro-Wilk W statistics. For historical compatibility, `LipidStatsResultsWithGaussian.xlsx` retains the earlier variance-selected branch, which uses an equal-variance t-test or Welch's t-test when both groups pass normality.

### Sex-effect analysis

Sex is analyzed separately for each lipid while preserving the WT and KO genotype groups. This pipeline uses a shifted log2 transformation as a prespecified, pragmatic choice; it is not presented as a universally preferred transformation. Log transformation is commonly used for positive metabolomics and lipidomics measurements because it expresses multiplicative differences as additive contrasts and can reduce right skew and mean-dependent variance. Its suitability depends on the measurement platform, abundance range, treatment of zeros, biological question, and statistical model.

Before testing, a lipid-specific pseudocount equal to one half of the smallest positive concentration is added to all 12 measurements, and the concentrations are transformed to log2 scale. The shift permits analysis when a measured concentration is zero. Because transformations near zero can be sensitive to the selected pseudocount, the transformation and pseudocount are reported explicitly rather than treated as universal defaults. The exact permutation test does not require normally distributed residuals; here, the transformation primarily defines a relative effect scale and reduces the influence of very large concentrations.

The sex effect is calculated from two within-genotype contrasts:

- WT sex contrast = mean log2 concentration in WT males minus mean log2 concentration in WT females
- KO sex contrast = mean log2 concentration in KO males minus mean log2 concentration in KO females
- Overall sex effect = average of the WT and KO sex contrasts

The overall effect is reported as `MaleMinusFemale_Log2`. A positive value indicates higher concentration in males, and a negative value indicates higher concentration in females. For example, an effect of `1` corresponds to an approximately twofold male-to-female difference on the log2 scale. The raw-scale group means are reported separately as `WT_FemaleMean`, `WT_MaleMean`, `KO_FemaleMean`, and `KO_MaleMean`.

An exact balanced permutation test is used because each genotype contains only three females and three males. Within WT, every possible assignment of three of the six samples as female is evaluated (`20` assignments). The same procedure is applied within KO, producing `20 x 20 = 400` combined assignments. Genotype membership and the three-versus-three balance are retained in every permutation. The two-sided sex p-value is the proportion of these 400 assignments whose absolute overall sex effect is at least as large as the observed effect.

The genotype-by-sex interaction tests whether the male-minus-female difference changes between genotypes:

`interaction = KO sex contrast - WT sex contrast`

A positive interaction means the male-minus-female contrast is greater in KO than in WT; a negative interaction means it is smaller or reversed. Its two-sided exact permutation p-value is calculated from the same 400 balanced assignments.

Benjamini-Hochberg false-discovery-rate correction is applied across all lipids separately for the sex-effect p-values and interaction p-values. Results are considered significant when the corresponding `SexQ_BH` or `InteractionQ_BH` is below `0.05`. Sample assignments are documented in the `Sample_Metadata` worksheet of every statistical result workbook.

### Transformation sensitivity check

The transformation choice was checked directly using this dataset. Six of 84 lipids contained at least one zero, corresponding to 25 zero measurements among 1,008 values. The exact sex analysis was repeated using four specifications: untransformed concentrations, square-root-transformed concentrations, log2 transformation with half-minimum replacement only for nonpositive values, and the shifted log2 transformation used in the scripts.

All four specifications produced the same overall inferential counts: 20 lipids had nominal sex-effect `p < 0.05`, no lipid had sex-effect `q < 0.05`, one lipid had nominal interaction `p < 0.05`, and no lipid had interaction `q < 0.05`. The minimum sex-effect q-value was `0.0525` under every specification. Individual effect estimates and p-values were not identical, but the multiple-testing-adjusted conclusion was unchanged.

The appropriate interpretation is therefore that **no sex effect or genotype-by-sex interaction met the prespecified BH-adjusted threshold in this dataset**. This should not be stated as proof that male and female lipidomes are equivalent. The design has three animals per sex within each genotype, and several nominal sex associations were present; consequently, the analysis has limited resolution for small or variable sex effects.

### Method references

- Goodacre R, et al. [Proposed minimum reporting standards for data analysis in metabolomics](https://doi.org/10.1007/s11306-007-0081-3). *Metabolomics*. 2007;3:231-241.
- van den Berg RA, et al. [Centering, scaling, and transformations: improving the biological information content of metabolomics data](https://doi.org/10.1186/1471-2164-7-142). *BMC Genomics*. 2006;7:142.
- Di Guida R, et al. [Non-targeted UHPLC-MS metabolomic data processing methods: a comparative investigation of normalisation, missing value imputation, transformation and scaling](https://pubmed.ncbi.nlm.nih.gov/27123000/). *Metabolomics*. 2016;12:93.
- Hughes G, et al. [MSPrep—summarization, normalization and diagnostics for processing of mass spectrometry-based metabolomic data](https://doi.org/10.1093/bioinformatics/btt589). *Bioinformatics*. 2014;30:133-134.
- Beyene HB, et al. [High-coverage plasma lipidomics reveals novel sex-specific lipidomic fingerprints of age and BMI](https://doi.org/10.1371/journal.pbio.3000870). *PLoS Biology*. 2020;18:e3000870.

## Outputs

- `LipidStatsResults.xlsx`: the historical combined genotype table, formerly named with the study year, containing the selected test, genotype P value, Gaussian and Shapiro-Wilk diagnostics, Shapiro-Wilk W statistics, and fold change; it also contains sex statistics and sample metadata
- `LipidStatsResultsMod.xlsx`: genotype statistics used to select and order the plotted lipids, including group means and fold change; it also contains sex statistics and sample metadata
- `LipidStatsResultsWithGaussian.xlsx`: the historical Gaussian and Shapiro-Wilk diagnostic table using the earlier variance-selected t-test branch, with sex statistics and sample metadata
- `LipidStatsResultsWithShapiroW2.xlsx`: the historical Shapiro-Wilk W diagnostic table, with sex statistics and sample metadata
- `Lipidomics_Heatmap.pdf` and `.png`: publication heatmap
- `Lipidomics_Heatmap_Values.pdf` and `.png`: the same heatmap with row-normalized z-scores printed inside the cells

`LipidStatsResultsWithGaussian.xlsx` and `LipidStatsResultsWithShapiroW2.xlsx` intentionally retain the same diagnostic columns so that either historical filename remains available to downstream workflows. Their `TestType` and `PValue` columns can differ for normally distributed lipids because the Gaussian workbook preserves the older variance-selected comparison, whereas the ShapiroW2 workbook follows the primary Welch pipeline. No year is included in the regenerated filenames.

The heatmap color scale runs from high values on the left of the legend to low values on the right. Fold change is knockout mean divided by wild-type mean.
