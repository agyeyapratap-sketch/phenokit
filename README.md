# phenokit

[![tests](https://github.com/agyeyapratap-sketch/phenokit/actions/workflows/tests.yml/badge.svg)](https://github.com/agyeyapratap-sketch/phenokit/actions/workflows/tests.yml)

**Sensor-to-insight toolkit for field crop phenotyping on handheld, ground-based and aerial (UAV) platforms.**

phenokit covers the whole chain a phenotyping node runs for every trial: plan the acquisition,
manage the data, check its integrity and quality, calibrate it, extract plot-level traits with
computer vision and machine learning, and validate those traits against field measurements
across crop species and platforms. It is written in Python with light dependencies: NumPy, SciPy,
pandas, scikit-learn, Matplotlib, Pillow and PyYAML, plus optional PyTorch. MATLAB interop
functions are included.

| Stage | Module | What it does |
|---|---|---|
| Acquisition | `acquisition` | Sensor/platform presets; GSD, footprint, overlap-driven line spacing, trigger interval, motion blur; LiDAR swath/density; ground-vehicle transects; NOAA solar position and acquisition windows; SOP checklists (`validate_acquisition`). |
| Data management | `catalog`, `io` | Standard layout `<trial>/<date>/<platform>_<sensor>/<level>`; MIAPPE-style `metadata.json`; SHA-256 manifests (`build_manifest`/`verify_manifest`); SQLite catalogue with QC results and provenance. Readers/writers: ENVI, `.mat`, PNG/JPEG + world files, LAS 1.2 (pure NumPy), spectroradiometer CSV. |
| Quality control | `qc` | RGB sharpness/exposure/colour cast/noise, perceptual-hash duplicates; hyperspectral SNR by spectral–spatial decorrelation, bad/absorption bands, impulse noise, reflectance plausibility, independent panel check; spectroradiometer detector-join steps; thermal invalid pixels and cover-adjusted vignetting; LiDAR density gaps, isolated points, duplicates; GNSS gaps/RTK fix/speed; train/test leakage, label agreement, plot-table outliers; HTML/CSV reports. |
| Calibration | `calibration` | Empirical line (with leave-one-panel-out check), dark/white reference, flat-field and IARR when no panels exist, DLS normalisation, thermal emissivity/reflected-temperature correction, splice correction. |
| Spectral | `spectral`, `indices` | Savitzky–Golay, SNV, MSC, derivatives, continuum removal, Gaussian SRF resampling (simulate a multispectral camera from hyperspectral data), PCA. Index library resolved **by wavelength** (and FWHM), so it refuses to compute an index the sensor cannot support. |
| Vision & 3-D | `segmentation`, `plots`, `pointcloud`, `thermal` | ExG/ExGR/k-means/NDVI + Otsu, object features; plot layouts and zonal statistics; voxel downsampling, outlier removal, RANSAC soil plane, DTM/CHM, plot height metrics, single-plant 3-D traits; canopy temperature and CWSI. |
| AI & validation | `ml`, `deep`, `validation` | Spatially blocked and leave-one-group-out CV, PLSR with 1-SE component selection and VIP, transfer matrices; 1-D CNN and compact U-Net (PyTorch); R², RMSE, CCC, RPD, Bland–Altman, per-species validation, cross-platform agreement, broad-sense heritability. |
| Orchestration | `pipeline`, `cli` | YAML-configured multi-platform pipeline with QC gating, provenance and an HTML report; `phenokit` command-line tool. |
| Test-beds | `datasets`, `synthetic` | Checksum-pinned downloaders for public datasets (including single-file extraction from a 12.7 GB remote zip via HTTP range requests); a synthetic five-crop, multi-platform trial with known truth and 13 injected faults. |

## Install

```bash
git clone https://github.com/agyeyapratap-sketch/phenokit.git && cd phenokit
pip install -e ".[dev]"          # core + pytest
pip install -e ".[deep]"         # + PyTorch for the 1-D CNN and U-Net
python -m pytest                 # 29 unit tests
```

## Tutorials (real public data + a synthetic trial)

| # | Script | Data | Topic |
|---|---|---|---|
| 00 | `t00_acquisition_planning.py` | — | Solar windows at a WA wheatbelt latitude, per-sensor UAV flight plans, ground-vehicle transects, protocol validation |
| 01 | `t01_hyperspectral_qc_and_leakage_indian_pines.py` | AVIRIS Indian Pines | Hyperspectral QC; RF/SVM/1-D CNN crop classification; random vs spatially blocked CV |
| 02 | `t02_growth_stage_phenotyping_salinas.py` | AVIRIS Salinas | Relative reflectance without panels; index trajectories over lettuce growth stages; PLSR + VIP; confounded designs |
| 03 | `t03_crop_weed_vision_cwfid.py` | CWFID (field robot) | Dataset integrity QC (finds a train/test leak in the official split); vegetation segmentation; crop/weed object RF vs U-Net |
| 04 | `t04_3d_maize_growth_pheno4d.py` | Pheno4D | Remote single-file access; point-cloud QC; 3-D traits; automatic vs annotated height; growth curve |
| 05 | `t05_end_to_end_multiplatform_trial.py` | Synthetic WA trial | Full pipeline: 13/13 injected faults detected, calibration, traits, cross-species transfer, cross-platform NDVI, heritability |

```bash
cd tutorials
python t00_acquisition_planning.py
python t01_hyperspectral_qc_and_leakage_indian_pines.py   # ~5 min on CPU (add --no-cnn to skip the CNN)
...
```

Data are downloaded to `data/public/<dataset>/` on first use and verified against pinned SHA-256
values. Figures are written to `docs/chapter/figures/` and numeric results to `outputs/tutorial_results/`
(both are generated locally and not tracked in git).
Each dataset has its own licence (see `phenokit datasets list`). CWFID is for non-commercial research only.

## Command line

```bash
phenokit datasets list
phenokit datasets fetch indian_pines salinas
phenokit synth data/synthetic
phenokit run configs/synthetic_trial.yaml --out outputs/run     # HTML QC report + plot_features.csv
phenokit plan --sensor ms_5band --width 34 --length 66 --altitude 40
phenokit sun --lat -31.65 --lon 116.67 --date 2026-08-20 --utc-offset 8
phenokit manifest build  /data/TRIAL-2026 ; phenokit manifest verify /data/TRIAL-2026
phenokit qc rgb frames/*.png --out frames_qc.html --strict
```

## Using it on a real trial

1. Copy `configs/synthetic_trial.yaml` and change `trial_root`, `date`, file names and thresholds.
2. Provide `plot_layout.csv` (`plot_id, xmin, xmax, ymin, ymax` in the raster CRS, plus design columns)
   and the field-measurement sheet (`plot_id, species, genotype, rep, <traits>`).
3. Run `phenokit manifest build` immediately after data offload, then `phenokit run`.

## MATLAB

`matlab/phenokit_read_envi.m`, `phenokit_index.m` and `phenokit_verify_manifest.m` read the same
products, compute the same indices and verify the same manifests in base MATLAB.

## Project layout

```
src/phenokit/      library            tests/        unit tests
tutorials/         worked examples    configs/      pipeline configs
matlab/            interop            .github/      CI workflow
```

## Notes on limits

* The synthetic reflectance model is phenomenological (not a radiative-transfer model): use it to test
  pipelines, not to draw biological conclusions.
* AVIRIS wavelengths in Tutorials 01–02 are approximate (linear spacing). Use the original band
  tables for quantitative spectroscopy.
* Pheno4D label semantics are inferred from geometry (0 = soil, 1 = stem, ≥2 = leaves). Check them
  against the dataset documentation.

## Development and use of AI tools

The initial release (v0.1.0) was drafted largely by a generative AI coding assistant (Anthropic's Claude)
under the author's direction, including most of the code, tests and tutorials. The author is reviewing it and
is responsible for its maintenance and for any results produced with it. Later changes are recorded in
[CHANGELOG.md](CHANGELOG.md). Contributions are reviewed by a human before
merging; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributing and support

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

Licence: MIT (code). Datasets keep their own licences.
