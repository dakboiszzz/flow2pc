# Dữ liệu dùng chung

Tất cả dữ liệu local nằm trực tiếp dưới `data/`, sắp xếp theo bộ dữ liệu thay vì phương pháp: `single_cell/`, `spatial_transcriptomics/`, `cell_tracking/`, `knot/`, `gom/`, `beijing/` và `synthetic/`. ALI-CFM, OTP-FM và MFM dùng chung các file nguồn này.

`mfm/` chứa catalog, liên kết tới dữ liệu nguồn và các snapshot MFM sau khi chạy lệnh `prepare`; hiện chưa có snapshot. LiDAR vẫn nằm ở checkout MFM bên ngoài và được liên kết tại `mfm/sources/lidar`. Output train nằm ngoài `data/`. Đường dẫn `data/...` trong tài liệu được hiểu từ gốc repository.

## Nguyên tắc dùng dữ liệu

Mỗi bộ dữ liệu cung cấp các marginal quan sát theo thời gian. Khi so sánh các model, giữ cố định số mẫu, seed, time mapping, preprocessing, evaluation samples và quy tắc chọn checkpoint. Báo cáo kết quả tại từng thời điểm và kết quả tổng hợp; chỉ so sánh các kết quả có cùng đơn vị tọa độ và observation budget.

| Dataset | Bài toán |
| --- | --- |
| `single_cell/` | Dự đoán phân phối trạng thái tế bào theo thời gian cho EB, CITE-seq và Multiome. |
| `spatial_transcriptomics/` | Tiến hóa vị trí tế bào u qua bốn slide HT206B1 U2–U5 đã căn chỉnh. |
| `cell_tracking/` | Chuyển động cell 4 trong chuỗi PhC-C2DH-U373, sequence 01. |
| `knot/` | Tiến hóa vòng lặp tổng hợp với đường cong giải tích làm tham chiếu. |
| `gom/`, `beijing/` | Các marginal vận chuyển thực với mốc thời gian holdout. |
| `synthetic/` | Các marginal Gaussian tổng hợp đã xuất ra NPZ. |

Các mục bên dưới giữ chi tiết nguồn gốc, tọa độ, thứ tự dữ liệu và protocol của từng dataset. `SHA256SUMS_ALI_CFM.txt` chứa checksum của các file nguồn ALI-CFM còn giữ lại sau khi gộp tài liệu. `manifest.json` và `source.json` trong các thư mục liên quan ghi checksum dữ liệu và provenance. Các NPZ portable tải bằng `np.load(path, allow_pickle=False)`. Nếu chuyển `data/` sang máy khác, cập nhật `data_dir` tuyệt đối trong các JSON cấu hình vận chuyển.

## Single-cell: EB, CITE-seq, Multiome

### Single-cell time-series experiments

#### Experiments and data

Run time-dependent distribution prediction on each of the paper's three
single-cell datasets:

| File | Dataset |
|---|---|
| `ebdata_v3.h5ad` | Embryoid Body (EB) |
| `op_cite_inputs_0.h5ad` | CITE-seq (CITE) |
| `op_train_multi_targets_0.h5ad` | Multiome (Multi) |

These processed H5AD files are from the paper's [Mendeley data release](https://data.mendeley.com/datasets/hhny5ff7yj/1)
(DOI `10.17632/hhny5ff7yj.1`); the paper repository lists all three as the
single-cell experiment inputs. Treat each annotated collection of cells at a
time point as an empirical marginal and model evolution between those
marginals.

#### Data interpretation and comparison

For CITE and Multi, the prepared inputs use `obsm['X_pca']` and `obs['day']`;
PCA is supplied and should not be recomputed when matching this experiment. The
documented day labels are 2, 3, 4, and 7, mapped to equally spaced pseudo-times
0, 1/3, 2/3, and 1. Preserve the dataset's cell counts per day and use the same
PCA dimensions and time mapping across models. Report predictions and metrics
per time point, including intermediate/held-out times where the chosen protocol
holds them out.

EB is part of the same paper experiment and is included as its processed H5AD.
Read its own time labels and representation metadata from the file; do not
assume its annotation schema or time mapping is identical to CITE or Multi.
Across all three, record any dimensionality reduction, normalization, split,
sampling, and evaluation choices, and do not modify the source H5AD files in
place.

All three files were copied byte-for-byte from
`/data/lthung/ali-cfm/single_cell/data/`. Verify them with the root
`SHA256SUMS_ALI_CFM.txt` manifest.

## Spatial transcriptomics

### Spatial transcriptomics preprocessing

The preprocessing scripts removed from the current checkout are restored from
commit `c2cad56`. The paper's Appendix C specifies BigWarp thin-plate-spline
registration of U2, U3, and U4 to U5, with images downscaled by 10 before
registration. The Git history contains the scripts, but not their input image
exports or the `landmarks/landmarks_*moving_*target.csv` files.

The supplied Seurat RDS files already contain spot coordinates, tumour purity,
and a low-resolution H&E image. Extract them with:

```bash
Rscript data/ST_images/extract_seurat_st.R data/st data/ST_images/extracted
```

This produces one image and one spot table per slide, plus a tumour-only table
using the paper's strict `Tumor_purity_estimate > 0.8` threshold. The spot table
contains `imagecol` and `imagerow` in the original image coordinate system;
divide these by 10 before matching landmarks measured on the paper's 1/10 scale
images. Tumour-only output is explicitly named `*_unaligned.csv` and must not
be used as model input until its coordinates are transformed to U5.

After supplying the three pairwise landmark CSVs, run the current SciPy based
alignment implementation:

```bash
python data/ST_images/align_extracted_st.py
```

It writes the aligned H&E images and tumour coordinate CSVs under
`data/ST_images/aligned/`. Each landmark CSV has the historical six-column
layout: source x/y in columns 3/4 and target x/y in columns 5/6 (one-based).
The three required files are `landmarks_U2moving_U3target.csv`,
`landmarks_U3moving_U4target.csv`, and `landmarks_U4moving_U5target.csv` in
`data/ST_images/landmarks/`. They must encode the manual BigWarp landmarks from
the paper or a newly performed visual alignment. The old OpenCV script is kept
as a historical reference; the current RDS exporter does not invent missing
landmarks or treat unregistered coordinates as paper-aligned data.

## Spatial transcriptomics: thí nghiệm

### Spatial transcriptomics experiment: HT206B1 U2–U5

#### Experiment

Model the evolution of tumour-cell spatial distributions across the four
observed slides U2, U3, U4, and U5. Use the prepared tumour-coordinate tables
in `aligned/aligned_spots/`; these are the available preprocessed observations
for this experiment. They have already been aligned into the common U5
reference frame and filtered using `Tumor_purity_estimate > 0.8`.

The time points are equally spaced: U2=0, U3=1/3, U4=2/3, U5=1. Each file is a
separate empirical marginal. The headerless coordinate columns are
`y_aligned,x_aligned`; use that order consistently or explicitly convert it.
Generate predictions at the observed times and intermediate times, and compare
models using the same samples, coordinate units, and evaluation procedure.

#### Inputs and interpretation

- `aligned/aligned_spots/U2_tumor_coordinates.csv` through `U5_tumor_coordinates.csv`:
  model observations.
- `aligned/aligned_spots/U*_all_aligned_spots.csv`: all spots after alignment,
  retained for alternate analyses; do not substitute these for tumour-only
  inputs without documenting the change.
- `aligned/ref_U5_warped_images/`: visual reference images, not model features.
- `extracted/`: pre-alignment spot tables and the landmark correspondences used
  in the preprocessing history. Use the aligned tumour tables for the experiment.

The folder contains the complete preprocessed dataset available for these
experiments. No additional raw-data preprocessing is required to run models on
these inputs. The source dataset is described by [Mo et al. (2024)](https://www.nature.com/articles/s41586-024-08087-4);
follow the data-use terms recorded in `README.md` when sharing results or data.

## Cell tracking

### PhC-C2DH-U373 cell-tracking experiment data

#### Contents and source

The raw images and segmentation masks come from the Cell Tracking Challenge
[PhC-C2DH-U373 training dataset](https://celltrackingchallenge.net/2d-datasets/)
(direct archive: `https://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip`).
This bundle contains only sequence `01/` (115 frames, `t000.tif` through
`t114.tif`) and its segmentation masks in `01_ST/SEG/` (115 masks). Those are
the inputs specified for the experiment; sequence 02, tracking markers, and
`01_GT` annotations are not needed for the reported cell-4 experiment.

#### Observation construction

The experiment selects label 4 from each `01_ST/SEG` mask. Coordinates are pixel
indices in `(x,y)` order, with x horizontal and y vertical. Observations at each
frame are marginal point clouds; row order does not imply persistent cell
identity. The paper figure uses ten subsampled observations per frame for
training visualization, while full cell-4 mask pixels are used for the overlay
and full-mask diagnostics. A model comparison must state whether it uses the
same ten-point subsampling, all cell-4 pixels, or another sample count.

`prepared/mask_cell4_stack.npy` is a pre-existing convenience export with shape
`(115,520,696)` and a binary cell-4 mask per frame. It was prepared from the
segmentation masks and is included so other models can consume the exact same
spatial support. The TIFFs remain the source of truth. The original training
code's model-specific normalization and sampling settings are not baked into
the TIFFs or NPY; keep preprocessing/model outputs elsewhere.

#### Experiment timing and evaluation

Frames are ordered by integer time from 0 through 114; a normalized time is
`t/114` when a model requires `[0,1]`. The paper visualizes frames
`1,25,50,75,100,115` (one-based numbering). For direct comparisons, fix the
sampling seed and report per-frame metrics and the chosen checkpoint rule. The
paper's qualitative figure is not itself a numerical score.

## Knot

### Knot synthetic marginals

There is no fixed observation file: the dataset is generated on demand by the
copied `knot_distribution.py`, which is the repository's implementation. This
keeps the analytic formulation available to any model without coupling the
bundle to ALI-CFM training code.

Given `N=size`, require `N` divisible by 3 and define `t=linspace(0,3,N)` and
`u=3*(t/max(t))-1.5`. Split `u` into three consecutive equal blocks. The
coordinate means are:

- First block: `x=3*(u+0.5)`, `y=-tanh(5*(u+1))/2+0.5`.
- Middle block: `x=cos(2*pi*(u-0.75))`, `y=sin(2*pi*(u-0.75))+1`.
- Last block: `x=3*(u-0.5)`, `y=tanh(5*(u-1))/2+0.5`.

For each coordinate, the implementation adds independent standard Gaussian
noise scaled by `std`, broadcasting each mean across 10 noisy samples. It
returns the two endpoint marginals, the intermediate marginal array, and times
`t/3`. Exact returned axes/layout are defined by `loop_distribution`; preserve
them when matching the original loader. Because noise is sampled at each
invocation, record the NumPy RNG seed, `size`, and `std` for every experiment.
The source function uses NumPy's global RNG, so seed it before calling.

The checked-in repository does not provide a single canonical `size`, `std`, or
seed in this function. Those are experiment settings and must be reported with
the model run rather than inferred from the formula.

## Gulf of Mexico

### Gulf of Mexico: multimarginal particle transport

Folder: `data/gom`.

Source: [SB-Iterative-Reference-Refinement](https://github.com/YunyiShen/SB-Iterative-Reference-Refinement),
`Notebooks/data/GoMvortex_data.npy`, pinned to commit
`5d91ad36c65e4f46ff16202d0c788619a7538bda`. `source.json` records the URL and
expected hash. The 16,112-byte NPY is the actual processed benchmark input; the
full oceanographic source archive is not required.

**Files and loading**

`GoMvortex_data.npy` is the original unnormalized float64 array of shape
`(9,111,2)`: axes are `(timepoint, observation, coordinate)`. It contains 999
observed positions. The row at a given index across timepoints must not be
interpreted as the same tracked particle. Use the provided coordinate system;
the loader does not convert it to geographic units.

`marginals.npz` is a portable float32 export, matching the loader's cast:

```python
from pathlib import Path
import numpy as np

folder = Path("data/gom")
with np.load(folder / "marginals.npz", allow_pickle=False) as data:
    samples = data["samples"].copy()         # (9, 111, 2), raw coordinates
    times = data["times"].copy()             # (9,), i/8
    train = data["train_indices"].copy()     # [0,2,4,6,8]
    holdout = data["holdout_indices"].copy() # [1,3,5,7]
marginals = [samples[i] for i in range(9)]
```

**Experiment protocol**

Use indices `[0,2,4,6,8]` for training and `[1,3,5,7]` for holdout interpolation.
Time is `i/8`. Endpoints are indices 0 and 8; intermediate potentials are at
`[0.25,0.5,0.75]`. There is no further train/validation observation split in the
provided configs.

For parity with the loader, fit a per-coordinate `StandardScaler` on all nine
marginals concatenated as `(999,2)`, including holdouts, using population variance
(`ddof=0`). Train only on the transformed training marginals. Saved arrays are raw;
do not pre-normalize them in place. Inverse-transform predictions and references
for W2 evaluation.

**Run OTP-FM**

From a checkout with experiment dependencies installed:

```bash
python -m experiments.train --dataset gulfofmexico --config "data/gom/configs/W2.json" --tag gom_w2 --seed 42
python -m experiments.train --dataset gulfofmexico --config "data/gom/configs/W2Inf.json" --tag gom_w2inf --seed 42
```

`MMD.json` and `KL.json` are also included. All configs explicitly contain the
full merged settings and migrated `data_dir`; no `--potential` flag is needed.
The source's missing-default-config warning can still appear, but the explicit
JSON supplies the complete settings. W2 means precomputed EMD row-argmax pairing
chains with `W2InfPotential`; W2Inf uses independent pairings. OT chains are
computed locally, not ground-truth tracks. POT is needed for OT and W2.

The configs use a 10-layer, 128-wide network, learning rate 0.001, batch size 64,
and 2,000 epochs. W2/W2Inf use strength 800 and a box kernel of width 0.2.
Preserve the method-specific differences in the MMD/KL configs. Those presets
contain a legacy `solver` key; the CLI reads `solver_type`, so their actual solver
defaults to safeguarded Anderson. The bundle preserves source settings rather
than silently changing this behavior.

Report W2 at `[1,3,5,7]`, their unweighted mean, and training W2 averaged over
`[2,4,6,8]` (source excluded). Checkpoint selection minimizes average holdout W2.
The existing trainer evaluates holdouts by taking the lowest W2 over up to seven
nearby trajectory-grid points; training times use the nearest single point.
For another model, explicitly choose either that implementation behavior or
evaluation at the specified fixed times and report which was used. Repeat across
seeds for mean/std; a single run does not reproduce the published uncertainty.

## Beijing PM2.5

### Beijing: Dingling PM2.5 experiment

Folder: `data/beijing`.

Source: [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data).
The full download contains 12 station CSVs for March 2013–February 2017. The
experiment uses only `PRSA_Data_Dingling_20130301-20170228.csv` and its `PM2.5`
column. Values are pollutant concentrations in micrograms per cubic meter.
`source.json` records the precise download URL and expected hash. The preparation
tool extracts the inner `PRSA2017_Data_20130301-20170228.zip`, working around the
incorrect inner filename in the checkout's automatic downloader.

**Experiment preprocessing**

Drop observations with missing PM2.5, sort by year/month/day/hour, and select
every other calendar month from March 2013 to March 2015 inclusive. Do not average
the observations: each month's hourly readings form a one-dimensional marginal.

| Index | Month | Count | Use |
| --- | --- | --- | --- |
| 0 | 2013-03 | 744 | train/source |
| 1 | 2013-05 | 706 | train |
| 2 | 2013-07 | 710 | holdout |
| 3 | 2013-09 | 716 | train |
| 4 | 2013-11 | 720 | train |
| 5 | 2014-01 | 721 | holdout |
| 6 | 2014-03 | 743 | train |
| 7 | 2014-05 | 734 | train |
| 8 | 2014-07 | 728 | holdout |
| 9 | 2014-09 | 702 | train |
| 10 | 2014-11 | 717 | train |
| 11 | 2015-01 | 730 | holdout |
| 12 | 2015-03 | 738 | train/target |

There are 9,409 values. Time index `i` maps to `i/12` in `[0,1]`. Training indices
are `[0,1,3,4,6,7,9,10,12]`; holdouts are `[2,5,8,11]`. There is no additional
observation-level validation split in the supplied real-data configs.

**Files and loading**

`dingling_pm25_13marginals.npy` is the repository-compatible cache: 13 object-array
entries containing unnormalized float32 arrays of shape `(count_i,1)`. It needs
`allow_pickle=True`. Prefer the equivalent `marginals.npz` for other models:

```python
from pathlib import Path
import numpy as np

folder = Path("data/beijing")
with np.load(folder / "marginals.npz", allow_pickle=False) as data:
    values = data["values"]                 # float32, (9409, 1), raw units
    offsets = data["offsets"]               # int64, (14,), starts at 0
    marginals = [values[offsets[i]:offsets[i+1]].copy() for i in range(13)]
    times = data["times"].copy()             # (13,), i/12
    train = data["train_indices"].copy()
    holdout = data["holdout_indices"].copy()
```

Other NPZ keys: `counts` `(13,)` and `month_indices` `[0,2,...,24]` relative to
March 2013. There is no trajectory identity linking readings in different months.

For parity with the loader, fit one `StandardScaler` to the concatenation of all
13 raw marginals, including holdouts, using population variance (`ddof=0`). Train
on the transformed training marginals only. Do not normalize the saved inputs
again in place. Inverse-transform predictions and references before W2 evaluation.

**Run OTP-FM**

From a checkout with experiment dependencies installed:

```bash
python -m experiments.train --dataset beijingair --config "data/beijing/configs/W2.json" --tag beijing_w2 --seed 42
python -m experiments.train --dataset beijingair --config "data/beijing/configs/W2Inf.json" --tag beijing_w2inf --seed 42
```

`MMD.json` and `KL.json` are also included. Every config is fully merged and
contains this folder's absolute `data_dir`, so no `--potential` flag is needed.
The source's missing-default-config warning can still occur; the explicit JSON
supplies the complete settings. W2 uses precomputed full-data EMD pairings plus
`W2InfPotential`; W2Inf uses independent pairings. These are not true observed
particle trajectories. POT is needed for coupling and Wasserstein evaluation.

The W2/W2Inf configurations train a 4-layer, 128-wide network for 600 epochs at
learning rate 0.003 and batch size 128. MMD/KL presets use 100 epochs. Preserve
the method-specific strengths, widths, and kernels in the config when comparing.

Report W2 at each held-out index and their unweighted mean. Training W2 averages
indices `[1,3,4,6,7,9,10,12]`, excluding the source. The trainer selects the epoch
with minimum average holdout W2; this uses holdouts for model selection. Its W2
helper truncates unequal sample sets to equal lengths. A new implementation using
all observations should label that protocol difference. Repeat across seeds for
mean/std; seed 42 alone is one run, not the published multi-seed result.

## Synthetic Gaussian

### Synthetic Gaussian marginals

Folder: `data/synthetic`. There is no external download. The preparation
script exports reproducible independent samples from three documented 1D Gaussian
examples in this checkout. These are representative inputs, not a complete set of
paper figure configurations or pretrained/exact solution trajectories.

| File | Means | Standard deviations | Times | Samples per marginal |
| --- | --- | --- | --- | --- |
| `cli_defaults.npz` | `[0,1,0.5]` | `[0.3,0.2,0.4]` | `[0,0.5,1]` | 2,000 |
| `quickstart.npz` | `[0.5,1.5,0.4,0.8]` | `[0.1,0.1,0.1,0.1]` | `[0,0.33,0.67,1]` | 100,000 |
| `exact_example.npz` | `[0,3,1]` | `[1,0.5,sqrt(2)]` | `[0,0.5,1]` | 2,000 |

Origins are `experiments/train.py:setup_gaussian`, notebook 01, and notebook 05,
respectively. For the last example, the source/intermediate/target variances are
`[1,0.25,2]`; the 2,000-sample export size is a bundle choice, not a notebook
training setting. Recipe JSONs and `manifest.json` record the actual sample count
and seed, including any CLI overrides.

**Generation and interpretation**

For each marginal, draw independent standard normal samples, multiply by its
standard deviation, and add its mean. The exporter uses NumPy PCG64 with default
seed 42, restarting the generator for each recipe. Output is float32. Source
examples use PyTorch random draws; identical seeds across NumPy/PyTorch do not
produce identical observations. Transfer the exported arrays themselves when
comparing models on exactly the same observations.

No normalization or OT pairing is applied. Samples at different marginal indices
are independent; matching row indices do not imply true trajectories. The first
90% of each marginal's rows are the training sample split, and the last 10% are
validation, following `create_gaussian_dataloaders`. These are row indices, not
held-out timepoints: every marginal participates in these Gaussian examples.

**Loading the same samples in any model**

```python
from pathlib import Path
import numpy as np

folder = Path("data/synthetic")
with np.load(folder / "quickstart.npz", allow_pickle=False) as data:
    samples = data["samples"].copy()         # float32, (T, N, 1)
    times = data["times"].copy()             # float64, (T,)
    means = data["means"].copy()             # float64, (T,)
    stds = data["stds"].copy()               # float64, (T,), not variances
    train = data["train_indices"].copy()     # row indices in each marginal
    val = data["val_indices"].copy()

### OTPFM.forward_with_loss expects (batch, T, dimension).
xs_train = np.transpose(samples[:, train, :], (1, 0, 2)).copy()
xs_val = np.transpose(samples[:, val, :], (1, 0, 2)).copy()
```

For PyTorch, use `torch.from_numpy(xs_train)` and a `TensorDataset`/`DataLoader`,
then pass each `(batch,T,1)` tensor to `model.forward_with_loss`. Intermediate
potential times must be exactly `times[1:-1]`. Sample model trajectories from
the source marginal. Do not pass the full `(T,N,1)` array as a minibatch.

The existing Gaussian CLI generates its own samples and does not read `data_dir`
or these NPZ files. To train on this export, adapt its dataloader to the example
above; changing `data_dir` alone will not do it. Also, `setup_gaussian` does not
forward arbitrary `tks`/dimension to its dataloader, so use the explicit arrays
for the quickstart's exact `[0.33,0.67]` times. This bundle does not change that
training interface.

Notebook 01 supplies a quickstart W2Inf model with strength 500, Gaussian width
0.15, hidden width 256, and two hidden layers. Notebook 05 compares learned flows
with `GaussianMarginalSolver`; running its numerical solver is a separate step
from sampling Gaussian observations. An analytic mean/std specification alone
does not uniquely define the trajectories: specify potentials and their strengths,
widths, training objective, schedule, and sampling steps for each comparison.

For fair comparisons, keep the exact exported sample split and evaluation times,
record the model seed separately from the data seed, and agree on the same metric
and normalization. There is no canonical single scalar paper target associated
with these three input files.
