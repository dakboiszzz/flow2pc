# Dữ liệu local: MFM → HCLT / Gaussian PC

Luồng snapshot mới có **20 profile**, không gồm AFHQ. Nó dùng source MFM gốc
tại `../pretrain/metric-flow-matching`, giữ hai giai đoạn geopath rồi velocity,
và distill bằng teacher velocity MSE. Những profile mới là adaptation, không phải
tuyên bố tái lập kết quả bài báo. Chưa xác minh runtime hoặc chất lượng training.

## Catalog

| Profile | Dữ liệu / biểu diễn | Marginal |
| --- | --- | --- |
| `cite5`, `cite50`, `cite100` | 31.240 mẫu, PCA có sẵn, ngày 2/3/4/7 | 4 |
| `multi5`, `multi50`, `multi100` | 33.701 mẫu, PCA có sẵn, ngày 2/3/4/7 | 4 |
| `eb_pca5`, `eb_pca50` | PCA từ EB H5AD, 18.203 mẫu | 5 |
| `eb_phate2` | PHATE 2D từ cùng EB H5AD | 5 |
| `st` | Tumor U2–U5 đã căn chỉnh, thứ tự tọa độ y,x | 4 |
| `cell_tracking` | Mask cell-4, thứ tự x,y | 115 |
| `gom` | Tọa độ 2D, 111 mẫu/marginal | 9 |
| `beijing` | PM2.5 Dingling, 1D, tổng 9.409 mẫu | 13 |
| `gaussian_cli_defaults` | 1D, 2.000 mẫu/marginal | 3 |
| `gaussian_quickstart` | 1D, 100.000 mẫu/marginal | 4 |
| `gaussian_exact_example` | 1D, 2.000 mẫu/marginal | 3 |
| `knot` | Generator đã có: 120 thời điểm, 10 mẫu/thời điểm, noise std 0.01 | 120 |
| `arch`, `sphere` | Generator MFM gốc, 2D/3D, seed cố định | 3 |
| `lidar` | Rainier 3D và hai phân phối do loader MFM gốc tạo | 2 |

Single-cell lấy từ `data/single_cell/`, **không** lấy EB
`eb_velocity_v5.npz` của bộ MFM cũ. Không tính lại PCA. Các khoảng ngày EB được
giữ nguyên dưới dạng nhãn; không suy diễn midpoint thành thời gian vật lý.

## Bố cục và chuẩn bị

Dữ liệu local nằm dưới một gốc `data/`, không tách theo phương pháp.
Thư mục `mfm/` chỉ chứa catalog, liên kết nguồn và snapshot do MFM chuẩn bị;
các bộ dữ liệu gốc nằm ngay bên cạnh và được dùng chung.

```text
data/mfm/
  catalog.json
  sources/                 # symlink, không di chuyển hay nhân đôi dữ liệu gốc
    single_cell/
    gom/
    beijing/
    spatial_transcriptomics/
    cell_tracking/
    gaussian/
    lidar                  # symlink đến rainier2-thin.las
  prepared/<profile>/seed42/
    manifest.json
    <index>_values.npy
    <index>_train.npy
    <index>_validation.npy
    metric_<index>.npy      # chỉ LiDAR: địa hình conditioning cho metric
```

Toàn bộ `data/` được Git bỏ qua. Mã catalog và các YAML được version-control.
Các symlink/catalog đã được tạo trong checkout này; **snapshot số học chưa được
materialize**. `prepare` hoặc `both` sẽ thực hiện việc đó khi bạn chạy lệnh.

```bash
python scripts/run_mfm_data.py list
python scripts/run_mfm_data.py prepare all --links-only
python scripts/run_mfm_data.py prepare all
```

Lệnh cuối đọc dữ liệu, tính checksum, tạo split và snapshot; không train hay
evaluate model. Lặp lại sẽ kiểm tra checksum và tái sử dụng dữ liệu đã chuẩn bị.
File xung đột hoặc source thay đổi sẽ gây lỗi, không tự ghi đè. Dùng data root
mới nếu muốn tạo một phiên bản khác.

`--bundle` chọn thư mục chứa `single_cell/`, `gom/`, `beijing/`, `synthetic/`;
mặc định `data/`. `--native-data` chọn nơi có file LiDAR, mặc định
`../pretrain/metric-flow-matching/data`. `--data-root` chọn đích chuẩn bị,
mặc định `data/mfm`. `--source-root` chọn checkout MFM. Đường dẫn CLI tương đối
tính từ thư mục gọi lệnh; đường dẫn trong YAML MFM tính từ root repository.

## Chạy từng profile hoặc tất cả

```bash
# Chỉ in kế hoạch và lệnh; không ghi file, không load dữ liệu/model.
python scripts/run_mfm_data.py both all --student both --dry-run

# Chuẩn bị → pretrain MFM → distill HCLT, rồi Gaussian.
python scripts/run_mfm_data.py both cite50 --student both --device cuda:0 --run-name exp1

# Tất cả 20 profile, tuần tự, dừng ở lỗi đầu tiên.
python scripts/run_mfm_data.py both all --student both --device cuda:0 --run-name exp1

# Tách từng bước; pretrain cần snapshot đã được prepare.
python scripts/run_mfm_data.py prepare gom
python scripts/run_mfm_data.py pretrain gom --run-name exp2
python scripts/run_mfm_data.py distill gom --student hclt --run-name exp2
```

Mặc định HCLT, seed 42, W&B online (project `flow2pc`), teacher tối đa 1.000 epoch mỗi giai đoạn
có early stopping, student 20.000 update. `--epochs`, `--steps`, `--batch-size`,
`--seed`, `--device`, `--logging` ghi đè các giá trị liên quan.
Số epoch là giới hạn, không phải ngân sách đã tune cho mỗi dataset.
`--python` hoặc `FLOW2PC_PYTHON` chọn Python cho cả preparation và model jobs.

Kết quả:

```text
outputs/mfm/<profile>/<run-name>/
  pretrain/
    resolved_config.yaml
    status.json
    native/{geopath,flow}/best.ckpt
    checkpoints/{flow,geopath}.pt
    teacher_pair.json
  distill/{hclt,gaussian}/
    checkpoints/{best,last}.pt
    config.json
    data_manifest.json
    metrics.csv
```

Teacher mới yêu cầu thư mục output trống; không tự overwrite/resume. Student
resume qua `--set distillation.resume_checkpoint=...` trên CLI `main.py` hiện có.
Muốn distill teacher khác, dùng `distill PROFILE --teacher-run /path/to/pretrain`;
script lấy đường dẫn snapshot từ binding của teacher, kiểm tra đúng profile.

Các YAML trực tiếp nằm trong `configs/pretrain/mfm/data/` và
`configs/distill/mfm/data/<profile>/`:

```bash
PYTHONPATH=src python main.py --config configs/pretrain/mfm/data/gom.yaml --validate-only
PYTHONPATH=src python main.py --config configs/pretrain/mfm/data/gom.yaml
PYTHONPATH=src python main.py --config configs/distill/mfm/data/gom/hclt.yaml
```

`--validate-only` của MFM kiểm tra cấu hình, không chạy kiểm tra dữ liệu/model
end-to-end. Các YAML student mặc định **có post-training evaluation** từ `best.pt`,
gồm EMD và ảnh so sánh teacher ODE/student ODE/direct PC. Validation velocity MSE
vẫn chọn student checkpoint; xem [quy ước theo dataset](EXPERIMENT_PROTOCOLS.md).

## Contract khoa học và tương thích

- Split 90/10 độc lập ở từng thời điểm quan sát, seed cố định. Gaussian dùng
  đúng chỉ số train/validation trong NPZ. Các hàng không có identity trajectory.
- Temporal holdout: GOM `[1,3,5,7]`, Beijing `[2,5,8,11]`; CITE/Multi, EB,
  ST, Arch và Sphere `[1]`; các profile còn lại không có temporal holdout.
  Holdout không có train/validation rows, chỉ có test references.
- CITE/Multi/EB/ST/tracking/GOM/Beijing chuẩn hóa theo mean/std của **train**.
  Arch/Sphere/Knot/Gaussian giữ tọa độ gốc. LiDAR giữ nguyên phép biến đổi của
  source, gồm scaler và hình học từ toàn bộ ground scene; manifest ghi ngoại lệ.
- Model time lấy từ grid explicit đã lưu; quickstart giữ `[0,.33,.67,1]`.
  Các nguồn còn lại giữ mapping ordinal hiện có và ghi nhãn/đơn vị riêng.
- Metric LAND chỉ nhận train rows, ngoại trừ LiDAR giữ toàn bộ địa hình
  conditioning như source gốc (đây không phải marginal test). Không thay địa hình
  bằng chỉ hai đám mây endpoint. Các khoảng MFM nối hai marginal quan sát
  liên tiếp qua một hoặc nhiều holdout. Adapter giữ nguyên loss/optimizer/steps
  upstream, đổi phần cấp dữ liệu và chọn interval; sửa mất chiều do `squeeze`
  và JVP cho dữ liệu 1D.
- Batch thực tế là min(requested batch, kích thước train/validation nhỏ nhất).
  Vì vậy GOM có thể dùng batch 12 và Knot dùng batch 1; không ngầm nhân mẫu.
  Native loader vẫn drop phần dư của một batch và dùng `min_size` giữa marginals.
  Metric-reference batches giữ cách lấy pool cân bằng của source; riêng LiDAR
  giữ hai nửa địa hình đầy đủ. Đây là pool riêng, không bị cắt theo batch network.
- Recipe native làm điểm khởi đầu được ghi ở đầu mỗi pretrain YAML. Các bộ mới
  dùng LAND từ recipe EB 5D; không dùng loss TorchCFM thay cho hai giai đoạn MFM.
  CITE/Multi 50D/100D giữ kiến trúc 1024 của recipe gốc nhưng chuyển sang LAND
  theo protocol đã chọn (`gamma=0.125`, `rho=0.001`, `alpha_metric=2`), không lấy
  `rho` âm của metric RBF làm tham số LAND. Đây là adaptation, không phải recipe
  RBF gốc; các wrapper MFM cũ vẫn giữ RBF.
- Geopath tốt nhất theo validation được đóng băng cho giai đoạn flow; flow tốt
  nhất theo validation được xuất cùng đúng geopath đó. Pair manifest ràng buộc
  weights, dataset, time grid, split, preprocessing, source hashes và revision.
  Loader từ chối sai cặp hoặc source đã thay đổi.
- Hai PC dùng biên đầu tự do và khởi tạo từ marginal train đầu. Không ép Gaussian
  chuẩn, không thêm source NLL. Khởi tạo không bảo đảm density khớp chính xác
  marginal đầu; velocity matching cũng không tự bảo đảm điều đó.
- Teacher, geopath đều đóng băng khi distill. Student dùng đúng split và tọa độ
  đã chuẩn bị, không fit lại scaler hoặc chia lại dữ liệu.
- Các recipe/checkpoint cũ và `scripts/run_mfm.sh` giữ đường chạy cũ. Không đổi
  protocol ALI-CFM/OTP-FM và không đổi toán học HCLT/Gaussian.

Môi trường cần optional dependencies `.[metric]` và source MFM. Upstream import
các helper ảnh ngay trong module trainer vector, nên extra này cũng khai báo
`lpips`, `fld`, `phate`, `scprep`; không tải dữ liệu hoặc weights ảnh để chạy
các profile này. Không dependency nào được tự cài trong đợt chỉnh sửa.

Đợt triển khai chỉ tạo mã, YAML, symlink và tài liệu, kiểm tra tĩnh. Các test
snapshot/launcher được thêm nhưng **chưa chạy**; training, distillation và
evaluation cũng **chưa chạy**.
