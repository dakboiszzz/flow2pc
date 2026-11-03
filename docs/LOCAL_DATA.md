# Chạy với dữ liệu trong repository

Để chạy **MFM → HCLT/Gaussian trên toàn bộ 20 profile, trừ AFHQ**, xem
[catalog và launcher MFM mới](MFM_LOCAL_DATA.md). Hướng dẫn dưới đây dành cho
launcher ALI-CFM/OTP-FM; các giới hạn distill bên dưới áp dụng cho hai teacher đó.

`scripts/run_local_data.py` nối các cấu hình hiện có với bundle tại
`data/`. Script không thay thuật toán, thời gian, cách chia dữ liệu,
preprocessing hay ngân sách huấn luyện của các cấu hình gốc.

| Dữ liệu / tên truyền vào script | ALI-CFM pretrain | ALI-CFM → HCLT | OTP-FM pretrain |
| --- | --- | --- | --- |
| `cite5`, `cite50`, `cite100` | Có | Có | Có |
| `multi5`, `multi50`, `multi100` | Có | Có | Có |
| `eb` (PHATE 2D), `st`, `cell_tracking` | Có | Có | Có |
| `knot` (sinh bằng công thức đã có) | Có | Có | Có |
| `gom`, `beijing` | Có | Chưa có protocol validation phù hợp | Có |
| `synthetic/{cli_defaults,quickstart,exact_example}.npz` | Chưa tích hợp | Chưa tích hợp | Chưa tích hợp |

“Có” nghĩa là đã có loader, trainer và cấu hình trong mã nguồn; không có nghĩa
là đã chạy thành công hoặc xác minh chất lượng trên máy này. Script chưa tích hợp
OTP-FM distill, MFM hoặc MMFM. MFM vẫn dùng `scripts/run_mfm.sh` và contract
dữ liệu/checkpoint riêng; MMFM chưa có implementation đăng ký.

Từ thư mục repository, xem trước lệnh cho cả hai giai đoạn:

```bash
python scripts/run_local_data.py both cite50 --dry-run
```

Chỉ khi bạn chủ động thực thi lệnh sau, script mới chạy pretrain rồi distill:

```bash
python scripts/run_local_data.py both cite50 --device cuda:0 --run-name experiment1
```

Hoặc chạy từng giai đoạn, với cùng `--run-name`:

```bash
python scripts/run_local_data.py pretrain eb --run-name experiment1
python scripts/run_local_data.py distill eb --run-name experiment1
```

Kết quả nằm ở `outputs/local/alicfm/<dataset>/<run-name>/pretrain/` và
`distill/`. Student đọc `pretrain/checkpoints/last.pt`; dữ liệu, preprocessing
và time map được phục hồi từ manifest của teacher. Không cần chép dữ liệu.
`--data-root` là thư mục **chứa trực tiếp các bộ dữ liệu**, mặc định là `data/` của repository.
`--output-root` đổi thư mục kết quả. Các đường dẫn tương đối của hai tùy chọn này
được tính từ thư mục gọi lệnh. Dùng `--run-name` mới cho teacher mới: trainer
không ghi đè thư mục đã tồn tại. Student giữ thiết lập resume và seed của recipe.

Với checkpoint framework đã có, chọn đúng dataset/dimension:

```bash
python scripts/run_local_data.py distill cite50 \
  --teacher-checkpoint /absolute/path/to/checkpoints/last.pt \
  --run-name existing-teacher
```

Checkpoint này phải đi kèm manifest/preprocessing và dữ liệu ở đường dẫn đã ghi
trong manifest. `--data-root` không sửa contract của checkpoint cũ.

OTP-FM pretrain dùng các recipe đã tích hợp:

```bash
python scripts/run_local_data.py pretrain gom --method otpfm --recipe w2inf --dry-run
python scripts/run_local_data.py pretrain beijing --method otpfm --recipe mmd
python scripts/run_local_data.py pretrain multi50 --method otpfm
```

GOM/Beijing có `w2`, `w2inf`, `kl`, `mmd`; các dataset còn lại dùng `w2inf`.
Kết quả OTP-FM nằm ở
`outputs/local/otpfm/<dataset>/<recipe>/<run-name>/pretrain/`.

Script mặc định tắt W&B (`--logging disabled`). Dùng `--logging online` hoặc
`offline` nếu muốn. **Các recipe giữ nguyên evaluation sau huấn luyện**, và
OTP-FM có cả callback theo protocol gốc. `--dry-run` không đọc mảng dữ liệu,
khởi tạo model hay ghi file. `--validate-only` gọi kiểm tra hiện có của `main.py`,
có thể đọc dữ liệu lớn và checkpoint; dùng riêng từng giai đoạn. Môi trường cần
các dependency của teacher như mô tả trong README; script không tự cài đặt.

Các phần cần quyết định trước khi mở rộng:

- **GOM/Beijing distill:** transport loader giữ toàn bộ hàng ở thời điểm quan sát
  cho train, còn thời điểm holdout dành cho test; tập row-validation rỗng. Cần
  chọn protocol validation cho student trước khi tạo recipe distill. Không tự
  chia lại dữ liệu hoặc dùng test để chọn checkpoint.
- **OTP-FM distill:** adapter cung cấp velocity EMA, nhưng sampler conditional
  dùng tuple nhiều marginal chưa nối vào student runner. Không dùng thay bằng
  interpolant hai đầu của ALI-CFM.
- **Gaussian NPZ:** cần loader giữ đúng split 90/10 đã xuất và thời gian đã lưu
  (đặc biệt `0.33, 0.67` của quickstart), cùng profile teacher tương ứng. Không
  tự đổi chúng thành `1/3, 2/3` hay coi các hàng là trajectory quan sát.

HCLT hiện dùng Fourier với `fourier_zero_center=False` và khởi tạo từ marginal
train đầu tiên. Đây là khởi tạo empirical đã có, không phải giả định marginal
đầu là Gaussian chuẩn, cũng không đảm bảo density ban đầu khớp chính xác dữ liệu.

Thay đổi này chỉ thêm launcher, tài liệu và test nguồn; chưa chạy training,
distillation, evaluation hoặc test.
