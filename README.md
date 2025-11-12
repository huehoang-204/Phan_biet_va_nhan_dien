

# Phân tách & Nhận dạng Người Nói (Diarization + Classification)

Ứng dụng web nhỏ giúp:

* Tách người nói (diarization) từ file audio dài.
* Gộp/cắt audio cho từng speaker.
* Nhận dạng người nói bằng mô hình **NeMo** fine-tune (`.nemo`) và **đổi tên file theo nhãn**.
* Vẽ waveform đầu vào, waveform có vùng speaker (màu theo nhãn) và waveform từng speaker.
* Giao diện đẹp, có overlay “Đang xử lý…“, progress, và nút **Reload về màn hình mặc định**.

---

## Cấu trúc repo

```
.
├── chayngaydi.py              # Pipeline diarize + classify + vẽ waveform
├── diar_infer_meeting.yaml    # Cấu hình NeMo diarization
├── part_002.wav               # Ví dụ audio (tuỳ chọn)
├── webapp/
│   ├── server.py              # Flask server
│   ├── templates/
│   │   └── index.html         # Giao diện trang chính
│   └── static/
│       └── style.css          # CSS giao diện
└── README.md
```

> Ghi chú: thư mục đầu ra mặc định: `labeled_outputs/` (tự tạo). Ảnh/âm thanh sinh ra được phục vụ qua router `/outputs/...`.

---

## Yêu cầu hệ thống

* Python 3.10 (khuyến nghị)
* GPU CUDA (khuyến nghị) hoặc CPU (chậm)
* Thư viện:

  * `nemo_toolkit[asr]`
  * `torch`, `torchaudio`
  * `soundfile`, `librosa`, `matplotlib`
  * `omegaconf`
  * `flask`

### Cài đặt nhanh (conda)

```bash
conda create -n nemo python=3.10 -y
conda activate nemo

# Torch + CUDA (chọn phiên bản phù hợp GPU/CUDA của bạn)
# Ví dụ CUDA 11.8:
pip install --upgrade pip
pip install torch==2.2.* torchaudio==2.2.* --index-url https://download.pytorch.org/whl/cu118

# NeMo & các thư viện khác
pip install nemo_toolkit[asr]==1.23.0  soundfile librosa matplotlib omegaconf flask
```

> Nếu dùng CPU: cài bản `torch` CPU theo hướng dẫn của PyTorch.

---

## Chuẩn bị mô hình & cấu hình

* Đặt file `.nemo` đã fine-tune vào đường dẫn bạn muốn (ví dụ `do_an/titanet_ft_last.nemo`).
* Kiểm tra/cập nhật đường dẫn trong **`chayngaydi.py`**:

  * `NEMO_PATH` → trỏ tới file `.nemo`
  * `YAML_PATH` → trỏ tới `diar_infer_meeting.yaml`
  * `OUT_DIR` / `PUBLIC_OUT_DIR` (mặc định `labeled_outputs/`)
  * (Tuỳ chọn) `LABELS_OVERRIDE = ["Bich","Chien","Chinh","Nam"]` để ép nhãn chữ.

---

## Chạy web app

```bash
cd webapp
python server.py
```

* Mặc định chạy tại `http://127.0.0.1:7860/`.
* Giao diện:

  * Chọn **File audio** (`.wav`/`.mp3`)
  * Chọn **Kiểu xuất file**:

    * `Ghép liền theo speaker (concat)`: gộp các phát ngôn của cùng 1 speaker liền nhau (không chèn khoảng lặng).
    * `Dài bằng audio gốc (full)`: mỗi speaker 1 file dài bằng bản gốc, chỗ không nói = 0.
    * (Tuỳ chọn) `segments`: cắt từng đoạn phát ngôn.
  * Bấm **Xử lý** → xuất kết quả.

### Kết quả hiển thị

* **Phổ Audio Đầu Vào**: 1 màu.
* **Phổ Audio Đầu Ra (Diarized)**: overlay theo màu **nhãn** ở timeline.
* **Audio Speakers** (danh sách bên phải):

  * Mỗi speaker có **audio riêng** và **waveform** riêng (màu theo nhãn).
  * Nút **Tải xuống** từng file.

> Màu sắc giữa “Phổ đầu ra (overlay)” và “waveform từng speaker” được đồng bộ theo **nhãn chữ**, tránh lệch do `speaker_0/1/...`. Hệ thống tự ánh xạ nhãn sau phân loại.

---

## Chạy pipeline không cần web (CLI)

Trong `chayngaydi.py`, chỉnh các hằng số, sau đó:

```bash
python chayngaydi.py
```

* Kết quả nằm trong `labeled_outputs/<tên_file>`:

  * `*_INPUT_WAVEFORM.png`
  * `*_OUTPUT_WAVEFORM.png` (overlay theo speaker)
  * `Bich_WAVEFORM.png`, `Chien_WAVEFORM.png`, … (từng speaker)
  * `*_Bich.wav`, `*_Chien.wav`, … (từng speaker, **đã đổi tên theo nhãn**)

---

## Các tính năng UI đã thêm

* **Overlay “Đang xử lý…”** kèm spinner và hiệu ứng shimmer khi submit.
* **Hiển thị tiến trình** (progress) giả lập trong overlay cho trải nghiệm mượt (có thể nối backend real-time nếu cần SSE/WebSocket).
* **Reload về trang mặc định**:

  * Nút reload (hoặc biểu tượng reload trình duyệt) sẽ điều hướng về `/` thay vì reload `/process` để **không hiện “Resubmit the form?”**.
  * Đã thêm đoạn JS chặn `bfcache` và dọn state để về màn hình chưa có kết quả.
* **Hiệu ứng hover của thẻ speaker** đã điều chỉnh để **bóng mờ không bị lệch** (dùng `::before` có `inset:0`, `pointer-events:none`, shadow mềm + translate nhẹ).

> Nếu bạn chưa có các thay đổi JS/CSS nói trên, mình có thể dán patch riêng ngay trong chat này.

---

## Mẹo/Tip vận hành

* **Hiệu năng**: Chế độ `full` có thể tốn VRAM. Trong `chayngaydi.py` đã hỗ trợ `CLASSIFY_VIA_SPEECH_ONLY_FOR_FULL` để classify trên **trích đoạn speech-only** (rồi map nhãn lại cho file full).
* **Đồng bộ màu**: Bảng màu cố định cho nhãn (`Bich`, `Chien`, `Chinh`, `Nam`). Nếu nhãn thay đổi, sửa `LABELS_OVERRIDE` và/hoặc `PALETTE`.
* **Tên file đầu ra**: luôn là **nhãn chữ**, tránh các số `0,1,2,3`. Nếu model trả nhãn rỗng/không hợp lệ, code sẽ **raise** để bạn kiểm tra `labels.txt` hoặc `LABELS_OVERRIDE`.

---

## Nén/giải nén dự án (terminal)

```bash
# ZIP cả dự án (bỏ .git)
zip -r project.zip . -x "*.git*"

# TAR.GZ (giữ permission tốt trên Linux)
tar -czf project.tar.gz .

# 7z (nén tốt)
7z a project.7z .
```

---

## Troubleshooting

* **CUDA OOM / chậm**
  Giảm kích cỡ audio, dùng `concat` thay vì `full`, bật `CLASSIFY_VIA_SPEECH_ONLY_FOR_FULL` và giới hạn `SPEECH_ONLY_MAX_SECONDS`.

* **Sai nhãn / ra số thay vì chữ**

  * Kiểm tra `LABELS_OVERRIDE` trong `chayngaydi.py`.
  * Hoặc tạo `labels.txt` cùng thư mục với `.nemo` (mỗi dòng 1 nhãn).
  * Đảm bảo thứ tự nhãn đúng với lúc fine-tune.

* **Màu không đồng bộ**
  Hãy giữ một **bảng màu duy nhất** (ví dụ `PALETTE`) và mọi chỗ lấy màu đều đi qua hàm `color_for(label)` (đã có trong mã).

* **Reload hỏi “Resubmit the form?”**
  Sử dụng nút Reload đã cài đặt (JS điều hướng về `/`) hoặc nhấn logo/tiêu đề để về trang `/`.

---

## Ghi công

* [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) cho diarization & speaker classification.
* `librosa`, `soundfile`, `matplotlib` cho xử lý & vẽ audio.
* `Flask` cho web server.

---

## License

Tuỳ chọn (MIT/GPL/…); thêm nội dung giấy phép bạn mong muốn.

---

## Liên hệ

* Tác giả: **Hoang Phuong**
* Email: `hoangphuonghue92@gmail.com`

Nếu bạn muốn, mình có thể tạo luôn file `README.md` với badge, ảnh chụp màn hình UI, và thêm script `requirements.txt/pyproject.toml`.
