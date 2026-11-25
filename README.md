# ML-for-Wireless-Security-Threats-Identification

Nhận diện tấn công WiFi trên AWID3 (bản CSV) bằng học máy (Machine Learning), kèm truy vết nguồn tấn công. Các thuật ngữ viết tắt được giải thích ở mục cuối.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `shared/` | Code dùng chung: `config.py`, `data_loader.py`, `evaluator.py`, `tracer.py`, `compare.py`, `prepare_train_test.ipynb` |
| `experiments/model_1_random_forest/` | Random Forest |
| `experiments/model_2_lightgbm/` | LightGBM |
| `experiments/model_3_mlp/` | Mạng nơ-ron nhiều lớp (Multi-Layer Perceptron), viết bằng PyTorch |
| `comparison/` | Bảng và biểu đồ so sánh, do `shared/compare.py` sinh ra |
| `demo/` | `detect_from_file.py` (chạy bằng dòng lệnh) và `dashboard.py` (Streamlit) |
| `tests/` | Test nhỏ cho tiền xử lý, chấm điểm, truy vết |

Mỗi `experiments/model_*/` có `train.ipynb`, `saved_model/` (file nặng, bị ignore) và `results/` (file nhẹ, được push).

## Chạy trên Kaggle

1. Tạo Kaggle Dataset chứa `awid3_clean.parquet`; thêm repo vào notebook, đặt `REPO_ROOT` nếu khác `/kaggle/working/ML-for-Wireless-Security-Threats-Identification`.
2. Chạy `shared/prepare_train_test.ipynb` một lần. Chia theo thời gian trong từng file: 70% train, 15% val, 15% test, có khoảng đệm. Ra `train.parquet`, `val.parquet`, `test.parquet`; lưu thành Kaggle Dataset và đặt `AWID3_DATA` trỏ tới nó.
3. Mỗi người chạy `train.ipynb` trong thư mục model của mình (mô hình mạng nơ-ron nhiều lớp bật bộ xử lý đồ họa NVIDIA T4). Tải `results/` về, thêm ảnh chụp log Kaggle `kaggle_run_capture.png`, rồi commit.
4. Khi đủ 3 `results/`, chạy `python -m shared.compare` ở gốc repo.

Link Kaggle Dataset và link tải `saved_model/`: chưa có, điền vào đây.

## Chạy local

```bash
pip install -r requirements.txt
python -m pytest
python demo/detect_from_file.py --model experiments/model_2_lightgbm/saved_model/model.pkl --packets sample.csv
streamlit run demo/dashboard.py
```

Mô hình là `model.pkl` (Random Forest, LightGBM) hoặc `model.pt` (mạng nơ-ron nhiều lớp).

## Quy tắc chống leakage

- Địa chỉ MAC, địa chỉ IP, tên mạng SSID, BSSID, số thứ tự frame, dấu thời gian, kênh và tần số không vào mô hình; chúng nằm ở bảng metadata để truy vết.
- Giá trị thiếu và `?` thành -1, kèm cờ `__missing`.
- Chia theo luồng, mỗi lần chỉ giữ vài trăm nghìn dòng trong bộ nhớ (đỉnh khoảng 3,6 GB trên file thật). Train bị giới hạn 500.000 dòng cho mỗi nhãn; val và test giữ phân bố thật.
- Seed cố định 42, mỗi model chạy một lần.
- Mô hình đạt gần 100% trên test: dừng và rà lại cột định danh và cách chia.

## Tổng hợp kết quả

Xem `comparison/summary_table.csv` sau khi chạy `shared/compare.py`.

| Mô hình | Macro-F1 | Accuracy | Thời gian train (giây) | Độ trễ dự đoán (mili giây mỗi frame) |
|---|---|---|---|---|
| Random Forest | chưa chạy | chưa chạy | chưa chạy | chưa chạy |
| LightGBM | chưa chạy | chưa chạy | chưa chạy | chưa chạy |
| Mạng nơ-ron nhiều lớp | chưa chạy | chưa chạy | chưa chạy | chưa chạy |

## Lưu ý truy vết

Với Deauth, Disas, Kr00k, RogueAP, địa chỉ MAC nguồn là của điểm truy cập (Access Point) hợp pháp bị mạo danh; công cụ gắn nhãn `impersonation` và không khẳng định kẻ tấn công. Với tấn công tầng IP, nguồn là `ip.src` (địa chỉ IP nguồn) hoặc `wlan.sa` (địa chỉ MAC nguồn).

## Thuật ngữ viết tắt

| Viết tắt | Nghĩa đầy đủ |
|---|---|
| ML | Machine Learning, học máy (có trong tên repo) |
| CSV | Comma-Separated Values, file văn bản dạng bảng |
| AWID3 | Aegean WiFi Intrusion Dataset 3, bộ dữ liệu tấn công WiFi dùng trong dự án |
| MAC | Media Access Control, địa chỉ phần cứng của thiết bị mạng |
| IP | Internet Protocol, giao thức Internet (địa chỉ IP) |
| SSID | Service Set Identifier, tên mạng WiFi |
| BSSID | Basic Service Set Identifier, địa chỉ MAC của điểm truy cập |
| `wlan.ta`, `wlan.sa` | Transmitter Address (địa chỉ máy phát), Source Address (địa chỉ nguồn) trong frame WiFi |
| `ip.src` | Địa chỉ IP nguồn |
| F1 | F1-score, trung bình điều hòa của precision và recall; macro-F1 là trung bình F1 của các lớp |
| Deauth, Disas | Deauthentication, Disassociation: tấn công ngắt kết nối |
| Krack | Key Reinstallation Attack |
| SSH | Secure Shell |
| SSDP | Simple Service Discovery Protocol |
| SQL | Structured Query Language |
| `.pkl`, `.pt` | File lưu mô hình: pickle (Python) và PyTorch |
| `__missing` | Hậu tố của cột cờ cho biết giá trị gốc bị thiếu |
