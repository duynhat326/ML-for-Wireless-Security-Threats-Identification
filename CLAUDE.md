# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Trạng thái repo

Repo mới ở giai đoạn khởi tạo: `src/`, `notebooks/`, `experiments/`, `app/` chỉ có `.gitkeep`; `requirements.txt` và `README.md` gần như trống; chưa có lệnh build/lint/test. Nguồn sự thật duy nhất là kế hoạch dự án: `reports/Kế hoạch dự án ML nhận diện mối đe dọa bảo mật không dây (AWID3).md` (tiếng Việt, nhóm 3 người, 6 tuần). Đọc nó trước khi thiết kế bất kỳ thứ gì; cập nhật CLAUDE.md này khi có lệnh thật (cài đặt, chạy thí nghiệm, test).

Dự án: nghiên cứu thực nghiệm ML nhận diện tấn công WiFi trên AWID3, chỉ dùng bản CSV (không pcap). Giá trị nằm ở thiết kế thí nghiệm đúng, không phải một con số accuracy.

## Dữ liệu

- `data/` không commit. CSV nằm ở `data/AWID3_Dataset_CSV/` (và file zip cùng cấp); tài liệu gốc ở `../Dataset/` (`read_me.pdf`, `code-features`).
- Cấu trúc cột, nhãn, cột thời gian/định danh, việc mỗi file tấn công có Normal riêng hay không đều **chưa được xác nhận**: kế hoạch yêu cầu làm data card ở Tuần 1. Không giả định tên cột; kiểm tra dữ liệu thật.
- Dữ liệu lớn: đọc theo khối, ép kiểu (float32/category), lưu Parquet rồi mới gộp. Bỏ cột trước khi bỏ dòng.

## Kiến trúc dự kiến (theo kế hoạch)

Pipeline: CSV → Parquet → tiền xử lý → chia dữ liệu → mô hình → đánh giá → demo.

- `src/`: logic dùng chung (`load.py`, `features.py`, `split.py`, `models.py`, `evaluate.py`). Notebook chỉ gọi hàm, không ai tự viết lại cách chia dữ liệu.
- `experiments/`: E1..E7, mỗi lần chạy có file cấu hình (mô hình, tham số, hạt giống, cách chia) và kết quả cùng tên.
- `app/`: demo Streamlit replay dữ liệu test theo thời gian qua mô hình tốt nhất.
- `reports/`: hình, bảng, bản nháp báo cáo.

Thí nghiệm: E1 nhị phân, E2 đa lớp, E3 chia ngẫu nhiên vs theo thời gian (đo leakage), E4 ablation theo nhóm đặc trưng, E5 đặc trưng cửa sổ 1 giây theo nguồn phát (cần cột thời gian + nguồn phát), E6 tấn công chưa từng thấy (Isolation Forest/Autoencoder train chỉ trên Normal), E7 SHAP. Lõi bắt buộc: E1–E4, E7; E5, E6, demo là nâng cao.

## Quy tắc chống leakage (không được vi phạm)

Thứ tự tiền xử lý cố định:
1. Bỏ cột dễ học thuộc: MAC, IP, SSID, số thứ tự frame, timestamp thô (giữ timestamp riêng chỉ để chia dữ liệu và tạo đặc trưng cửa sổ).
2. Thiếu giá trị: điền sentinel (ví dụ -1) + cờ "có header X"; không xóa bừa.
3. Chia **theo thời gian trong từng file**: ~70% train / 15% val / 15% test, có khoảng đệm giữa các phần. Nếu không có Normal riêng hoặc timestamp không đáng tin, chia theo file capture.
4. Scaler, chọn đặc trưng, SMOTE chỉ fit trên train. Undersample Normal chỉ trên train; test giữ phân bố thật.
5. Siêu tham số chọn chỉ trên validation; test chạm một lần ở cuối.

Nếu một mô hình đạt gần 100% trên test: dừng, rà cột định danh và cách chia trước khi báo kết quả.

## Quy ước đánh giá

- Chỉ số chính: macro-F1 và F1 từng lớp (không dùng accuracy đơn lẻ); PR-AUC cho nhị phân; confusion matrix chuẩn hóa theo hàng.
- Mỗi cấu hình chạy ≥ 3 hạt giống, báo trung bình ± độ lệch chuẩn; cố định `random_state`; ghi tỉ lệ giảm mẫu vào README/báo cáo.
- Đi từ đơn giản (Logistic Regression, Decision Tree) đến phức tạp (RF, LightGBM, MLP, bất thường); chỉ giữ mô hình phức tạp nếu thắng baseline có ý nghĩa.
- Nhánh `main` luôn chạy được; làm trên nhánh riêng, merge qua pull request. Không commit dữ liệu lớn hay mô hình nặng.
