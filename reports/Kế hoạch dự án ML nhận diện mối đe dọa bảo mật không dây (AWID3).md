# Kế hoạch dự án: ML nhận diện mối đe dọa bảo mật không dây (AWID3)

Cập nhật: 2026-11-10 · Nhóm 3 người · Dữ liệu: AWID3 (chỉ dùng bản CSV)

## 1. Tổng quan và mục tiêu

Nhóm làm một nghiên cứu thực nghiệm đánh giá chặt chẽ các mô hình ML nhận diện tấn công WiFi trên AWID3, thay vì chỉ huấn luyện một mô hình rồi báo accuracy. Điểm mạnh của dự án nằm ở thiết kế thí nghiệm đúng (không leakage), phân tích sâu và giải thích được kết quả.

**Mục tiêu chính**

- Xây pipeline tái lập được: dữ liệu AWID3 → tiền xử lý → mô hình → đánh giá.
- So sánh ít nhất 5 mô hình (cây quyết định, Random Forest, LightGBM, MLP, Isolation Forest/Autoencoder) trên cùng một cách chia dữ liệu.
- Chỉ ra đặc trưng nào, ở tầng nào của giao thức, giúp nhận diện từng loại tấn công.
- Đo khả năng phát hiện tấn công chưa từng thấy.

**Trong phạm vi:** phân loại nhị phân và đa lớp, phát hiện bất thường, giải thích mô hình, demo replay trên dữ liệu có sẵn.

**Ngoài phạm vi (ghi rõ trong báo cáo là hướng phát triển):** bắt gói tin trực tiếp bằng card monitor mode, triển khai IDS thật, tự thu thập dữ liệu mới, xử lý pcap.

**Sản phẩm đầu ra:** repo GitHub, báo cáo, slide thuyết trình, demo Streamlit.

## 2. Câu hỏi nghiên cứu và đóng góp của nhóm

| Mã | Câu hỏi | Thí nghiệm |
| --- | --- | --- |
| RQ1 | Khi đánh giá đúng cách (không leakage), các mô hình ML nhận diện tấn công WiFi trong AWID3 tốt đến đâu? | E1, E2 |
| RQ2 | Cách chia dữ liệu ngẫu nhiên làm accuracy bị thổi phồng bao nhiêu so với chia theo thời gian? | E3 |
| RQ3 | Nhóm đặc trưng nào (802.11, IP/TCP, ứng dụng, đặc trưng cửa sổ) đóng góp nhiều nhất cho từng loại tấn công? | E4, E5, E7 |
| RQ4 | Mô hình có phát hiện được loại tấn công chưa từng thấy khi huấn luyện không? | E6 |

**Đóng góp riêng so với các project AWID3 phổ biến:**

- Chứng minh bằng số liệu ảnh hưởng của leakage (RQ2), thay vì chỉ báo một con số accuracy cao.
- Phân tích theo nhóm đặc trưng và theo từng loại tấn công, có giải thích bằng SHAP.
- Đánh giá tình huống unseen attack, sát với bài toán an ninh thực tế hơn.

## 3. Dữ liệu AWID3

Nhóm chỉ dùng bản CSV đã trích đặc trưng và không dùng pcap, vì pcap rất nặng và cần giải mã trước khi dùng. Mọi thông tin dưới đây cần được xác nhận lại khi nhận được dữ liệu thật (Tuần 1).

| Định dạng | Ưu điểm | Nhược điểm | Vai trò |
| --- | --- | --- | --- |
| CSV | Nhẹ, đã có nhãn, dùng ngay với pandas | Cột nhiều và thưa (nhiều NaN), không tự sửa được đặc trưng | Dữ liệu duy nhất |
| pcap | Đầy đủ, tự trích đặc trưng tùy ý | Rất lớn, phải giải mã, tốn giờ xử lý | Không dùng |

**Việc cần làm ngay trong Tuần 1 (data card):**

- Ghi lại số dòng, số cột, kiểu dữ liệu, tỉ lệ thiếu theo cột, phân bố nhãn theo từng file.
- Xác định cột nào là nhãn, cột nào là thời gian, cột nào là định danh (MAC, SSID, số thứ tự frame).
- Kiểm tra mỗi file tấn công có chứa traffic Normal riêng không, vì điều này quyết định cách chia dữ liệu ở mục 5.
- Kiểm tra CSV còn cột thời gian và cột nguồn phát (BSSID hoặc MAC nguồn) không, vì E5 và cách chia theo thời gian phụ thuộc vào chúng; nếu thiếu, E5 làm bản đơn giản hơn hoặc bỏ, và chia theo file capture.

**Đặc điểm cần lưu ý:** dữ liệu mất cân bằng mạnh; mạng mục tiêu là WPA2-Enterprise có bảo vệ management frame nên một số tấn công (như Deauth) khó nhận diện hơn trên mạng WiFi thông thường; dữ liệu lấy từ môi trường lab nên kết quả không đại diện cho mọi mạng thực tế. Nhóm cần nêu điểm cuối này trong phần hạn chế của báo cáo.

**Giảm dung lượng (nếu cần xuống khoảng 5-10GB):**

- Xử lý từng file riêng, ép kiểu (float64 → float32, chuỗi → category), lưu Parquet rồi mới gộp.
- Bỏ cột trước khi bỏ dòng (cột định danh, hằng số, gần như toàn trống).
- Giảm mẫu lớp đa số (Normal), giữ toàn bộ lớp thiểu số.
- Lấy theo đoạn thời gian liên tục, không lấy ngẫu nhiên từng dòng.
- Chia train/validation/test trước, chỉ giảm mẫu tập train; giữ tập test gần phân bố thật.
- Ghi tỉ lệ giảm mẫu và `random_state` vào README và báo cáo.

## 4. Thiết kế thí nghiệm

Bảy thí nghiệm E1 đến E7 trả lời bốn câu hỏi ở mục 2; E1 đến E4 cộng E7 là phần lõi bắt buộc, E5 và E6 là phần nâng cao.

| Mã | Thí nghiệm | Cách làm | Chỉ số chính | Trả lời |
| --- | --- | --- | --- | --- |
| E1 | Nhị phân Normal vs Attack | Tất cả mô hình, chia theo thời gian | F1 của lớp Attack, PR-AUC | RQ1 |
| E2 | Đa lớp theo loại tấn công | Tất cả mô hình, cùng cách chia | Macro-F1, F1 từng lớp, confusion matrix | RQ1 |
| E3 | Chia ngẫu nhiên vs chia theo thời gian | Cùng một mô hình, hai cách chia | Chênh lệch macro-F1 giữa hai cách | RQ2 |
| E4 | Ablation theo nhóm đặc trưng | Chỉ 802.11, thêm IP/TCP, thêm tầng ứng dụng, tất cả | Macro-F1 theo từng bộ đặc trưng | RQ3 |
| E5 | Đặc trưng cửa sổ thời gian | Gom 1 giây theo nguồn phát: số frame deauth, tỉ lệ beacon, số BSSID trên cùng SSID | Macro-F1, nhất là Evil Twin, Rogue AP, Deauth | RQ3 |
| E6 | Tấn công chưa từng thấy | Bỏ một loại tấn công khỏi train, test trên loại đó; Isolation Forest/Autoencoder train chỉ trên Normal | Recall của loại bị bỏ, tỉ lệ cảnh báo giả | RQ4 |
| E7 | Giải thích mô hình | SHAP và permutation importance trên mô hình tốt nhất, theo từng loại tấn công | Top đặc trưng mỗi lớp | RQ3 |

**Quy tắc chung cho mọi thí nghiệm**

- Cùng `random_state`, cùng tập train/validation/test, ghi cấu hình vào file riêng cho từng lần chạy.
- Chạy mỗi cấu hình ít nhất 3 hạt giống ngẫu nhiên, báo cáo trung bình và độ lệch chuẩn.
- Chọn siêu tham số chỉ trên validation, tập test chỉ chạm vào một lần ở bước cuối.
- Không dùng accuracy làm chỉ số duy nhất; luôn kèm macro-F1 và F1 từng lớp.

## 5. Tiền xử lý và chia dữ liệu chống leakage

Leakage là rủi ro lớn nhất của dự án: các packet liền kề trong cùng một lần capture rất giống nhau, nên chia ngẫu nhiên sẽ cho điểm gần 100% nhưng vô nghĩa. Vì vậy mọi bước dưới đây có thứ tự cố định và Người 1 chịu trách nhiệm kiểm tra.

1. **Nạp dữ liệu:** đọc CSV theo từng khối, ép kiểu hợp lý, lưu sang định dạng Parquet để các bước sau nhanh và nhẹ RAM.
2. **Loại cột dễ gây học thuộc:** MAC, IP, SSID, số thứ tự frame, timestamp thô. Giữ timestamp lại ở bước riêng chỉ để chia dữ liệu và tạo đặc trưng cửa sổ.
3. **Xử lý giá trị thiếu:** nhiều cột trống vì packet không có header đó. Điền giá trị sentinel (ví dụ -1) và thêm cờ "có header X", không xóa bừa cột hay dòng.
4. **Làm sạch:** bỏ cột hằng số, cột trùng lặp, chuyển cột hex/categorical sang số.
5. **Chia dữ liệu theo thời gian trong từng file:** khoảng 70% đầu cho train, 15% giữa cho validation, 15% cuối cho test, có khoảng đệm giữa các phần để packet liền kề không rơi vào hai tập.
6. **Fit chỉ trên train:** scaler, chọn đặc trưng, SMOTE (nếu dùng) đều chỉ fit trên tập train rồi áp dụng sang validation/test.
7. **Mất cân bằng lớp:** ưu tiên class weight và undersample lớp Normal; SMOTE chỉ là tùy chọn, chỉ trên train.
8. **Đặc trưng cửa sổ (E5):** gom theo cửa sổ 1 giây và theo nguồn phát trước khi bỏ timestamp. Chỉ làm được nếu CSV còn cột thời gian và cột nguồn phát.

**Kiểm tra bắt buộc trước khi báo kết quả:** nếu một mô hình đạt gần 100% trên test, dừng lại và rà soát cột định danh cùng cách chia, vì đó thường là dấu hiệu leakage.

**Điểm cần chốt sớm:** nếu CSV không có sẵn một file Normal riêng hoặc timestamp không đáng tin, nhóm đổi sang chia theo file capture (train trên một số file, test trên file khác) và ghi rõ trong báo cáo.

## 6. Mô hình và quy trình huấn luyện

Nhóm đi từ mô hình đơn giản đến phức tạp và chỉ giữ mô hình phức tạp nếu nó thắng baseline có ý nghĩa.

| Nhóm | Mô hình | Dùng cho | Ghi chú |
| --- | --- | --- | --- |
| Baseline | Logistic Regression, Decision Tree | E1, E2 | Mốc so sánh, dễ giải thích |
| Cây tổ hợp | Random Forest, LightGBM (hoặc XGBoost) | E1 đến E5 | Dự kiến là nhóm tốt nhất trên dữ liệu bảng |
| Mạng nơ-ron | MLP; 1D-CNN hoặc LSTM trên chuỗi cửa sổ nếu còn thời gian | E1, E2, E5 | Phần cắt đầu tiên nếu trễ tiến độ |
| Bất thường | Isolation Forest, Autoencoder | E6 | Huấn luyện chỉ trên traffic Normal |

**Quy trình mỗi mô hình**

1. Huấn luyện trên train với cấu hình mặc định để lấy mốc.
2. Tinh chỉnh siêu tham số (random search hoặc Optuna) chỉ dựa trên validation.
3. Huấn luyện lại với cấu hình tốt nhất, chạy 3 hạt giống, lưu mô hình và kết quả vào thư mục thí nghiệm.
4. Đo thêm thời gian huấn luyện và thời gian suy luận mỗi mẫu, để bàn về khả năng triển khai.

**Tài nguyên:** nếu máy cá nhân yếu, dùng Google Colab hoặc Kaggle cho phần huấn luyện; mô hình cây không cần GPU, chỉ MLP và Autoencoder cần.

## 7. Đánh giá, giải thích và phân tích lỗi

Kết quả có giá trị khi nhóm giải thích được vì sao mô hình đúng và vì sao nó sai, không chỉ đưa ra một con số.

**Chỉ số báo cáo**

- Precision, Recall, F1 từng lớp; macro-F1 làm chỉ số tổng hợp chính.
- Confusion matrix chuẩn hóa theo hàng để thấy lớp nào bị nhầm với lớp nào.
- ROC-AUC và PR-AUC cho bài toán nhị phân; PR-AUC quan trọng hơn khi dữ liệu mất cân bằng.
- Với phát hiện bất thường: tỉ lệ cảnh báo giả trên traffic Normal ở ngưỡng cố định.
- Độ lệch chuẩn qua 3 hạt giống và thời gian suy luận mỗi mẫu.

**Giải thích mô hình (E7)**

- SHAP toàn cục và theo từng loại tấn công để chỉ ra đặc trưng quyết định.
- Gắn đặc trưng với kiến thức giao thức: ví dụ khung quản lý 802.11 với Deauth/Disassoc, thông tin SSID/BSSID với Evil Twin.
- Đối chiếu kết quả với bài báo AWID3 gốc để xem nhận định có nhất quán không.

**Phân tích lỗi**

- Với mỗi lớp có F1 thấp, xem 20 mẫu bị phân loại sai và nêu nguyên nhân (thiếu mẫu, đặc trưng không phân biệt được, nhãn nhiễu).
- Bàn riêng các tấn công cần ngữ cảnh theo thời gian (Evil Twin, Rogue AP, Deauth flood) và cho thấy đặc trưng cửa sổ ở E5 cải thiện được bao nhiêu.
- Ghi các hạn chế: dữ liệu lab, một cấu hình mạng, chưa kiểm thử trên mạng khác.

## 8. Demo replay

Demo phát lại một đoạn dữ liệu test theo thứ tự thời gian qua mô hình tốt nhất và hiển thị cảnh báo trên một dashboard nhỏ, không cần card WiFi hay môi trường thật.

**Chức năng tối thiểu (Streamlit)**

- Chọn một đoạn dữ liệu test (ví dụ đoạn có Deauth hoặc Evil Twin) và tốc độ phát lại.
- Biểu đồ số cảnh báo theo thời gian, bảng cảnh báo gần nhất với loại tấn công dự đoán và độ tin cậy.
- So sánh nhãn thật với dự đoán để người xem thấy chỗ mô hình đúng và sai.
- Panel hiển thị 5 đặc trưng SHAP quan trọng nhất cho cảnh báo đang chọn.

**Nếu còn thời gian:** thêm quy tắc đơn giản "một SSID xuất hiện với nhiều BSSID" cho Evil Twin và so sánh với dự đoán của mô hình.

**Cách trình bày:** demo dài khoảng 3 phút, kịch bản cố định, có sẵn ảnh chụp dự phòng nếu demo trực tiếp bị lỗi.

## 9. Phân công 3 người

Mỗi người có một phần lõi riêng và một phần review chéo, để không ai là điểm nghẽn duy nhất. Điền tên thành viên vào cột đầu tiên.

| Thành viên | Vai trò | Phụ trách chính | Thí nghiệm | Review chéo |
| --- | --- | --- | --- | --- |
| Nhất | Dữ liệu | Nạp dữ liệu, EDA, tiền xử lý, chia dữ liệu, đặc trưng cửa sổ, kiểm tra leakage | Chuẩn bị cho E1 đến E5 | Kiểm tra pipeline của hai người còn lại |
| Phát | ML truyền thống | Baseline, Random Forest, LightGBM, tinh chỉnh, SHAP | E1, E2, E3, E4, E7 | Duyệt cách đánh giá và biểu đồ |
| Huy | Deep learning và bất thường | MLP, Isolation Forest, Autoencoder, demo Streamlit | E5 (phần mạng nơ-ron), E6 | Duyệt thiết kế thí nghiệm unseen attack |

**Việc chung cả nhóm**

- Họp ngắn 2 lần mỗi tuần (đầu tuần chốt việc, cuối tuần xem kết quả), ghi lại quyết định vào Issues của repo.
- Mỗi người viết phần báo cáo của mình theo từng tuần, không để dồn đến Tuần 6.
- Mọi thay đổi vào nhánh chính phải qua pull request và có ít nhất một người xem.
- Người 1 giao dữ liệu sạch cho cả nhóm vào cuối Tuần 2; trong lúc chờ, hai người còn lại dựng khung huấn luyện trên một mẫu nhỏ.

## 10. Lộ trình 6 tuần

Sáu tuần chia thành ba luồng chạy song song; hai mốc quan trọng nhất là dữ liệu sạch cuối Tuần 2 và xong E1 đến E4 cuối Tuần 4.

| Luồng | Việc | T1 | T2 | T3 | T4 | T5 | T6 |
| --- | --- | :-: | :-: | :-: | :-: | :-: | :-: |
| Người 1: Dữ liệu | Tải dữ liệu, EDA, data card | ■ | | | | | |
| | Tiền xử lý, chia theo thời gian | | ■ | | | | |
| | Đặc trưng cửa sổ (E5) | | | ■ | ■ | | |
| Người 2: ML truyền thống | Baseline và khung đánh giá | | ■ | | | | |
| | RF, LightGBM, E1 đến E4 | | | ■ | ■ | | |
| | SHAP, phân tích lỗi (E7) | | | | | ■ | |
| Người 3: DL và bất thường | MLP, Isolation Forest, Autoencoder | | ■ | ■ | ■ | | |
| | E6 unseen attack và demo | | | | ■ | ■ | |
| Cả nhóm | Hoàn thiện báo cáo, slide | | | | | | ■ |

**Điều kiện qua mỗi mốc**

- Cuối Tuần 2: dữ liệu sạch dạng Parquet, hàm chia dữ liệu nằm trong `src/`, baseline E1 chạy được.
- Cuối Tuần 4: E1 đến E4 có đủ 3 hạt giống; MLP và mô hình bất thường đã chạy thử.
- Cuối Tuần 5: dừng thêm thí nghiệm mới, chỉ còn SHAP, phân tích lỗi và demo; Tuần 6 dành cho báo cáo và tập thuyết trình.

## 11. Cấu trúc repo và quy ước làm việc

Một repo duy nhất, cấu trúc cố định từ Tuần 1, để mọi kết quả tái lập được bằng vài lệnh.

```text
awid3-wifi-ml/
├── data/              # không commit; README ghi cách tải và đặt file
├── notebooks/         # 01_eda, 02_preprocess, 03_baseline, ...
├── src/               # load.py, features.py, split.py, models.py, evaluate.py
├── experiments/       # E1..E7: file cấu hình và kết quả từng lần chạy
├── app/               # demo Streamlit
├── reports/           # hình, bảng, bản nháp báo cáo
├── requirements.txt
└── README.md
```

**Quy ước**

- Logic dùng chung (nạp, chia, đánh giá) nằm trong `src/`, notebook chỉ gọi hàm; không ai tự viết lại cách chia dữ liệu.
- Mỗi thí nghiệm có một file cấu hình (mô hình, tham số, hạt giống, cách chia) và lưu kết quả cùng tên.
- Dữ liệu lớn và mô hình nặng không đưa vào Git; ghi đường dẫn và checksum trong README.
- Nhánh `main` luôn chạy được; làm việc trên nhánh riêng và merge bằng pull request.
- README ghi rõ: cài đặt, tải dữ liệu, chạy từng thí nghiệm, tái tạo bảng và hình trong báo cáo.

## 12. Rủi ro và phương án dự phòng

| Rủi ro | Dấu hiệu | Phương án |
| --- | --- | --- |
| Không tải được dữ liệu (link tải lỗi hoặc hết hạn sau 7 ngày) | Tải dừng giữa chừng, lỗi truy cập | Gửi lại email cho nhóm AWID ngay; chỉ cần có CSV là làm được toàn bộ dự án |
| Hết RAM khi nạp dữ liệu | Notebook bị kill, máy treo | Đọc theo khối, dùng Parquet, downsample Normal, chạy trên Colab hoặc Kaggle |
| Leakage làm kết quả ảo | Macro-F1 gần 100% ngay lần chạy đầu | Rà soát cột định danh và cách chia; đổi sang chia theo file capture nếu cần |
| Một số lớp quá ít mẫu | F1 gần 0 ở vài lớp | Gộp phân tích theo nhóm (quản lý 802.11, mạng, ứng dụng) hoặc báo cáo thẳng hạn chế đó |
| Trễ tiến độ | Quá Tuần 3 mà chưa xong E2 | Cắt theo thứ tự: deep learning nâng cao, rồi E5, rồi demo; giữ E1 đến E4 và E7 |
| Thành viên bị nghẽn việc | Một phần chờ phần khác quá 3 ngày | Dựng khung trên mẫu nhỏ để làm song song; họp ngắn gỡ vướng |

## 13. Cấu trúc báo cáo và thuyết trình

**Báo cáo**

1. Giới thiệu: bài toán, các tấn công WiFi trong AWID3, câu hỏi nghiên cứu.
2. Công trình liên quan: bài báo AWID3 gốc và một số nghiên cứu IDS WiFi dùng ML.
3. Dữ liệu và tiền xử lý: nhấn mạnh cách chia dữ liệu chống leakage.
4. Phương pháp: các mô hình và thiết kế thí nghiệm E1 đến E7.
5. Kết quả: so sánh mô hình, kết quả leakage (E3), ablation (E4), cửa sổ thời gian (E5), unseen attack (E6).
6. Thảo luận: SHAP, phân tích lỗi, ý nghĩa đối với an ninh WiFi.
7. Hạn chế và hướng phát triển: dữ liệu lab, chưa triển khai thực tế, có thể mở rộng sang bắt gói tin trực tiếp.
8. Tài liệu tham khảo, gồm bài báo AWID3 và mọi repo hoặc mã nguồn đã tham khảo.

**Thuyết trình (khoảng 10 đến 12 phút)**

- Vấn đề và dữ liệu (2 phút), thiết kế thí nghiệm (2 phút), kết quả chính và phát hiện về leakage (3 phút), giải thích mô hình (2 phút), demo (3 phút).
- Mỗi người trình bày phần mình phụ trách để trả lời được câu hỏi về phần đó.
- Chuẩn bị sẵn 5 câu hỏi dự kiến, ví dụ: vì sao chia theo thời gian, vì sao chọn macro-F1, mô hình có dùng được trên mạng khác không.

## 14. Checklist hoàn thành

**Lõi (bắt buộc)**

- [ ] Có dữ liệu AWID3 (CSV) và data card đã điền
- [ ] Pipeline tiền xử lý chạy từ đầu đến cuối, chia dữ liệu theo thời gian, đã kiểm tra leakage
- [ ] E1 và E2 xong với ít nhất 4 mô hình, 3 hạt giống
- [ ] E3 xong, có biểu đồ so sánh chia ngẫu nhiên và chia theo thời gian
- [ ] E4 xong, có bảng ablation theo nhóm đặc trưng
- [ ] E7 xong, có SHAP cho ít nhất 5 loại tấn công
- [ ] Phân tích lỗi cho các lớp F1 thấp
- [ ] README đủ để người khác tái lập kết quả
- [ ] Báo cáo và slide hoàn chỉnh, trích dẫn đủ nguồn

**Nâng cao (nếu còn thời gian)**

- [ ] E5 đặc trưng cửa sổ thời gian
- [ ] E6 unseen attack và phát hiện bất thường
- [ ] Demo Streamlit chạy được, có ảnh chụp dự phòng
- [ ] MLP, 1D-CNN hoặc LSTM