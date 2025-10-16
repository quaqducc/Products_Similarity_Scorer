# Metrics cho Product Similarity

Tài liệu mô tả các chỉ số đánh giá được tính trong notebook `kaggle_test.ipynb`.

## Đầu vào
- `overall`: điểm dự đoán của mô hình, thang 0–4.
- `label`: nhãn ground-truth trong `75_samples.csv`, cột `Level of similarity` (số nguyên 0–4).
- (Tuỳ chọn) `channels_of_trade`: cột mô tả kênh thương mại từ dữ liệu đầu vào.

## Chỉ số
- **Accuracy (exact match)**
  - Định nghĩa: tỉ lệ mẫu mà điểm dự đoán `overall` (làm tròn xuống int) trùng khớp `label`.
  - Công thức: `accuracy = mean(int(overall) == label)` trên tập các dòng có đủ `overall` và `label`.

- **MSE (Mean Squared Error)**
  - Định nghĩa: sai số bình phương trung bình giữa dự đoán và nhãn.
  - Công thức: `mse = mean((overall - label) ** 2)` trên tập các dòng hợp lệ.

- (Tuỳ chọn) Có thể bổ sung MAE, RMSE, hoặc phân tích theo từng kênh `channels_of_trade` nếu cần.

## Ghi chú
- Các dòng thiếu `overall` hoặc `label` sẽ bị loại khỏi tính toán.
- Ngưỡng/chiến lược phân loại nhị phân không dùng trong bộ chỉ số chính; đánh giá so sánh trực tiếp trên thang điểm 0–4.

