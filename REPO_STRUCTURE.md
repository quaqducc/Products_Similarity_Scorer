## Cấu trúc repository: Product_Similarity

Tài liệu này mô tả nhanh các thư mục, module và script chính trong repo, giúp bạn định vị thành phần cần chỉnh sửa/chạy.

### Sơ đồ thư mục tổng quan

```
Product_Similarity/
  ├─ cli.py
  ├─ eval.py
  ├─ METRICS.md
  ├─ README.md
  ├─ requirements.txt
  ├─ kaggle_test.ipynb
  ├─ examples/
  │   └─ KAGGLE_GUIDE.md
  ├─ product_similarity/
  │   ├─ __init__.py
  │   ├─ agents.py
  │   ├─ judge.py
  │   ├─ model.py
  │   ├─ pipeline.py
  │   ├─ prompt.py
  │   └─ retriever.py
  ├─ data/
  │   ├─ 100_samples.csv
  │   ├─ 75_samples.csv
  │   ├─ fewshot_cases.json
  │   └─ nice_chunks.json
  ├─ data_nice_cls/
  │   └─ group_1.json ... group_45.json
  ├─ tools/
  │   ├─ merge_nice_cls.py
  │   └─ prepare_75_samples.py
  └─ spsc_data/
      ├─ SPSC.xlsx
      ├─ SPSC_Tree_Builder.ipynb
      └─ spsc_data/
          └─ spsc_tree.json
```

### Thư mục và module chính

- `product_similarity/` (thư viện lõi)
  - `pipeline.py`: Hàm đầu cuối `run_similarity(...)` dựng prompt, truy xuất ngữ cảnh NICE và tùy chọn gọi mô hình (HF hoặc Chat API). Trả về `contexts`, `prompt`, `output_text`, `scores`.
  - `retriever.py`: Truy xuất ngữ cảnh từ `data/nice_chunks.json` theo từ khóa hoặc trực tiếp theo số class (`contexts_from_class_numbers`). Có cache dữ liệu NICE.
  - `prompt.py`: Xây dựng prompt gồm hướng dẫn, few-shot, context và case mới. Chuẩn định dạng đầu ra với các mục Nature/Purpose/Overall.
  - `model.py`:
    - `LLMWrapper`: gọi mô hình HuggingFace (text2text-generation).
    - `ChatAPIWrapper`: gọi API Chat chuẩn OpenAI-compatible (ví dụ NVIDIA).
  - `agents.py`: Định nghĩa `FactorAgent` đánh giá theo từng tiêu chí (vd. Nature, Intended Purpose, Channel of trade), trả về reasoning + `Score` 0–4. Hỗ trợ HF hoặc Chat API.
  - `judge.py`: `LLMJudge` gộp điểm các tiêu chí bằng trọng số, xuất `overall_similarity` (số nguyên 0–4).
  - `__init__.py`: Khởi tạo gói.

- `data/` (dữ liệu chạy và đánh giá)
  - `fewshot_cases.json`: Few-shot ví dụ cho prompt.
  - `nice_chunks.json`: Dữ liệu NICE đã tiền xử lý từ `data_nice_cls/`.
  - `100_samples.csv`, `75_samples.csv`: Mẫu/nhãn dùng đánh giá.

- `data_nice_cls/`: Nguồn dữ liệu NICE dạng nhóm (`group_*.json`) dùng để hợp nhất thành `data/nice_chunks.json`.

- `tools/` (tiện ích)
  - `merge_nice_cls.py`: Hợp nhất `data_nice_cls/` → `data/nice_chunks.json`.
  - `prepare_75_samples.py`: Chuẩn bị/tinh chỉnh dữ liệu mẫu 75.
  - (CLI có subcommand `build-tree` tham chiếu tool dựng cây từ Excel; nếu tool đó không có, có thể bỏ qua subcommand này.)

- `examples/`
  - `KAGGLE_GUIDE.md`: Hướng dẫn cho kịch bản trên Kaggle/notebook.

- `spsc_data/`
  - `SPSC.xlsx`, `SPSC_Tree_Builder.ipynb`: Tài liệu và notebook xây dựng cây SPSC.
  - `spsc_data/spsc_tree.json`: Cây phân loại SPSC đã xuất.

### Script/chạy nhanh

- CLI tổng (`cli.py`):
  - `run`: chạy đánh giá hai mô tả sản phẩm, có thể chỉ dựng prompt hoặc chạy mô hình HF/Chat API.
  - `build-nice`: hợp nhất dữ liệu NICE từ `data_nice_cls/` vào `data/nice_chunks.json`.
  - `build-tree`: (tùy chọn) dựng JSON cây phân cấp từ Excel nếu có tool tương ứng.

- Đánh giá đa agent (`eval.py`):
  - Chạy Analyzer (dựng/hoặc sinh văn bản phân tích), chạy nhiều `FactorAgent`, rồi `LLMJudge` gộp điểm.
  - Xuất `metrics` (ví dụ `exact_match`) và `results` chi tiết.

### Phụ thuộc & môi trường

- `requirements.txt`: Danh sách thư viện cần thiết. Cài đặt:

```bash
pip install -r requirements.txt
```

### Gợi ý luồng làm việc

1) Chuẩn bị dữ liệu NICE (nếu chưa có):
```bash
python cli.py build-nice
```
2) Chạy thử pipeline cho 2 mô tả sản phẩm:
```bash
python cli.py run --p1 "Make-up preparations" --p2 "Tissues of paper for removing make-up"
```
3) Đánh giá theo bộ mẫu CSV và đa agent + judge:
```bash
python eval.py --csv data/100_samples.csv
```


