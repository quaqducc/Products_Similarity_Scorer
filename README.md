# Product Similarity (NICE Classification)

Dự án refactor từ `Project 3` để đánh giá độ tương đồng hàng hóa/dịch vụ theo phân loại NICE. Cấu trúc gọn gàng, có CLI và tài liệu hướng dẫn.

## Cấu trúc thư mục

```
.Lab/Product_Similarity/
  ├─ product_similarity/
  │   ├─ __init__.py
  │   ├─ prompt.py
  │   ├─ retriever.py
  │   ├─ model.py
  │   └─ pipeline.py
  ├─ tools/
  │   └─ merge_nice_cls.py
  ├─ data/
  │   ├─ fewshot_cases.json
  │   └─ nice_chunks.json  # sinh từ data_nice_cls
  ├─ data_nice_cls/
  │   └─ group_*.json
  ├─ requirements.txt
  └─ cli.py
```

## Cài đặt

- Python 3.10+
- (Tùy chọn) Cài Transformers để chạy mô hình cục bộ:

```bash
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu NICE

Nếu đã có `data/nice_chunks.json` thì bỏ qua bước này. Nếu chưa có, xây dựng từ `data_nice_cls`:

```bash
python cli.py build-nice
```

Kết quả sẽ ghi vào `data/nice_chunks.json`.

## Chạy đánh giá tương đồng

Chạy pipeline qua CLI. Có thể chọn chạy không mô hình (chỉ build prompt + retriever) hoặc chạy với mô hình HF cục bộ.

- Không chạy mô hình (chỉ xem prompt và context, `scores` rỗng):

```bash
python cli.py run --p1 "Make-up preparations" --p2 "Tissues of paper for removing make-up"
```

- Chạy với mô hình HF (ví dụ `google/flan-t5-base`):

```bash
python cli.py run --p1 "Paints" --p2 "construction materials" --model google/flan-t5-base --device -1 --max-new-tokens 256
```

Tham số chính:
- `--p1`, `--p2`: mô tả hai sản phẩm
- `--max-fewshot`: số lượng ví dụ few-shot (mặc định 2)
- `--top-k`: số context NICE lấy từ retriever (mặc định 3)
- `--model`: id mô hình HF (bỏ trống để không chạy mô hình)
- `--device`: -1 dùng CPU, 0 dùng GPU

CLI sẽ in JSON gồm `contexts`, `prompt`, `output_text`, và `scores` (nếu có mô hình).

## Sử dụng như thư viện

```python
from product_similarity import run_similarity

res = run_similarity(
    "Chemicals for industrial use",
    "chemical additives for detergents",
    model_name=None,  # hoặc 'google/flan-t5-base'
)
print(res["scores"])
```

## Ghi chú
- Bộ retriever hiện tại dựa trên keyword đơn giản, có thể thay thế bằng BM25/embeddings.
- Khi chạy mô hình cục bộ lần đầu, Transformers sẽ tải model/tokenizer từ HuggingFace Hub.
- Nếu máy yếu, cân nhắc để `model_name=None` và chỉ xuất prompt.
