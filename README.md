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
  │   ├─ pipeline.py
  │   ├─ agents.py           # Multi-agent theo tiêu chí (Nature, Purpose, ...)
  │   └─ judge.py            # Judge gộp điểm các tiêu chí
  ├─ tools/
  │   └─ merge_nice_cls.py
  ├─ data/
  │   ├─ fewshot_cases.json
  │   └─ nice_chunks.json  # sinh từ data_nice_cls
  ├─ data_nice_cls/
  │   └─ group_*.json
  ├─ eval.py                 # Orchestrator Analyzer -> Agents -> Judge + metrics
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

- Không chạy mô hình (chỉ xem prompt và context, `scores` rỗng). Nếu đã biết class số của hai sản phẩm, thêm `--class1`, `--class2` để lấy context trực tiếp theo class (bỏ qua truy xuất từ khóa):

```bash
python cli.py run --p1 "Make-up preparations" --p2 "Tissues of paper for removing make-up"
# hoặc (biết sẵn class):
python cli.py run --p1 "Make-up preparations" --p2 "Tissues of paper for removing make-up" --class1 3 --class2 16
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
- `--class1`, `--class2`: số class NICE tương ứng cho p1, p2; nếu truyền thì sẽ trích context trực tiếp từ class đó.

CLI sẽ in JSON gồm `contexts`, `prompt`, `output_text`, và `scores` (nếu có mô hình).

## Multi-agent + Judge (mới)

Chúng tôi bổ sung mô-đun đa agent theo từng tiêu chí và Judge để gộp điểm:

- `product_similarity/agents.py`: lớp `FactorAgent` chạy từng tiêu chí (Nature, Intended Purpose, Channel of trade, ...), mỗi agent có thể dùng model riêng (mặc định `mistralai/Mistral-7B-Instruct-v0.2`).
- `product_similarity/judge.py`: lớp `LLMJudge` gộp điểm theo trọng số và xuất `overall_similarity`.
- `eval.py`: chạy Analyzer (giữ nguyên prompt từ `product_similarity`), rồi gọi các agent và Judge, tính đơn giản Exact Match với nhãn vàng trong CSV.

Cách chạy:

```bash
python eval.py --csv data/100_samples.csv \
  --analyzer-model "google/flan-t5-base" \
  --agent-model "mistralai/Mistral-7B-Instruct-v0.2" \
  --device -1 --max-new-tokens 256

# Hoặc dùng Chat API (OpenAI-compatible, ví dụ NVIDIA):
python eval.py --csv data/100_samples.csv \
  --chat-api-base-url "https://integrate.api.nvidia.com/v1" \
  --chat-api-key "$YOUR_KEY" \
  --chat-api-model "meta/llama-3.1-8b-instruct"
```

Kết quả xuất gồm `metrics` (ví dụ `exact_match`) và `results` chi tiết cho từng hàng.

Xem hướng dẫn notebook Kaggle: `examples/KAGGLE_GUIDE.md`.

## Sử dụng như thư viện

```python
from product_similarity import run_similarity

res = run_similarity(
    "Make-up preparations",
    "Tissues of paper for removing make-up",
    class_1=3,
    class_2=16,
    model_name=None,  # hoặc 'google/flan-t5-base'
)
print(res["scores"])
```

## Ghi chú
- Khi có `class_1`, `class_2`, hệ thống sẽ dùng trực tiếp dữ liệu theo class và bỏ qua keyword retrieval.
- Khi chạy mô hình cục bộ lần đầu, Transformers sẽ tải model/tokenizer từ HuggingFace Hub.
- Nếu máy yếu, cân nhắc để `model_name=None` và chỉ xuất prompt.
