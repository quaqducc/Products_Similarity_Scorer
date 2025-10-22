# Hướng dẫn chạy notebook Kaggle

Notebook `kaggle_test.ipynb` cho phép bạn chạy pipeline Multi-agent + Judge trên Kaggle.

## 1) Chuẩn bị Datasets (Inputs)
- Repo dataset: chứa toàn bộ mã nguồn của thư mục dự án (bao gồm `product_similarity/`, `eval.py`, `tools/`, ...).
- Data dataset: chứa thư mục `data/` (có `fewshot_cases.json`, `nice_chunks.json`). Nếu chưa có `nice_chunks.json`, có thể tạo từ `data_nice_cls` bằng `tools/merge_nice_cls.py`.

Trên Kaggle, vào tab Add data:
- Thêm dataset repo (ví dụ: `product-similarity-scorer`).
- Thêm dataset data (ví dụ: `products-similarity-scorer-data`).

## 2) Cấu hình notebook
Trong Cell cấu hình của notebook:
- `USE_ANALYZER_MODEL`: điền HF model id (vd: `google/flan-t5-base`) hoặc để `None` để bỏ qua sinh text từ Analyzer.
- `AGENT_MODEL_ID`: model cho các FactorAgent (mặc định `mistralai/Mistral-7B-Instruct-v0.2`).
- `USE_CHAT_API`: `True` nếu dùng OpenAI-compatible API (vd NVIDIA), khi đó điền `CHAT_API_BASE_URL`, `CHAT_API_MODEL`, và set secret `NV_API_KEY` trong Settings → Secrets.
- `DEVICE`: `-1` CPU hoặc `0` GPU (nếu runtime có GPU).
- `MAX_NEW_TOKENS`: giới hạn token sinh.

## 3) Chạy đơn lẻ
Cell “Single-run example” sẽ tạo một CSV 1 dòng tạm thời và gọi `evaluate_dataset` để chạy Analyzer → Agents → Judge.
- Kết quả in `metrics` và `pred_overall` của dòng đó.

## 4) Chạy batch từ CSV
- Notebook sẽ ưu tiên `data/75_samples.csv`, nếu không có sẽ dùng `data/100_samples.csv`.
- Cell “Batch evaluation” sẽ gọi `evaluate_dataset(csv_path, ...)` và xuất:
  - `/kaggle/working/batch_results.csv`: gồm `p1`, `p2`, `pred`, `label`.
  - `/kaggle/working/metrics.json`: gồm `exact_match` và số lượng hàng có nhãn.

## 5) Lưu ý
- Nếu dùng Chat API NVIDIA, mẫu cấu hình:
  - `CHAT_API_BASE_URL = "https://integrate.api.nvidia.com/v1"`
  - `CHAT_API_MODEL = "meta/llama-3.1-8b-instruct"` (hoặc model khác trong tenant của bạn)
  - Tạo Secret `NV_API_KEY` chứa API key.
- Nếu GPU không đủ, có thể chạy Analyzer `None` để chỉ dùng Agents, hoặc giảm `MAX_NEW_TOKENS`.
