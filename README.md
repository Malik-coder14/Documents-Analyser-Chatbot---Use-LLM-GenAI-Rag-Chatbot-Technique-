# Business Intelligence + Financial RAG Assistant

## 1. Install

```bash
pip install -r requirements.txt
```

For chart/image OCR, install Tesseract OCR separately on your operating system.

## 2. Run

```bash
streamlit run app.py
```

## 3. Upload

Supported:
- PDF
- CSV
- TXT
- Markdown
- PNG/JPG/WEBP chart images

## 4. Configure

In the left sidebar:
- Enter Groq API key
- Select LLM
- Set chunk size
- Set chunk overlap
- Set Top-K
- Set temperature

## 5. Recommended company files

Upload a mixture such as:

```text
annual_report_2024.pdf
annual_report_2025.pdf
sales.csv
inventory.csv
customers.csv
products.pdf
discount_policy.pdf
pricing_policy.pdf
expenses.csv
```

The app uses:
- Sentence Transformers for semantic retrieval
- Pandas for deterministic CSV calculations
- Groq for natural-language reasoning and explanation
- PDF/TXT/image extraction for unstructured sources

## Important

Column names are detected heuristically. For best results, use names such as:

Revenue:
`revenue`, `net_revenue`, `sales`, `sales_amount`

Expenses:
`expense`, `expenses`, `operating_expense`, `cost`

Profit:
`profit`, `gross_profit`, `net_profit`

Product:
`product`, `product_name`, `product_id`, `sku`

Customer:
`customer`, `customer_name`, `customer_id`

Inventory:
`inventory`, `stock`, `stock_quantity`, `units_in_stock`

Category:
`category`, `product_category`

Date:
`date`, `order_date`, `sales_date`, `transaction_date`

This is a portfolio/prototype implementation. For production use, add authentication,
document-level permissions, persistent vector storage, stronger evaluation,
PII controls, and a safer structured-query layer.
