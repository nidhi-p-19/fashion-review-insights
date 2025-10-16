# 🧵 Fashion Review Insights — LLM-Powered NLP Summarization System

> Transform unstructured customer reviews into structured, attribute-level insights across six key fashion dimensions using **Gemini AI** and **Streamlit**.

---

## 📘 Project Summary

This project automatically reads customer reviews from Excel/CSV files and uses **Large Language Models (LLMs)** to generate:

- Attribute-level **sentiment summaries**
- **Positive/negative/neutral** metrics
- Concise **summary bullets**
- Representative **evidence snippets**
- An overall **product sentiment summary**

All results are displayed interactively using a **Streamlit dashboard** and can be exported in JSON, Excel, or Markdown formats.

---

## 🧩 Key Features

✅ Multi-sheet Excel ingestion (handles all product sheets)  
✅ LLM-based aspect extraction (Gemini 2.0/2.5 or OpenAI GPT)  
✅ Batch summarization for large review volumes  
✅ Attribute-wise aggregation & sentiment scoring  
✅ Interactive Streamlit UI with chips, bars & summaries  
✅ One-click Markdown summary export  
✅ Real-time caching for faster UI response  

---

## ⚙️ Core Attributes

| Attribute | Description | Example Keywords |
|------------|--------------|------------------|
| **Silhouette** | Shape or cut of the garment | flowy, A-line, fitted |
| **Proportion / Fit** | Sizing and comfort | true to size, runs large |
| **Detail** | Design elements | embroidery, buttons, seams |
| **Color** | Shade or tone comments | fades, vibrant, dull |
| **Print / Pattern** | Surface design | stripes, floral, checks |
| **Fabric** | Material quality | cotton, silk, itchy, soft |

---

## 🧱 System Architecture

```
┌────────────────────────────┐
│  Excel / CSV Review Data   │
└──────────────┬─────────────┘
               │
               ▼ Data Preprocessing (pandas)
               │
               ▼ Prompt Builder (JSON schema)
               │
┌────────────────────────────┐
│   Gemini / OpenAI API      │
│   → Aspect Detection       │
│   → Sentiment Extraction   │
│   → Summary Generation     │
└──────────────┬─────────────┘
               │
               ▼ Post-Processing (Python)
               │
┌────────────────────────────┐
│  Streamlit Visualization   │
│   → Attribute chips        │
│   → Metrics & bullets      │
│   → Download report        │
└────────────────────────────┘
```

---

## 🧮 Sentiment Logic

Each attribute’s **net sentiment** is computed as:

```python
net_sentiment = (positive - negative) / max(1, mentions)
```

🟢 Positive → ≥ 0.25  
🟡 Neutral → between -0.25 and 0.25  
🔴 Negative → ≤ -0.25  

---

## 🛠️ Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/fashion-review-insights.git
cd fashion-review-insights
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Add Environment Variables
Create a `.env` file and add your API keys:
```bash
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

---

## 🚀 Usage

### ▶️ CLI Mode (Generate Summaries)
```bash
python summarize_reviews.py \
  --input "Data - 10 Styles Insights.xlsx" \
  --style-number 100146 \
  --provider gemini \
  --model gemini-2.5-flash
```

**Outputs:**
```bash
summary.json
summary.xlsx
```

### ▶️ Streamlit (Live Mode)
```bash
streamlit run app_live.py
```
Then:
- Upload Excel/CSV  
- Select sheet (or “All”)  
- Enter style number  
- Generate live summary  

### ▶️ Streamlit (Offline Viewer)
```bash
streamlit run app.py
```
Then upload your `summary.json` or folder of summaries.

---

## 📂 Repository Structure

```bash
├── app_live.py              # Streamlit app for live LLM summarization
├── app.py                   # Streamlit viewer for precomputed outputs
├── summarize_reviews.py     # CLI summarizer (core LLM pipeline)
├── requirements.txt
├── README.md
└── Data - 10 Styles Insights.xlsx  # Sample data
```

---

## 🧠 Technical Details

### 🔸 LLM Prompt Engineering
- Strict JSON schema for consistent parsing  
- Semantic synonym mapping for attributes  
- Limit snippets to ≤12 tokens  
- Temperature = 0.1 for factual consistency  
- Response MIME type set to `application/json`

### 🔸 Batch Processing
- Large inputs split into chunks of 60 reviews  
- Each batch summarized separately  
- Results merged with weighted average sentiment  

### 🔸 Post-Processing
- Merges all batches  
- Recomputes net sentiment  
- Keeps top 3 summary bullets & snippets per attribute  
- Ensures all 6 attributes are present in final JSON  

---

## 🧾 Example Output (JSON)

```json
{
  "product_id": "100146",
  "overall_summary": "Most customers appreciate the lightweight fabric and accurate fit...",
  "attributes": [
    {
      "name": "fabric",
      "mentions": 34,
      "positive": 28,
      "negative": 4,
      "neutral": 2,
      "net_sentiment": 0.70,
      "summary_bullets": [
        "Soft, breathable cotton ideal for summer",
        "Some users mention slight shrinkage"
      ],
      "evidence_snippets": [
        "fabric feels soft",
        "slightly shrinks after wash"
      ]
    }
  ]
}
```

---

## 🧰 Tech Stack

| Layer | Tool / Library | Purpose |
|--------|----------------|----------|
| Backend | Python 3.11 | Core orchestration |
| LLM API | Google Gemini 2.0 / 2.5 Flash | Summarization & Sentiment |
| Frontend | Streamlit | Interactive visualization |
| Data | Pandas, OpenPyXL | Excel processing |
| Deployment | Streamlit Cloud | Hosting & sharing |

---

## 📊 Example Run

| Step | Output |
|------|---------|
| Upload Excel | Multi-sheet data read & merged |
| Generate Summary | LLM processes 80 reviews |
| Dashboard | Attribute chips show 🟢 / 🟡 / 🔴 |
| Drilldown | Snippets + Summary Bullets |
| Download | Markdown one-pager per attribute |

---

## 💡 Future Enhancements

🔍 Add ranking module for snippet importance *(Sentence-BERT)*  
🌐 Support multi-language reviews *(English + regional)*  
📈 Sentiment trend graphs & radar charts  
🧾 Integration with Google Sheets API  

---

## 🏷️ Project Badges

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?logo=streamlit)
![Gemini AI](https://img.shields.io/badge/Gemini%20AI-LLM-yellow?logo=google)
![License](https://img.shields.io/badge/license-MIT-green)

---
'''
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)
'readme_created = True'
