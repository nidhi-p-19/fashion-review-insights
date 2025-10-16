from dotenv import load_dotenv
load_dotenv()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Powered Customer Review Summarizer (Fixed 6 Attributes)
- Reads CSV/XLSX
- Filters by style_number
- Builds the exact "word prompt" you specified
- Calls Gemini (default) or OpenAI
- Saves JSON result + a tidy Excel with per-attribute stats

Usage:
  python summarize_reviews.py --input data.xlsx --style-number D123 --provider gemini --model gemini-1.5-flash
"""

import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any

# ----------------------------
# LLM CLIENTS (Gemini / OpenAI)
# ----------------------------
def call_gemini(system_prompt: str, user_prompt: str, model: str = "gemini-2.5-flash") -> str:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var.")
    genai.configure(api_key=api_key)

    prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
    model_obj = genai.GenerativeModel(model)

    # optional: relax safety (reviews are benign)
    safety = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUAL: HarmBlockThreshold.BLOCK_NONE,
    }

    def _gen(max_tokens):
        return model_obj.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.9,
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
            },
            safety_settings=safety,
        )

    resp = _gen(3000)  # first try
    # robust extract
    text = None
    if getattr(resp, "text", None):
        text = resp.text
    elif resp.candidates:
        c0 = resp.candidates[0]
        # Inspect finish_reason for debugging
        print(f"[Gemini] finish_reason={getattr(c0, 'finish_reason', 'UNKNOWN')}")
        # Try to recover string from parts if present
        if getattr(c0, "content", None) and getattr(c0.content, "parts", None):
            try:
                text = "".join(getattr(p, "text", "") for p in c0.content.parts)
            except Exception:
                text = None

    # If still empty, retry once with stricter output expectations
    if not text:
        print("[Gemini] Retrying with stricter limits...")
        tighter = user_prompt + "\n\nReturn compact JSON. Limit evidence_snippets to at most 2 per attribute."
        resp = model_obj.generate_content(
            f"SYSTEM:\n{system_prompt}\n\nUSER:\n{tighter}",
            generation_config={
                "temperature": 0.1,
                "top_p": 0.9,
                "max_output_tokens": 2000,
                "response_mime_type": "application/json",
            },
            safety_settings=safety,
        )
        text = getattr(resp, "text", None)

    if not text:
        raise ValueError("Gemini returned no text (likely max_tokens/safety). Reduce --max-reviews and try again.")
    return text


def call_openai(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=4000
    )
    return resp.choices[0].message.content

# ----------------------------
# PROMPTS (exactly your spec)
# ----------------------------
SYSTEM_PROMPT = (
    "You are an expert NLP pipeline that extracts structured insights from product reviews.\n"
    "Your task is to analyze multiple customer reviews of a product and return a structured, JSON-formatted summary containing:\n"
    "- a general overview,\n"
    "- six fixed attributes (silhouette, proportion_or_fit, detail, color, print_or_pattern, fabric),\n"
    "- short evidence snippets and sentiment scores for each attribute,\n"
    "- and brief attribute-level summaries.\n"
    "Rules:\n"
    "- Return STRICT JSON only (no markdown, no commentary).\n"
    "- Use ONLY the provided text; do not add outside facts.\n"
    "- English only.\n"
)

USER_PROMPT_TEMPLATE = """You are given raw customer reviews for a product, each with a rating, headline, and comment.  
Perform the following steps carefully and return ONLY valid JSON.  

### Step 1. Review Understanding
- Combine each review’s headline and comment into a single text.
- Break the text into individual sentences for precise analysis.

### Step 2. Attribute Detection
- Use ONLY these six canonical attributes:
  ["silhouette", "proportion_or_fit", "detail", "color", "print_or_pattern", "fabric"]
- For every sentence, decide which attributes (zero or more) it mentions.
- Use semantic understanding and common fashion synonyms.
  - “A-line, boxy, flowy” → silhouette  
  - “true to size, runs small” → proportion_or_fit  
  - “buttons, embroidery” → detail  
  - “blue, red, fades” → color  
  - “polka dot, stripes” → print_or_pattern  
  - “cotton, silk, itchy, soft” → fabric  

### Step 3. Snippet & Sentiment Extraction
For every (sentence, attribute) pair:
- Extract a short verbatim snippet (≤ 12 tokens) that reflects the customer’s opinion.  
- Assign a sentiment label:  
  - "positive" → 1  
  - "negative" → -1  
  - "na" or unclear → 0  

Example object:
{{
 "sentence":"The fabric feels soft but the color fades fast.",
 "pairs":[
   {{"attribute":"fabric","snippet":"fabric feels soft","score":1}},
   {{"attribute":"color","snippet":"color fades fast","score":-1}}
 ]
}}

### Step 4. Aggregate per Attribute
For each of the six attributes:
- Count total mentions, positives (score = 1), negatives (score = -1), and neutrals (0).  
- Compute the attribute’s net sentiment = mean(score).

 ### Step 5. Generate Summaries
 a) Overall summary — 70–100 words describing the general sentiment across all attributes.  
 b) Attribute summaries — 2–3 bullets (35–60 words each) highlighting key opinions, both positive and negative, supported by snippets.
+Output limits:
+- For each attribute, return at most 3 evidence_snippets. Each snippet must be <= 12 tokens.


### Step 6. Output Schema
Return strictly in this JSON format:
{{
 "product_id": "{style_number}",
 "overall_summary": "string",
 "attributes": [
   {{
     "name": "silhouette",
     "mentions": int,
     "positive": int,
     "negative": int,
     "neutral": int,
     "net_sentiment": float,
     "summary_bullets": ["...","..."],
     "evidence_snippets": ["...", "..."]
   }},
   {{
     "name": "proportion_or_fit",
     "mentions": int,
     "positive": int,
     "negative": int,
     "neutral": int,
     "net_sentiment": float,
     "summary_bullets": ["...","..."],
     "evidence_snippets": ["...", "..."]
   }},
   {{
     "name": "detail",
     "mentions": int,
     "positive": int,
     "negative": int,
     "neutral": int,
     "net_sentiment": float,
     "summary_bullets": ["...","..."],
     "evidence_snippets": ["...", "..."]
   }},
   {{
     "name": "color",
     "mentions": int,
     "positive": int,
     "negative": int,
     "neutral": int,
     "net_sentiment": float,
     "summary_bullets": ["...","..."],
     "evidence_snippets": ["...", "..."]
   }},
   {{
     "name": "print_or_pattern",
     "mentions": int,
     "positive": int,
     "negative": int,
     "neutral": int,
     "net_sentiment": float,
     "summary_bullets": ["...","..."],
     "evidence_snippets": ["...", "..."]
   }},
   {{
     "name": "fabric",
     "mentions": int,
     "positive": int,
     "negative": int,
     "neutral": int,
     "net_sentiment": float,
     "summary_bullets": ["...","..."],
     "evidence_snippets": ["...", "..."]
   }}
 ]
}}

### Input Reviews
{reviews_block}

Return ONLY valid JSON — no markdown, no commentary, no prose.
"""

# ----------------------------
# HELPERS
# ----------------------------
FIXED_ATTRS = ["silhouette", "proportion_or_fit", "detail", "color", "print_or_pattern", "fabric"]

def load_reviews_df(path: str, sheet: str | None = None) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        df["__sheet__"] = "csv"
    else:
        # read one sheet or ALL sheets
        if sheet is None:
            all_sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
            frames = []
            for sh_name, sh_df in all_sheets.items():
                if sh_df is None or sh_df.empty:
                    continue
                sh_df["__sheet__"] = str(sh_name)
                frames.append(sh_df)
            if not frames:
                raise ValueError("No non-empty sheets found in workbook.")
            df = pd.concat(frames, ignore_index=True)
        else:
            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
            df["__sheet__"] = str(sheet)

    # normalize columns
    df.columns = [c.lower() for c in df.columns]
    need = {"style_number", "headline", "rating", "comments"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")

    # clean style_number as string
    df["style_number"] = df["style_number"].astype(str).str.strip()
    # basic trims
    for c in ["headline", "comments"]:
        df[c] = df[c].astype(str).str.replace("\n", " ").str.strip()

    return df


def to_review_records(df: pd.DataFrame, max_reviews: int = 500) -> List[Dict[str, Any]]:
    # Simple clean & truncate to keep tokens reasonable
    def norm(s):
        s = str(s or "").strip().replace("\n", " ")
        return " ".join(s.split())
    recs = []
    for _, r in df.head(max_reviews).iterrows():
        recs.append({
            "style_number": str(r.get("style_number", "")),
            "headline": norm(r.get("headline", ""))[:400],
            "rating": int(r.get("rating", 0)) if pd.notna(r.get("rating")) else 0,
            "comments": norm(r.get("comments", ""))[:1500]
        })
    return recs

def make_user_prompt(style_number: str, reviews: List[Dict[str, Any]]) -> str:
    reviews_block = json.dumps(reviews, ensure_ascii=False, indent=2)
    return USER_PROMPT_TEMPLATE.format(style_number=style_number, reviews_block=reviews_block)

def save_excel_from_attributes(attrs: List[Dict[str, Any]], out_xlsx: str, product_id: str, overall_summary: str):
    # Build a tidy DataFrame for Excel
    rows = []
    for a in attrs:
        rows.append({
            "product_id": product_id,
            "attribute": a.get("name", ""),
            "mentions": a.get("mentions", 0),
            "positive": a.get("positive", 0),
            "negative": a.get("negative", 0),
            "neutral": a.get("neutral", 0),
            "net_sentiment": a.get("net_sentiment", 0.0),
            "summary_bullets": " | ".join(a.get("summary_bullets", [])),
            "evidence_snippets": " | ".join(a.get("evidence_snippets", []))
        })
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="attributes")
        meta = pd.DataFrame([{"product_id": product_id, "overall_summary": overall_summary}])
        meta.to_excel(writer, index=False, sheet_name="overall")
    print(f"Saved Excel → {out_xlsx}")

# ----------------------------
# MAIN
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV/XLSX with reviews")
    ap.add_argument("--style-number", required=True, help="style_number to summarize (product id)")
    ap.add_argument("--provider", default="gemini", choices=["gemini", "openai"], help="LLM provider")
    ap.add_argument("--model", default=None, help="Model name (default: gemini-1.5-flash or gpt-4o-mini)")
    ap.add_argument("--max-reviews", type=int, default=500, help="Max reviews to include")
    ap.add_argument("--out-json", default="summary.json", help="Output JSON path")
    ap.add_argument("--out-xlsx", default="summary.xlsx", help="Output Excel path")
    args = ap.parse_args()

    df = load_reviews(args.input)
    product_df = df[df["style_number"].astype(str) == str(args.style_number)]
    if product_df.empty:
        raise ValueError(f"No rows found for style_number={args.style_number}")

    reviews = to_review_records(product_df, max_reviews=args.max_reviews)
    user_prompt = make_user_prompt(args.style_number, reviews)

    model = args.model or ("gemini-1.5-flash" if args.provider == "gemini" else "gpt-4o-mini")

    print(f"Calling {args.provider.upper()} model: {model}  | Reviews: {len(reviews)}")

    # One call that executes your entire pipeline prompt end-to-end
    if args.provider == "gemini":
        raw = call_gemini(SYSTEM_PROMPT, user_prompt, model=model)
    else:
        raw = call_openai(SYSTEM_PROMPT, user_prompt, model=model)

    # Basic JSON validation + retry hint if needed
    try:
        data = json.loads(raw)
    except Exception as e:
        # One soft retry with minimal nudge (Gemini/OpenAI often fix on 2nd try)
        print("First parse failed, retrying with 'Return valid JSON only.' nudge...")
        if args.provider == "gemini":
            raw = call_gemini(SYSTEM_PROMPT, user_prompt + "\n\nReturn valid JSON only.", model=model)
        else:
            raw = call_openai(SYSTEM_PROMPT, user_prompt + "\n\nReturn valid JSON only.", model=model)
        data = json.loads(raw)  # if this fails, let it raise

    # Sanity: ensure the six attributes exist; if any missing, add empty shells
    have = {a.get("name") for a in data.get("attributes", [])}
    for name in FIXED_ATTRS:
        if name not in have:
            data.setdefault("attributes", []).append({
                "name": name,
                "mentions": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "net_sentiment": 0.0,
                "summary_bullets": [],
                "evidence_snippets": []
            })

    # Save JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON → {args.out_json}")

    # Save Excel (tidy attributes + overall sheet)
    save_excel_from_attributes(
        attrs=data.get("attributes", []),
        out_xlsx=args.out_xlsx,
        product_id=str(data.get("product_id", args.style_number)),
        overall_summary=data.get("overall_summary", "")
    )

if __name__ == "__main__":
    main()
