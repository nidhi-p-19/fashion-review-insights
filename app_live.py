import json, os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ================== CONFIG ==================
load_dotenv()  # reads GEMINI_API_KEY from .env if present

FIXED_ATTRS = ["silhouette","proportion_or_fit","detail","color","print_or_pattern","fabric"]
ATTR_LABELS = {
    "silhouette":"Silhouette",
    "proportion_or_fit":"Proportion / Fit",
    "detail":"Detail",
    "color":"Color",
    "print_or_pattern":"Print / Pattern",
    "fabric":"Fabric",
}

SYSTEM_PROMPT = (
    "You are an expert NLP pipeline that extracts structured insights from product reviews.\n"
    "Return ONLY strict JSON. Use ONLY these attributes: silhouette, proportion_or_fit, detail, color, print_or_pattern, fabric."
)

USER_PROMPT_TEMPLATE = """You are given raw customer reviews for a product, each with a rating, headline, and comment.
Perform the following steps and return ONLY valid JSON.

### Step 1. Review Understanding
- Combine each review’s headline and comment into one text.
- Split into sentences.

### Step 2. Attribute Detection
Use ONLY these attributes: ["silhouette","proportion_or_fit","detail","color","print_or_pattern","fabric"].
Map common synonyms (A-line/boxy/flowy→silhouette; true to size/runs small→proportion_or_fit; buttons/embroidery→detail; blue/red/fades→color; polka dot/stripes→print_or_pattern; cotton/silk/itchy/soft→fabric).

### Step 3. Snippet & Sentiment (per (sentence,attribute))
- Extract a short verbatim snippet (<=12 tokens).
- Score: positive=1, negative=-1, unclear=0.

### Step 4. Aggregate per Attribute
- mentions, positive, negative, neutral, net_sentiment = mean(score).

### Step 5. Summaries
- Overall summary (70–100 words).
- Attribute mini-summaries (2–3 bullets, 35–60 words).
Output limits:
- For each attribute, return at most 3 evidence_snippets. Each snippet must be <= 12 tokens.

### Step 6. Output JSON schema
{{
 "product_id": "{style_number}",
 "overall_summary": "string",
 "attributes": [
   {{"name":"silhouette","mentions":int,"positive":int,"negative":int,"neutral":int,"net_sentiment":float,"summary_bullets":["..."],"evidence_snippets":["..."]}},
   {{"name":"proportion_or_fit",...}},
   {{"name":"detail",...}},
   {{"name":"color",...}},
   {{"name":"print_or_pattern",...}},
   {{"name":"fabric",...}}
 ]
}}

### Input Reviews
{reviews_block}

Return ONLY valid JSON — no markdown.
"""

# ================== DATA HELPERS ==================
def load_reviews_df(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    """
    Reads CSV or XLSX. If sheet is None for XLSX, reads ALL sheets and concatenates.
    Keeps a __sheet__ column for debugging.
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        df["__sheet__"] = "csv"
    else:
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

    df.columns = [c.lower() for c in df.columns]
    need = {"style_number","headline","rating","comments"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")

    df["style_number"] = df["style_number"].astype(str).str.strip()
    for c in ["headline","comments"]:
        df[c] = df[c].astype(str).str.replace("\n"," ").str.strip()
    return df

def to_review_records(df: pd.DataFrame, max_reviews: int = 80) -> List[Dict]:
    def norm(s): return " ".join(str(s or "").split())
    recs = []
    for _, r in df.head(max_reviews).iterrows():
        recs.append({
            "style_number": str(r.get("style_number","")),
            "headline": norm(r.get("headline",""))[:400],
            "rating": int(r.get("rating",0)) if pd.notna(r.get("rating")) else 0,
            "comments": norm(r.get("comments",""))[:1200],
        })
    return recs

def make_user_prompt(style_number: str, reviews: List[Dict]) -> str:
    block = json.dumps(reviews, ensure_ascii=False, indent=2)
    return USER_PROMPT_TEMPLATE.format(style_number=style_number, reviews_block=block)

# ================== GEMINI CALLER (ROBUST) ==================
def call_gemini(system_prompt: str, user_prompt: str, model: str = "gemini-2.0-flash") -> str:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)

    # backward-compatible safety (SDKs differ in enum names)
    try:
        safety = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
    except Exception:
        safety = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def _gen(prompt, max_tokens):
        return model_obj.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1, "top_p": 0.9,
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
            },
            safety_settings=safety,
        )

    prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
    resp = _gen(prompt, 2400)
    text = getattr(resp, "text", None)

    if not text and getattr(resp, "candidates", None):
        c0 = resp.candidates[0]
        parts = getattr(getattr(c0, "content", None), "parts", None)
        if parts:
            text = "".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None))

    if not text:
        tighter = user_prompt + "\n\nReturn compact JSON. Limit evidence_snippets to 2 per attribute."
        prompt2 = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{tighter}"
        resp2 = _gen(prompt2, 1600)
        text = getattr(resp2, "text", None)

    if not text:
        raise ValueError("Gemini returned no text (token limit). Lower max reviews or use batching.")
    return text

# ================== BATCH MAP–MERGE ==================
BATCH_SIZE = 60

def summarize_batch(style_number: str, batch_reviews: list, model: str) -> dict:
    raw = call_gemini(SYSTEM_PROMPT, make_user_prompt(style_number, batch_reviews), model=model)
    return json.loads(raw)

def merge_batches(batch_outputs: list) -> dict:
    merged = {"product_id":"", "overall_summary":"", "attributes":[]}
    acc = {n: {"name": n, "mentions":0, "positive":0, "negative":0, "neutral":0,
               "net_sentiment":0.0, "summary_bullets":[], "evidence_snippets":[]} for n in FIXED_ATTRS}
    for out in batch_outputs:
        if not merged["product_id"]:
            merged["product_id"] = out.get("product_id","")
        for a in out.get("attributes", []):
            t = acc.get(a.get("name"))
            if not t:
                continue
            t["mentions"] += a.get("mentions",0)
            t["positive"] += a.get("positive",0)
            t["negative"] += a.get("negative",0)
            t["neutral"]  += a.get("neutral",0)
            for s in (a.get("evidence_snippets") or []):
                if len(t["evidence_snippets"]) < 3:
                    t["evidence_snippets"].append(s)
            for b in (a.get("summary_bullets") or []):
                if len(t["summary_bullets"]) < 3:
                    t["summary_bullets"].append(b)
    for t in acc.values():
        m = max(1, t["mentions"])
        t["net_sentiment"] = round((t["positive"] - t["negative"]) / m, 3)
        merged["attributes"].append(t)
    return merged

def batch_pipeline(style_number: str, reviews: list, model: str) -> dict:
    outs = []
    for i in range(0, len(reviews), BATCH_SIZE):
        outs.append(summarize_batch(style_number, reviews[i:i+BATCH_SIZE], model))
    merged = merge_batches(outs)
    bullets = [o.get("overall_summary","") for o in outs if o.get("overall_summary")]
    if bullets:
        sys = 'Return JSON {"overall_summary":"..."} of 80-100 words using ONLY these bullets.'
        user = "Bullets:\n- " + "\n- ".join(bullets[:8])
        try:
            merged["overall_summary"] = json.loads(call_gemini(sys, user, model)).get("overall_summary","")
        except Exception:
            merged["overall_summary"] = bullets[0][:300]
    return merged

# ================== UI HELPERS ==================
def sentiment_emoji(net: float) -> str:
    return "🟢" if net >= 0.25 else ("🔴" if net <= -0.25 else "🟡")

def recompute_net(a: Dict) -> float:
    m = max(1, int(a.get("mentions",0)))
    return round((int(a.get("positive",0)) - int(a.get("negative",0))) / m, 3)

def attribute_order(attrs: List[Dict]) -> List[Dict]:
    by = {a.get("name"): a for a in attrs}
    ordered = [by.get(n, {"name":n,"mentions":0,"positive":0,"negative":0,"neutral":0,"net_sentiment":0.0,"summary_bullets":[],"evidence_snippets":[]}) for n in FIXED_ATTRS]
    return sorted(ordered, key=lambda x: -(x.get("mentions",0) or 0))

def metric_line(a: Dict) -> str:
    return f"Mentions: {a.get('mentions',0)} | ✅ {a.get('positive',0)}  ❌ {a.get('negative',0)}  ⚪ {a.get('neutral',0)}"

def onepager_md(payload: dict, attr: Optional[dict], avg_rating: Optional[float], n_reviews: Optional[int]) -> str:
    lines = [f"# Customers say — {payload.get('product_id','')}"]
    if avg_rating is not None and n_reviews is not None:
        lines.append(f"**Rating:** {avg_rating:.2f} ⭐  |  **Reviews:** {n_reviews}")
    lines += ["", payload.get("overall_summary",""), "\n---\n"]
    if attr:
        label = ATTR_LABELS.get(attr['name'], attr['name'].title())
        lines.append(f"## {label}")
        lines.append(metric_line(attr))
        lines.append(f"Net sentiment: {recompute_net(attr):.2f}\n")
        for b in (attr.get("summary_bullets") or [])[:3]: lines.append(f"- {b}")
        lines.append("\n**Representative quotes**")
        for q in (attr.get("evidence_snippets") or [])[:6]: lines.append(f"> {q}")
    return "\n".join(lines)

# ================== STREAMLIT PAGE ==================
st.set_page_config(page_title="Customers say (live)", page_icon="🛍️", layout="wide")
st.title("Customers say — Live (LLM)")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.subheader("Input")
    excel_file = st.file_uploader("Excel/CSV with reviews", type=["xlsx","xls","csv"])
    excel_path = None
    if excel_file:
        excel_path = "uploaded_reviews.tmp.xlsx"
        with open(excel_path, "wb") as f: f.write(excel_file.getbuffer())
    else:
        default_guess = [p for p in Path(".").glob("*.xlsx")]
        excel_path = str(default_guess[0]) if default_guess else None

    # sheet picker (default = All)
    sheet_choice = None
    if excel_path and excel_path.lower().endswith((".xlsx",".xls")):
        try:
            xls = pd.ExcelFile(excel_path, engine="openpyxl")
            names = xls.sheet_names
            sheet_pick = st.selectbox("Pick a sheet (or 'All')", ["All"] + names, index=0)
            if sheet_pick != "All":
                sheet_choice = sheet_pick
        except Exception:
            pass

    style_number = st.text_input("style_number", value="")
    max_reviews = st.slider("max reviews", 20, 200, 80, 10)
    model = st.selectbox("Gemini model", ["gemini-2.0-flash","gemini-2.0-flash-lite","gemini-2.5-flash"], index=0)
    st.caption("Tip: use 2.0-flash for stability; lower max reviews if you hit token limits.")
    run_btn = st.button("Generate summary")

# ---------- SESSION STORAGE ----------
# cache the last payload so chip clicks don't require pressing the button again
if "payload" not in st.session_state:
    st.session_state["payload"] = None
if "payload_key" not in st.session_state:
    st.session_state["payload_key"] = None
if "avg_rating" not in st.session_state:
    st.session_state["avg_rating"] = None
if "n_reviews" not in st.session_state:
    st.session_state["n_reviews"] = None

# identify current input combo (used to invalidate cache)
current_key = (
    excel_path or "",
    sheet_choice or "ALL",
    str(style_number or ""),
    int(max_reviews),
    model,
)

# require button only on first ever run
if not st.session_state["payload"] and not run_btn:
    st.info("Upload your Excel/CSV, optionally pick a sheet, enter a style_number, then click **Generate summary**.")
    st.stop()

need_new_run = run_btn or (st.session_state["payload_key"] != current_key)

# ---------- LOAD + FILTER + CALL LLM (when needed) ----------
if need_new_run:
    try:
        df = load_reviews_df(excel_path, sheet=sheet_choice)
    except Exception as e:
        st.exception(e); st.stop()

    sdf = df[df["style_number"].astype(str).str.strip() == str(style_number)]
    if sdf.empty:
        st.error(f"No rows for style_number={style_number}. "
                 f"Hint: it may be on a different sheet — choose 'All' or pick the correct sheet.")
        st.stop()

    records = to_review_records(sdf, max_reviews=max_reviews)
    st.caption(f"Using {len(records)} reviews for product {style_number} "
               f"(from sheets: {', '.join(sorted(sdf['__sheet__'].unique()))})")

    avg_rating = float(sdf["rating"].dropna().astype(float).mean()) if sdf["rating"].notna().any() else None
    n_reviews = int(len(sdf)) if len(sdf) else None

    try:
        if len(records) > 80:
            payload = batch_pipeline(style_number, records, model=model)
        else:
            raw = call_gemini(SYSTEM_PROMPT, make_user_prompt(style_number, records), model=model)
            payload = json.loads(raw)
    except Exception as e:
        st.exception(e); st.stop()

    Path("live_outputs").mkdir(exist_ok=True)
    json_path = Path("live_outputs") / f"summary_{style_number}.json"
    with open(json_path, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
    st.success(f"Saved JSON → {json_path.as_posix()}")

    st.session_state["payload"] = payload
    st.session_state["payload_key"] = current_key
    st.session_state["avg_rating"] = avg_rating
    st.session_state["n_reviews"] = n_reviews
else:
    payload = st.session_state["payload"]
    avg_rating = st.session_state["avg_rating"]
    n_reviews = st.session_state["n_reviews"]

# ---------- PERSIST ATTRIBUTE SELECTION ----------
pid = str(payload.get("product_id", ""))

if "selected_attr" not in st.session_state:
    st.session_state["selected_attr"] = None
if "prev_pid" not in st.session_state:
    st.session_state["prev_pid"] = None
if st.session_state["prev_pid"] != pid:
    st.session_state["selected_attr"] = None
    st.session_state["prev_pid"] = pid

def _set_attr(name: str):
    st.session_state["selected_attr"] = name

# ---------- RENDER ----------
with st.container(border=True):
    st.write((payload.get("overall_summary") or "").strip())

attrs = attribute_order(payload.get("attributes", []))
cols = st.columns(len(FIXED_ATTRS))

for i, a in enumerate(attrs[:len(FIXED_ATTRS)]):
    name = a.get("name")
    label = ATTR_LABELS.get(name, name.title())
    m = a.get("mentions", 0)
    net = recompute_net(a)
    icon = sentiment_emoji(net)
    tip = f"{label}\nMentions: {m}\n+ {a.get('positive',0)} / - {a.get('negative',0)} / 0 {a.get('neutral',0)}"
    cols[i].button(
        f"{icon} {label} · {m}",
        key=f"chip_{pid}_{name}",
        help=tip,
        on_click=_set_attr,
        args=(name,),
    )

st.markdown("---")

selected_attr_name = st.session_state["selected_attr"]
if not selected_attr_name:
    st.caption("Select an attribute chip above to see details.")
    st.stop()

a = next((x for x in attrs if x.get("name") == selected_attr_name), None)
label = ATTR_LABELS.get(selected_attr_name, selected_attr_name.title())
st.subheader(label)

if not a or a.get("mentions",0) == 0:
    st.info("No mentions for this attribute.")
    st.stop()

c1, c2 = st.columns([2,1])
with c1: st.write(metric_line(a))
with c2: st.metric("Net sentiment", f"{recompute_net(a):.2f}")

pos, neg = int(a.get("positive",0)), int(a.get("negative",0))
tot = max(1, pos+neg)
st.write("**Positive vs Negative (mentions only):**")
st.progress(pos/tot)
st.caption(f"Positive: {pos}  |  Negative: {neg}")

bullets = (a.get("summary_bullets") or [])[:3]
quotes  = (a.get("evidence_snippets") or [])[:6]
if bullets:
    st.markdown("### Summary")
    for b in bullets: st.markdown(f"- {b}")
if quotes:
    st.markdown("### Representative quotes")
    for q in quotes: st.write(f"“{q}”")

st.download_button(
    "⬇️ Download one-pager (Markdown)",
    data=onepager_md(payload, a, avg_rating, n_reviews),
    file_name=f"customers_say_{pid}_{a['name']}.md",
    mime="text/markdown"
)
