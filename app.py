import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# =============== SETTINGS ===============
EXCEL_PATH = "Data - 10 Styles Insights.xlsx"   # <- change if needed
FIXED_ATTRS = ["silhouette", "proportion_or_fit", "detail", "color", "print_or_pattern", "fabric"]
ATTR_LABELS = {
    "silhouette": "Silhouette",
    "proportion_or_fit": "Proportion / Fit",
    "detail": "Detail",
    "color": "Color",
    "print_or_pattern": "Print / Pattern",
    "fabric": "Fabric",
}

# =============== UTILS ===============
def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsons_from_folder(folder: Path) -> Dict[str, Dict]:
    out = {}
    for p in sorted(folder.glob("*.json")):
        try:
            j = load_json(p)
            pid = str(j.get("product_id") or p.stem)
            out[pid] = j
        except Exception:
            pass
    return out

def get_rating_stats(product_id: str) -> Optional[Dict]:
    try:
        df = pd.read_excel(EXCEL_PATH)
        df.columns = [c.lower() for c in df.columns]
        sdf = df[df["style_number"].astype(str) == str(product_id)]
        if sdf.empty:
            return None
        return {
            "avg_rating": float(sdf["rating"].dropna().astype(float).mean()),
            "n_reviews": int(len(sdf)),
        }
    except Exception:
        return None

def recompute_net(attr: Dict) -> float:
    m = max(1, int(attr.get("mentions", 0)))
    return round((int(attr.get("positive", 0)) - int(attr.get("negative", 0))) / m, 3)

def attribute_order(attrs: List[Dict]) -> List[Dict]:
    byname = {a.get("name"): a for a in attrs}
    ordered = []
    # ensure all fixed attrs exist (even if zero)
    for n in FIXED_ATTRS:
        a = byname.get(n, {"name": n, "mentions": 0, "positive": 0, "negative": 0, "neutral": 0, "net_sentiment": 0.0, "summary_bullets": [], "evidence_snippets": []})
        ordered.append(a)
    # optional: sort by mentions desc
    ordered = sorted(ordered, key=lambda x: -(x.get("mentions", 0) or 0))
    return ordered

def sentiment_emoji(net: float) -> str:
    return "🟢" if net >= 0.25 else ("🔴" if net <= -0.25 else "🟡")

def metric_line(a: Dict) -> str:
    return f"Mentions: {a.get('mentions',0)} | ✅ {a.get('positive',0)}  ❌ {a.get('negative',0)}  ⚪ {a.get('neutral',0)}"

def onepager_md(payload: dict, attr: Optional[dict], stats: Optional[dict]) -> str:
    lines = []
    lines.append(f"# Customers say — {payload.get('product_id','')}")
    if stats:
        lines.append(f"**Rating:** {stats['avg_rating']:.2f} ⭐  |  **Reviews:** {stats['n_reviews']}")
    lines.append("")
    lines.append(payload.get("overall_summary",""))
    lines.append("\n---\n")
    if attr:
        label = ATTR_LABELS.get(attr["name"], attr["name"].title())
        lines.append(f"## {label}")
        lines.append(f"{metric_line(attr)}")
        lines.append(f"Net sentiment: {recompute_net(attr):.2f}\n")
        for b in (attr.get("summary_bullets") or [])[:3]:
            lines.append(f"- {b}")
        lines.append("\n**Representative quotes**")
        for q in (attr.get("evidence_snippets") or [])[:6]:
            lines.append(f"> {q}")
    return "\n".join(lines)

# =============== PAGE ===============
st.set_page_config(page_title="Customers say", page_icon="🛍️", layout="wide")

# Dark theme polish
st.markdown("""
<style>
    .stMetric > div { background: #111827; padding: 0.75rem 1rem; border-radius: 0.75rem; }
    .chip button { border-radius: 999px !important; }
    .stButton>button { border-radius: 999px; padding: .5rem 1rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Data source")
    mode = st.radio("Choose input mode", ["Single JSON file", "Folder of JSONs"], horizontal=False)

    data_map = {}
    selected_pid = None

    if mode == "Single JSON file":
        file = st.file_uploader("Upload summary.json", type=["json"])
        if file:
            data = json.loads(file.getvalue().decode("utf-8"))
            pid = str(data.get("product_id") or "Product")
            data_map[pid] = data
            selected_pid = pid
    else:
        folder_str = st.text_input("Folder path with *.json", value="")
        if folder_str and Path(folder_str).exists():
            data_map = load_jsons_from_folder(Path(folder_str))
        if data_map:
            selected_pid = st.selectbox("Select product", sorted(data_map.keys()))

    st.markdown("---")
    st.caption("Tip: Drop multiple JSONs in a folder to compare products.")

if not data_map or not selected_pid:
    st.title("Customers say")
    st.info("Upload a JSON summary or set a folder in the sidebar to begin.")
    st.stop()

payload = data_map[selected_pid]
pid = str(payload.get("product_id", selected_pid))

st.title("Customers say")
st.caption(f"Product ID: **{pid}**")

stats = get_rating_stats(pid)
if stats:
    st.caption(f"⭐ **{stats['avg_rating']:.2f}** avg from **{stats['n_reviews']}** reviews")

# ---------- Overall Summary ----------
overall = (payload.get("overall_summary") or "").strip()
with st.container(border=True):
    st.write(overall if overall else "No overall summary found.")

st.write("")

# ---------- Attribute Chips ----------
attrs = attribute_order(payload.get("attributes", []))
cols = st.columns(len(FIXED_ATTRS))
selected_attr_name = st.session_state.get("selected_attr")

for i, a in enumerate(attrs[:len(FIXED_ATTRS)]):
    name = a.get("name")
    label = ATTR_LABELS.get(name, name.title())
    m = a.get("mentions", 0)
    net = recompute_net(a)
    tip = f"{label}\nMentions: {m}\n+ {a.get('positive',0)} / - {a.get('negative',0)} / 0 {a.get('neutral',0)}"
    if cols[i].button(f"{sentiment_emoji(net)} {label} · {m}", key=f"chip_{name}", help=tip):
        st.session_state["selected_attr"] = name
        selected_attr_name = name

st.markdown("---")

# ---------- Attribute Detail ----------
if not selected_attr_name:
    st.caption("Select an attribute chip above to see details.")
    st.stop()

a = next((x for x in attrs if x.get("name") == selected_attr_name), None)
label = ATTR_LABELS.get(selected_attr_name, selected_attr_name.title())
st.subheader(label)

if not a or a.get("mentions", 0) == 0:
    st.info("No mentions for this attribute in the selected reviews.")
    st.stop()

c1, c2 = st.columns([2, 1])
with c1:
    st.write(metric_line(a))
with c2:
    st.metric("Net sentiment", f"{recompute_net(a):.2f}")

# Mini bar for pos/neg
pos, neg = int(a.get("positive", 0)), int(a.get("negative", 0))
tot_pn = max(1, pos + neg)
st.write("**Positive vs Negative (mentions only):**")
st.progress(pos / tot_pn)
st.caption(f"Positive: {pos}  |  Negative: {neg}")

# Bullets + quotes (capped)
bullets = (a.get("summary_bullets") or [])[:3]
quotes = (a.get("evidence_snippets") or [])[:6]

if bullets:
    st.markdown("### Summary")
    for b in bullets:
        st.markdown(f"- {b}")

if quotes:
    st.markdown("### Representative quotes")
    for q in quotes:
        st.write(f"“{q}”")

# Export one-pager
st.download_button(
    "⬇️ Download one-pager (Markdown)",
    data=onepager_md(payload, a, stats),
    file_name=f"customers_say_{pid}_{selected_attr_name}.md",
    mime="text/markdown"
)
