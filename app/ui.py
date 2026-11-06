"""Look and feel: page styles and the small HTML building blocks used by the app."""
import math

import streamlit as st

GREEN_DARK, GREEN, GREEN_LIGHT = "#173f2c", "#1f5a3d", "#2f9e63"
RED, AMBER = "#c0392b", "#c98500"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=DM+Sans:wght@400;500;700&display=swap');
:root { --ink:#1d2b23; --muted:#5f6f65; --card:#fffdf8; --line:#e7e1d3; --green:#1f5a3d; --green-dark:#173f2c; --mint:#2f9e63; }
html, body, .stApp { font-family: 'DM Sans', system-ui, sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 85% -10%, #e4efe2 0%, transparent 60%), #f7f3e9; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3, .serif { font-family: 'Fraunces', Georgia, serif !important; color: var(--ink); letter-spacing: -0.01em; }
[data-testid="stSidebar"] { background: #f3eee2; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .stButton button { width: 100%; justify-content: flex-start; border-radius: 12px; padding: 0.7rem 1rem; font-weight: 500; }
.stButton button[kind="primary"], .stFormSubmitButton button, .stDownloadButton button[kind="primary"] { background: var(--green); color: #fff !important; border: none; border-radius: 12px; font-weight: 600; }
.stFormSubmitButton button p, .stButton button[kind="primary"] p { color: #fff !important; }
.stButton button[kind="primary"]:hover, .stFormSubmitButton button:hover { background: var(--green-dark); }
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] .card-anchor) { background: var(--card); }
div[data-testid="stForm"], .card { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 1.2rem 1.3rem; box-shadow: 0 6px 24px rgba(31,90,61,0.06); }
[data-testid="stFileUploaderDropzone"] { background: #fffdf8; border: 2px dashed #b9cdb9; border-radius: 18px; padding: 2.2rem 1rem; }
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 0.9rem 1.1rem; }
[data-testid="stMetricValue"] { font-family: 'Fraunces', Georgia, serif; color: var(--green-dark); }
.hero-title { font-family: 'Fraunces', Georgia, serif !important; font-size: 2.7rem !important; font-weight: 700; color: var(--ink); line-height: 1.1; margin: 0; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.2rem; }
.kpi .k-label { color: var(--muted); font-size: .9rem; }
.kpi .k-value { font-family: 'Fraunces', Georgia, serif; font-size: 2rem; font-weight: 700; color: var(--green-dark); line-height: 1.2; }
.kpi .k-note { color: var(--muted); font-size: .82rem; }
.hero-sub { color: var(--muted); font-size: 1.08rem !important; margin: 0.4rem 0 0; }
.pill { display:inline-flex; align-items:center; gap:.45rem; padding:.35rem .8rem; border-radius:999px; font-size:.85rem; font-weight:600; }
.pill-ok { background:#e5f2e8; color:var(--green); border:1px solid #c9e2cf; }
.dot { width:9px; height:9px; border-radius:50%; background:var(--mint); display:inline-block; }
.step { color: var(--mint); font-weight: 700; font-size: .85rem; letter-spacing: .08em; text-transform: uppercase; }
.chip { display:inline-block; padding:.25rem .65rem; margin:.15rem .25rem .15rem 0; border-radius:8px; background:#efe9da; color:var(--ink); font-size:.82rem; font-weight:500; }
.result { background: linear-gradient(135deg, #1c4a33 0%, #123526 100%); color: #f4f1e8; border-radius: 20px; padding: 1.3rem 1.5rem; box-shadow: 0 10px 30px rgba(18,53,38,0.25); }
.result h3 { color:#f4f1e8 !important; margin:0 0 .8rem; font-size:1.25rem; }
.result-grid { display:flex; gap:1.4rem; align-items:center; flex-wrap:wrap; }
.result .big { font-family:'Fraunces',Georgia,serif; font-size:2.1rem; font-weight:700; }
.result .label { color:#b9d3c2; font-size:.85rem; }
.risk-pill { display:inline-block; padding:.35rem .9rem; border-radius:10px; font-weight:700; font-size:.95rem; }
.whatif { flex:1; min-width:220px; border-left:1px solid rgba(255,255,255,.15); padding-left:1.2rem; }
.whatif-row { display:flex; justify-content:space-between; gap:.8rem; font-size:.9rem; padding:.3rem 0; border-bottom:1px solid rgba(255,255,255,.08); }
.alert { border-radius:14px; padding:.9rem 1.1rem; font-weight:600; margin:.9rem 0; }
.alert-red { background:#fde8e5; color:#a3261b; border:1px solid #f6c9c2; }
.alert-green { background:#e5f2e8; color:var(--green); border:1px solid #c9e2cf; }
.list-title { font-family:'Fraunces',Georgia,serif; font-size:1.15rem; font-weight:700; color:var(--ink); margin:.2rem 0 .5rem; }
.reason { display:flex; gap:.6rem; align-items:flex-start; padding:.25rem 0; color:var(--ink); }
.reason:before { content:""; width:8px; height:8px; border-radius:50%; background:var(--mint); margin-top:.45rem; flex:none; }
.action { display:inline-block; padding:.5rem .95rem; margin:.25rem .35rem .25rem 0; border-radius:999px; border:1.5px solid var(--green); color:var(--green); font-weight:600; font-size:.88rem; background:#fffdf8; }
.action:first-of-type { background:var(--green); color:#fff; }
.profile-row { display:flex; justify-content:space-between; padding:.55rem 0; border-bottom:1px solid var(--line); font-size:.93rem; }
.profile-row span:first-child { color:var(--muted); }
.profile-row span:last-child { font-weight:600; color:var(--ink); }
.avatar { width:54px; height:54px; border-radius:50%; background:#dfe9dd; color:var(--green); display:flex; align-items:center; justify-content:center; font-family:'Fraunces',Georgia,serif; font-weight:700; font-size:1.2rem; flex:none; }
.level-card { margin-top:1rem; border-radius:16px; padding:1rem 1.2rem; background:#eef4ec; border:1px solid #d3e3d2; }
.brand { display:flex; gap:.7rem; align-items:center; margin-bottom:1rem; }
.brand-name { font-family:'Fraunces',Georgia,serif; font-weight:700; font-size:1.35rem; color:var(--ink); line-height:1.1; }
.brand-sub { color:var(--muted); font-size:.85rem; }
.promo { border-radius:18px; padding:1.1rem; margin-top:1.2rem; background:linear-gradient(160deg,#e3eedf,#f1ede0); border:1px solid #d6e2d2; }
.promo b { font-family:'Fraunces',Georgia,serif; font-size:1.2rem; color:var(--ink); display:block; line-height:1.2; }
.status { border-radius:14px; padding:.8rem 1rem; margin-top:1rem; background:#fffdf8; border:1px solid var(--line); font-size:.85rem; color:var(--muted); }
.status b { color:var(--ink); }
.alert-amber { background:#fdf3dc; color:#7a5200; border:1px solid #f1dca8; }
.quality { display:grid; grid-template-columns: 1fr auto; gap:.3rem 1.4rem; align-items:center; margin-bottom:1rem; }
.quality .q-sub { color:var(--muted); font-size:.92rem; }
.quality .q-score { font-family:'Fraunces',Georgia,serif; font-size:2.3rem; font-weight:700; line-height:1; grid-row: span 2; text-align:right; }
.quality .q-score small { display:block; font-family:'DM Sans',sans-serif; font-size:.78rem; font-weight:500; color:var(--muted); margin-top:.25rem; }
.q-bar { grid-column: 1 / -1; height:9px; border-radius:99px; background:#ece6d7; overflow:hidden; margin:.5rem 0 .2rem; }
.q-bar span { display:block; height:100%; border-radius:99px; }
.q-chips { grid-column: 1 / -1; display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.4rem; }
.qchip { display:inline-flex; align-items:center; gap:.35rem; padding:.3rem .7rem; border-radius:999px; font-size:.83rem; font-weight:600; background:#efe9da; color:var(--ink); }
.qchip.ok { background:#e5f2e8; color:var(--green); }
.qchip.warn { background:#fdf3dc; color:#7a5200; }
.qchip.out { background:#f1ece4; color:#5f5547; }
.q-note { grid-column: 1 / -1; color:var(--muted); font-size:.86rem; margin-top:.35rem; }
.st-key-problems { background:#fdf3dc; border:1px solid #f1d58f; border-left:6px solid #e0a817; border-radius:16px;
  padding:1rem 1.2rem 1.1rem; margin:.4rem 0 1rem; box-shadow:0 6px 20px rgba(201,133,0,.10); }
.st-key-problems .p-title { font-family:'Fraunces',Georgia,serif; font-size:1.3rem; font-weight:700; color:#5c4100; }
.st-key-problems .p-sub { font-family:'DM Sans',sans-serif; font-size:.95rem; font-weight:500; color:#7a5a12; }
.st-key-problems .p-body { color:#6b5418; font-size:.93rem; margin:.25rem 0 .2rem; }
</style>
"""

BANK_SVG = """<svg width="{s}" height="{s}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
<path d="M32 6 L58 20 H6 Z" fill="#1f5a3d"/><rect x="8" y="22" width="48" height="4" rx="1" fill="#1f5a3d"/>
<rect x="12" y="28" width="6" height="22" rx="1" fill="#2f9e63"/><rect x="23" y="28" width="6" height="22" rx="1" fill="#2f9e63"/>
<rect x="35" y="28" width="6" height="22" rx="1" fill="#2f9e63"/><rect x="46" y="28" width="6" height="22" rx="1" fill="#2f9e63"/>
<rect x="6" y="52" width="52" height="6" rx="1.5" fill="#1f5a3d"/></svg>"""


def html(markup):
    """Render HTML. Lines are stripped so Markdown never turns indented HTML into a code block."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def setup():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle, step=None, right=""):
    step_html = f'<div class="step">{step}</div>' if step else ""
    html(f"""<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;margin-bottom:1.2rem;">
    <div>{step_html}<div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div></div>
    <div style="display:flex;align-items:center;gap:1.2rem;">{right}{BANK_SVG.format(s=86)}</div></div>""")


def kpi(label, value, note=""):
    return f'<div class="kpi"><div class="k-label">{label}</div><div class="k-value">{value}</div><div class="k-note">{note}</div></div>'


def model_pill(text="Model ready"):
    return f'<span class="pill pill-ok"><span class="dot"></span>{text}</span>'


def gauge(prob, color):
    """A 270-degree ring that fills with the churn probability."""
    r, c = 62, 2 * math.pi * 62
    arc = c * 0.75
    fill = arc * max(0.0, min(prob, 1.0))
    return f"""<svg width="180" height="170" viewBox="0 0 170 160">
    <g transform="rotate(135 85 85)">
    <circle cx="85" cy="85" r="{r}" fill="none" stroke="rgba(255,255,255,0.14)" stroke-width="14" stroke-linecap="round" stroke-dasharray="{arc:.1f} {c:.1f}"/>
    <circle cx="85" cy="85" r="{r}" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round" stroke-dasharray="{fill:.1f} {c:.1f}"/>
    </g>
    <text x="85" y="88" text-anchor="middle" font-family="Fraunces, Georgia, serif" font-size="34" font-weight="700" fill="#f4f1e8">{prob:.0%}</text>
    <text x="85" y="110" text-anchor="middle" font-size="11" fill="#b9d3c2">churn risk</text></svg>"""


LEVEL_STYLE = {"High": ("#e8574a", "#fde3df", "#a3261b"),
               "Medium": ("#e2a33a", "#fbefd5", "#8a5a00"),
               "Low": ("#4cc38a", "#dff3e7", "#1f5a3d")}


QUALITY_COLOR = {"Excellent": "#2f9e63", "Good": "#2f9e63", "Limited": "#c98500", "Very limited": "#c0392b"}


def quality_card(label, cov, chips, note=""):
    """Data-quality summary: score, bar, and chips like '86 values fixed'."""
    color = QUALITY_COLOR[label]
    chip_html = "".join(f'<span class="qchip {kind}">{text}</span>' for kind, text in chips)
    note_html = f'<div class="q-note">{note}</div>' if note else ""
    return f"""<div class="card quality"><div class="list-title" style="margin:0;">Data quality: {label}</div>
    <div class="q-score" style="color:{color};">{cov:.0%}<small>of the model's information</small></div>
    <div class="q-sub">Your file was checked and cleaned before scoring. Here's what was found.</div>
    <div class="q-bar"><span style="width:{cov * 100:.1f}%;background:{color};"></span></div>
    <div class="q-chips">{chip_html}</div>{note_html}</div>"""


def indian_number(value):
    """12345678 -> '1,23,45,678' (lakh and crore grouping)."""
    n = int(round(abs(value)))
    digits = str(n)
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ("-" if value < 0 else "") + ",".join(groups + [tail])


def money(value, currency="EUR"):
    """An amount in the file's currency: '₹4,52,300', '€97,199', '$60,000'."""
    symbol = {"EUR": "€", "INR": "₹", "USD": "$"}.get(currency, "")
    return symbol + (indian_number(value) if currency == "INR" else f"{value:,.0f}")
