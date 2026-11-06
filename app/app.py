"""Bank Churn Risk Scorer.

Run from the project folder:   streamlit run app/app.py

Page 1: upload a customer file (CSV, Excel, PDF, Word, JSON or text).
Page 2: score one customer, or see every at-risk customer by name.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import cleaner
import loader
import review
import scoring
import ui

APP_DIR = Path(__file__).resolve().parent
FRIENDLY = loader.FRIENDLY
NOT_IN_FILE = "(not in file)"
TOP_PICK, TOP_TABLE = 300, 1000      # limits that keep the page fast with very large files

st.set_page_config(page_title="Bank Churn Risk Scorer", page_icon="🏦", layout="wide",
                   initial_sidebar_state="expanded")
ui.setup()


@st.cache_resource
def get_model():
    return scoring.load()


@st.cache_data(show_spinner=False, max_entries=5)
def csv_bytes(key, _table):
    """Build the download file once per filter choice (the leading _ tells Streamlit not to hash the table)."""
    return _table.to_csv(index=False).encode("utf-8")


model, info = get_model()
THRESHOLD, BASE = info["threshold"], info["overall_churn_rate"]
state = st.session_state
state.setdefault("page", "upload")
state.setdefault("view", "one")


def load_file(name, data):
    try:
        df = loader.read_any(name, data)
    except Exception as err:            # show the reason instead of crashing
        st.error(f"Couldn't read **{name}**: {err}")
        return
    df.columns = [str(c).strip() for c in df.columns]
    state.pop("check_key", None)
    state.update(raw=df, file_name=name, mapping=loader.guess_mapping(df.columns),
                 name_col=loader.find_column(df.columns, loader.NAME_ALIASES),
                 id_col=loader.find_column(df.columns, loader.ID_ALIASES))


def check_data():
    """Clean the file with the current column matching. Runs again only when the file or matching changes."""
    key = (state.file_name, tuple(state.mapping.items()), state.name_col, state.id_col)
    if state.get("check_key") == key:
        return state.cleaned, False
    with st.spinner(f"Checking {len(state.raw):,} rows for missing and wrong values…"):
        c = cleaner.clean(state.raw, state.mapping, state.name_col, state.id_col)
    state.update(cleaned=c, check_key=key, check_token=state.get("check_token", 0) + 1,
                 check_summary=review.summary(c))
    return c, True


@st.dialog("Review your data", width="large")
def review_dialog():
    review.render(state.cleaned, state.raw, state.name_col, state.file_name, ui.html, token=state.check_token)


def problems_card(is_new):
    """The pop-up and card that say how many rows have problems, with a button to see them highlighted."""
    p = state.check_summary
    bad, fixed, gone, cols = p["bad_rows"], p["fixed_rows"], p["removed"], p["bad_cols"]
    if is_new and (bad or gone):
        st.toast(f"**{bad:,} rows have missing or wrong values** in {len(cols)} column(s). "
                 "Click *Show me the problems* to see them highlighted.", icon="⚠️", duration="long")
    if not (bad or gone or fixed):
        ui.html('<div class="alert alert-green">✓ No missing or wrong values found in your data.</div>')
        return
    with st.container(key="problems"):
        extra = []
        if gone:
            extra.append(f"<b>{gone:,}</b> rows will be removed (duplicates or empty)")
        if fixed:
            extra.append(f"<b>{fixed:,}</b> rows had messy values that were fixed automatically")
        if bad:
            title = (f"⚠ {bad:,} of {p['rows']:,} rows have missing or wrong values "
                     f"<span class='p-sub'>in {len(cols)} column{'s' if len(cols) != 1 else ''}: {', '.join(cols[:6])}"
                     f"{' …' if len(cols) > 6 else ''}</span>")
        else:
            title = "✓ No missing or wrong values"
        ui.html(f"""<div class="p-title">{title}</div><div class="p-body">{' · '.join(extra)}.
        You can still continue: wrong or missing values get a typical customer's value, and every change is listed.</div>""")
        if st.button("Show me the problems (highlighted)" if bad or gone else "See what was fixed",
                     type="primary", icon=":material/search:", key="open_review"):
            review_dialog()


def run_scoring():
    with st.spinner(f"Scoring {len(state.raw):,} customers…"):
        c, _ = check_data()
        if len(c.X) == 0:
            st.error("After cleaning, no customer had any usable values, so there is nothing to score. "
                     "Check that the right columns are matched above and that the file has customer rows.")
            return False
        probs = scoring.score(model, c.X)
        scored = c.X.copy()
        scored.insert(0, "Name", c.names)
        scored["Risk"] = probs
        scored["Level"] = scoring.levels(probs, THRESHOLD, BASE)
        scored["Main reason"] = scoring.main_reasons(c.X)
        scored["Data check"] = ("Guessed: " + c.guessed).where(c.guessed != "", "Complete")
    missing = [f for f, col in state.mapping.items() if col is None]
    state.update(scored=scored.sort_values("Risk", ascending=False).reset_index(drop=True),
                 report=c.report, summary=c.summary, real_share=c.real_share,
                 missing=missing, page="score", view="list")
    return True


# ------------------------------------------------------------------ page 1: upload
def upload_page():
    ui.hero("Upload your customer file",
            "Any file with a table of customers works: CSV, Excel, PDF, Word, JSON or text.",
            step="Step 1 of 2", right=ui.model_pill())
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        up = st.file_uploader("Customer file", type=loader.SUPPORTED, label_visibility="collapsed")
        if up is not None and up.name != state.get("last_upload"):
            state.last_upload = up.name
            load_file(up.name, up.getvalue())
        b1, b2 = st.columns(2)
        if b1.button("No file? Use sample data (200 customers)", width="stretch"):
            sample = APP_DIR / "sample_customers.csv"
            load_file(sample.name, sample.read_bytes())
        if b2.button("Try a messy file (see the cleaning)", width="stretch"):
            sample = APP_DIR / "messy_sample.csv"
            load_file(sample.name, sample.read_bytes())
    with right:
        rows = "".join(f'<span class="chip">{FRIENDLY[f]}</span>' for f in loader.FEATURES)
        ui.html(f"""<div class="card"><div class="list-title">What the file needs</div>
        <div style="color:#5f6f65;font-size:.92rem;margin-bottom:.5rem;">These 10 columns (names don't have to match exactly):</div>
        {rows}<div style="margin-top:.8rem;color:#5f6f65;font-size:.92rem;"><b style="color:#1d2b23">Optional:</b>
        a name or customer ID column, so you can see <i>who</i> is at risk.</div>
        <div style="margin-top:.6rem;color:#5f6f65;font-size:.92rem;"><b style="color:#1d2b23">Messy data is fine:</b>
        blanks, typos, text in number columns, duplicates and title rows are cleaned automatically.</div></div>""")

    if "raw" not in state:
        return
    df, mapping = state.raw, state.mapping
    matched = sum(v is not None for v in mapping.values())
    names_msg = f"names from <b>{state.name_col}</b>" if state.name_col else (
        f"IDs from <b>{state.id_col}</b>" if state.id_col else "no name column (rows will be numbered)")
    ui.html(f"""<div class="alert alert-green">Read {len(df):,} rows and {df.shape[1]} columns from
    <b>{state.file_name}</b> · matched {matched} of 10 model columns · {names_msg}</div>""")
    card_slot = st.container()

    with st.expander("Check how your columns were matched", expanded=matched < 10):
        options = [NOT_IN_FILE] + list(df.columns)
        cols = st.columns(2)
        for i, feat in enumerate(loader.FEATURES):
            current = mapping[feat] if mapping[feat] in df.columns else NOT_IN_FILE
            pick = cols[i % 2].selectbox(FRIENDLY[feat], options, index=options.index(current), key=f"map_{feat}_{state.file_name}")
            mapping[feat] = None if pick == NOT_IN_FILE else pick
        name_opts = ["(none)"] + list(df.columns)
        pick = st.selectbox("Name column (optional)", name_opts,
                            index=name_opts.index(state.name_col) if state.name_col else 0)
        state.name_col = None if pick == "(none)" else pick

    missing = [f for f, c in mapping.items() if c is None]
    if len(missing) == len(loader.FEATURES):
        st.dataframe(df.head(8), width="stretch", hide_index=True)
        st.error("None of the 10 columns were found, so this file can't be scored. Check the column "
                 "matching above, or make sure the file is a table of customers.")
        return
    c, is_new = check_data()
    with card_slot:
        problems_card(is_new)
    first = np.arange(min(8, len(df)))
    st.caption("First rows of your file · yellow = missing or wrong · green = fixed automatically · red = will be removed")
    st.dataframe(review.styled_page(c, df, first, state.name_col), width="stretch", hide_index=True,
                 column_config={review.WHAT: st.column_config.TextColumn(review.WHAT, width="medium")})
    if missing:
        cov = scoring.coverage({f: 0.0 for f in missing})
        st.warning(f"**{len(missing)} column(s) not in your file:** {', '.join(FRIENDLY[f] for f in missing)}. "
                   f"You can still score: these will be filled with a typical customer's value. "
                   f"Your file gives the model **{cov:.0%} of the information** it uses "
                   f"(prediction quality: **{scoring.quality(cov)}**). Age and number of products matter most.")
    if st.button("Continue to scoring  →", type="primary") and run_scoring():
        st.rerun()


def quality_panel(compact=False):
    """Data quality after cleaning: missing columns, fixed values, guesses and removed rows."""
    summ, missing = state.summary, state.get("missing", [])
    cov = scoring.coverage(state.real_share)
    label = scoring.quality(cov)
    chips = []
    if summ["fixed"]:
        chips.append(("ok", f"✓ {summ['fixed']:,} messy values fixed"))
    if summ["filled"]:
        chips.append(("warn", f"{summ['filled']:,} values guessed · {summ['customers_guessed']:,} customers"))
    if summ["duplicates"] + summ["same_id"]:
        chips.append(("out", f"{summ['duplicates'] + summ['same_id']:,} duplicate rows removed"))
    if summ["empty"]:
        chips.append(("out", f"{summ['empty']:,} empty rows removed"))
    if summ["other_country"]:
        chips.append(("warn", f"{summ['other_country']:,} from unknown countries"))
    if not chips and not missing:
        chips.append(("ok", "✓ No problems found"))
    note = ""
    if missing:
        note = (f"<b>Not in your file</b> (typical values used for everyone): {', '.join(FRIENDLY[f] for f in missing)}. "
                "Adding these columns would make the scores more accurate.")
    if compact:
        kind = "alert-green" if label in ("Excellent", "Good") else "alert-amber"
        ui.html(f"""<div class="alert {kind}" style="font-weight:500;">Data quality: <b>{label}</b> ({cov:.0%} of the
        model's information) · {" · ".join(t for _, t in chips)}</div>""")
        return
    ui.html(ui.quality_card(label, cov, chips, note))
    if st.button("Review your data, with problems highlighted", icon=":material/search:", key="open_review_2"):
        review_dialog()
    if len(state.report):
        with st.expander(f"See the full cleaning report ({len(state.report)} kinds of problem fixed)"):
            st.dataframe(state.report, width="stretch", hide_index=True, column_config={
                "Column": st.column_config.TextColumn("Column", width="small"),
                "Rows": st.column_config.NumberColumn("Rows", format="%d", width="small"),
                "Examples": st.column_config.TextColumn("Examples (what was in the file)", width="medium"),
                "What the tool did": st.column_config.TextColumn("What the tool did", width="large")})
            st.caption("Guessed values use a typical customer from the 10,000-customer training data "
                       "(the median for numbers, the most common value for categories). The customer list "
                       "has a Data check column that shows which customers have guessed values.")


# ------------------------------------------------------------------ page 2: sidebar
def sidebar():
    with st.sidebar:
        ui.html(f"""<div class="brand">{ui.BANK_SVG.format(s=44)}<div>
        <div class="brand-name">Churn Risk Scorer</div><div class="brand-sub">Bank retention tool</div></div></div>""")
        for key, label in (("one", "Score one customer"), ("list", "Score a customer list")):
            if st.button(label, type="primary" if state.view == key else "secondary", key=f"nav_{key}", width="stretch"):
                state.view = key
                st.rerun()
        ui.html("""<div class="promo"><b>Prevent churn,<br>grow loyalty</b>
        <span style="color:#5f6f65;font-size:.85rem;">Predict. Retain. Grow.</span></div>""")
        s = state.scored
        ui.html(f"""<div class="status"><span class="dot"></span> <b>Ready to predict</b><br>
        {info['model']} · AUC {info['test_roc_auc']}<br>File: <b>{state.file_name}</b><br>
        {len(s):,} customers · <b>{(s.Level == 'High').sum():,}</b> to call</div>""")
        if st.button("Upload a different file", width="stretch"):
            for k in ("raw", "scored", "file_name", "mapping", "last_upload", "report", "summary", "real_share",
                      "cleaned", "check_key", "check_summary"):
                state.pop(k, None)
            state.page = "upload"
            st.rerun()


# ------------------------------------------------------------------ page 2: one customer
def one_customer_page():
    ui.hero("Bank Churn Risk Scorer", "Identify at-risk customers and act before they leave.",
            right=ui.model_pill())
    quality_panel(compact=True)
    s = state.scored
    find = st.text_input("Find a customer by name", placeholder="Type part of a name, or leave empty for the riskiest")
    pool = s[s.Name.str.contains(find, case=False, na=False, regex=False)] if find else s
    pool = pool.head(TOP_PICK)                                    # keep the dropdown fast for huge files
    labels = ["Enter details manually"] + [f"{n} · {r:.0%} risk" for n, r in zip(pool.Name, pool.Risk)]
    note = f" (top {TOP_PICK} matches)" if len(s) > TOP_PICK else ""
    choice = st.selectbox(f"Pick a customer{note}, riskiest first", labels, index=1 if len(pool) else 0)
    idx = labels.index(choice) - 1
    if idx >= 0:
        base = pool.iloc[idx]
        name = base.Name
    else:
        base = pd.Series({"CreditScore": 620, "Geography": "Germany", "Gender": "Female", "Age": 52, "Tenure": 3,
                          "Balance": 110000.0, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 0,
                          "EstimatedSalary": 90000.0})
        name = "New customer"
    k = f"_{base.name if idx >= 0 else 'manual'}"
    if idx >= 0 and str(base.get("Data check", "Complete")).startswith("Guessed"):
        items = base["Data check"].removeprefix("Guessed: ")
        many = "," in items
        ui.html(f"""<div class="alert alert-amber" style="font-weight:500;">⚠ <b>{name}</b>: {items} {"were" if many else "was"}
        missing or unreadable in the file, so {"typical values were" if many else "a typical value was"} used.
        If you know the real {"values" if many else "value"}, change {"them" if many else "it"} below.</div>""")

    def fit(value, lo, hi):              # keep file values inside each input's range
        return min(max(value, lo), hi)

    c1, c2, c3 = st.columns([1, 1.5, 0.85], gap="medium")
    with c1, st.form("details"):
        st.markdown('<div class="list-title">Customer details</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        countries = ["France", "Germany", "Spain"]
        geo = a.selectbox("Country", countries, index=countries.index(base.Geography), key="geo" + k)
        gender = b.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(base.Gender), key="gen" + k)
        age = a.number_input("Age", 18, 100, fit(int(base.Age), 18, 100), key="age" + k)
        tenure = b.number_input("Years with bank", 0, 50, fit(int(base.Tenure), 0, 50), key="ten" + k)
        products = a.selectbox("Products", [1, 2, 3, 4], index=fit(int(base.NumOfProducts), 1, 4) - 1, key="pro" + k)
        credit = b.number_input("Credit score", 300, 900, fit(int(base.CreditScore), 300, 900), key="cre" + k)
        balance = a.number_input("Account balance", 0.0, 1e10, fit(float(base.Balance), 0.0, 1e10), step=1000.0, key="bal" + k)
        salary = b.number_input("Estimated salary", 0.0, 1e10, fit(float(base.EstimatedSalary), 0.0, 1e10), step=1000.0,
                                key="sal" + k)
        active = a.segmented_control("Active member?", ["Yes", "No"],
                                     default="Yes" if base.IsActiveMember else "No", key="act" + k)
        card = b.segmented_control("Has credit card?", ["Yes", "No"],
                                   default="Yes" if base.HasCrCard else "No", key="card" + k)
        st.form_submit_button("Predict churn risk  →", width="stretch")

    row = {"CreditScore": credit, "Geography": geo, "Gender": gender, "Age": age, "Tenure": tenure,
           "Balance": balance, "NumOfProducts": products, "HasCrCard": int(card != "No"),
           "IsActiveMember": int(active == "Yes"), "EstimatedSalary": salary}
    prob, changes = scoring.what_if(model, row)
    lvl = scoring.level(prob, THRESHOLD, BASE)
    ring, pill_bg, pill_fg = ui.LEVEL_STYLE[lvl]

    with c2:
        whatif = "".join(f'<div class="whatif-row"><span>{label}</span><b>{prob:.0%} → {p:.0%}</b></div>'
                         for label, p in changes) or '<div class="label">Nothing the bank can change lowers this further.</div>'
        ui.html(f"""<div class="result"><h3>Churn risk result</h3><div class="result-grid">
        <div>{ui.gauge(prob, ring)}</div>
        <div><span class="risk-pill" style="background:{pill_bg};color:{pill_fg};">{lvl} risk</span>
        <div class="label" style="margin-top:1rem;">vs bank average ({BASE:.0%})</div>
        <div class="big">{prob / BASE:.1f}×</div></div>
        <div class="whatif"><div class="label" style="margin-bottom:.3rem;">What would lower the risk</div>{whatif}</div>
        </div></div>""")
        if lvl == "High":
            ui.html('<div class="alert alert-red">⚠ Call this customer. Their risk is above the profit-maximising threshold.</div>')
        else:
            ui.html('<div class="alert alert-green">✓ No call needed now. Their risk is below the call threshold.</div>')
        why = "".join(f'<div class="reason">{r}</div>' for r in scoring.reasons(row)) or \
              '<div class="reason">No major risk signs found.</div>'
        acts = "".join(f'<span class="action">{a}</span>' for a in scoring.actions(row))
        ui.html(f"""<div class="card"><div class="list-title">Why this customer may leave</div>{why}
        <div class="list-title" style="margin-top:1rem;">Suggested actions</div>{acts}</div>""")

    with c3:
        initials = "".join(w[0] for w in str(name).split()[:2]).upper()
        rows = [("Products", products), ("Credit score", credit), ("Balance", f"{balance:,.0f}"),
                ("Salary", f"{salary:,.0f}"), ("Active member", active or "No"), ("Has credit card", card or "Yes"),
                ("Years as customer", tenure)]
        body = "".join(f'<div class="profile-row"><span>{a}</span><span>{b}</span></div>' for a, b in rows)
        ui.html(f"""<div class="card"><div class="list-title">Customer profile</div>
        <div style="display:flex;gap:.8rem;align-items:center;margin:.6rem 0 .8rem;">
        <div class="avatar">{initials}</div><div><b>{name}</b><br>
        <span style="color:#5f6f65;font-size:.9rem;">{geo} · {gender}, {age} years</span></div></div>{body}
        <div class="level-card"><span style="color:#5f6f65;">Risk level</span><br>
        <span class="serif" style="font-size:1.6rem;font-weight:700;color:{pill_fg};">{lvl}</span></div></div>""")


# ------------------------------------------------------------------ page 2: customer list
def list_page():
    s = state.scored
    ui.hero("At-risk customers", f"{state.file_name} · {len(s):,} customers scored", right=ui.model_pill())
    quality_panel()
    high = (s.Level == "High").sum()
    cards = [ui.kpi("Customers scored", f"{len(s):,}", "from your file"),
             ui.kpi("At risk: call now", f"{high:,}", f"{high / len(s):.0%} of customers"),
             ui.kpi("Expected to leave", f"{s.Risk.sum():,.0f}", "sum of churn probabilities"),
             ui.kpi("Average risk", f"{s.Risk.mean():.0%}", f"bank average {BASE:.0%}")]
    ui.html('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-bottom:1rem;">'
            + "".join(cards) + "</div>")

    f1, f2, f3 = st.columns([1.2, 1.2, 1.6])
    levels = f1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High"])
    countries = f2.multiselect("Country", sorted(s.Geography.unique()), default=sorted(s.Geography.unique()))
    search = f3.text_input("Search by name", placeholder="Type a name…")
    only_guessed = st.toggle("Only customers with guessed values", value=False)
    view = s[s.Level.isin(levels) & s.Geography.isin(countries)]
    if only_guessed:
        view = view[view["Data check"] != "Complete"]
    if search:
        view = view[view.Name.str.contains(search, case=False, na=False)]

    shown = view.head(TOP_TABLE)
    if len(view) > TOP_TABLE:
        st.caption(f"Showing the {TOP_TABLE:,} riskiest of {len(view):,} matching customers. The download has all of them.")
    table = shown[["Name", "Geography", "Age", "NumOfProducts", "IsActiveMember", "Balance", "Risk", "Level",
                   "Main reason", "Data check"]].copy()
    table["IsActiveMember"] = table["IsActiveMember"].map({1: "Yes", 0: "No"})
    st.dataframe(table, width="stretch", hide_index=True, height=420, column_config={
        "Geography": "Country", "NumOfProducts": "Products", "IsActiveMember": "Active",
        "Balance": st.column_config.NumberColumn("Balance", format="%.0f"),
        "Risk": st.column_config.ProgressColumn("Churn risk", min_value=0.0, max_value=1.0, format="percent")})
    key = (state.file_name, len(s), tuple(levels), tuple(countries), search, only_guessed)
    st.download_button(f"Download this list ({len(view):,} customers)", csv_bytes(key, view),
                       "at_risk_customers.csv", "text/csv", type="primary")

    colors = alt.Scale(domain=["High", "Medium", "Low"], range=["#e8574a", "#e2a33a", "#4cc38a"])
    counts = s.Level.value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
    counts.columns = ["Level", "Customers"]
    by_country = s[s.Level == "High"].Geography.value_counts().reset_index()
    by_country.columns = ["Country", "At-risk customers"]
    g1, g2 = st.columns(2)
    g1.altair_chart(alt.Chart(counts, title="Customers by risk level").mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Customers:Q"), y=alt.Y("Level:N", sort=["High", "Medium", "Low"], title=None),
        color=alt.Color("Level:N", scale=colors, legend=None), tooltip=["Level", "Customers"]).properties(height=220),
        width="stretch")
    g2.altair_chart(alt.Chart(by_country, title="At-risk customers by country").mark_bar(
        cornerRadiusEnd=4, color="#1f5a3d").encode(
        x=alt.X("At-risk customers:Q"), y=alt.Y("Country:N", sort="-x", title=None),
        tooltip=["Country", "At-risk customers"]).properties(height=220), width="stretch")


# ------------------------------------------------------------------ router
if state.page == "upload" or "scored" not in state or "summary" not in state:
    upload_page()
else:
    sidebar()
    if state.view == "one":
        one_customer_page()
    else:
        list_page()
