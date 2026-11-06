"""Review screen: the uploaded table with every problem cell highlighted.

    yellow    missing or wrong, so the model gets a typical value instead
    green     messy but read or corrected automatically ('50k' -> 50,000)
    red row   removed (exact duplicate, repeated customer ID, or nothing usable)

Only one page of rows is styled at a time, so it stays fast even for a million-row file.
"""
import math

import numpy as np
import pandas as pd
import streamlit as st

import cleaner as cl

PAGE_SIZE = 500
CSS = {"bad": "background-color:#ffe795;color:#4d3a00;font-weight:600;",
       "unknown": "background-color:#dde9f7;color:#1e3f66;",
       "fixed": "background-color:#d6ecdc;color:#174a31;",
       "removed": "background-color:#f6d3cc;color:#7a2318;",
       "": ""}
DROP_TEXT = {cl.DUPLICATE: "Exact copy of an earlier row → removed",
             cl.SAME_ID: "Same customer ID as an earlier row → removed",
             cl.EMPTY: "Nothing the model uses could be read → removed"}
SHOW = ["All rows", "Missing or wrong", "Not known to the model", "Fixed automatically", "Removed"]
ROW, WHAT = "Row #", "What's wrong"


# ------------------------------------------------------------------ what's wrong where
def masks(c):
    """Per row of the original file: removed / has a missing or wrong value / has a value the model
    doesn't know (e.g. an Indian state) / had messy values that were fixed (and nothing wrong)."""
    n = len(c.drop)
    removed = c.drop != cl.KEPT
    bad, unknown, fixed = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    for f in c.mapping:
        bad |= np.isin(c.status[f], cl.WRONG)
        unknown |= c.status[f] == cl.OTHER
        fixed |= np.isin(c.status[f], cl.CHANGED)
    if c.name_blank is not None:
        fixed |= c.name_blank                    # a blank name is handled by using the ID instead
    bad &= ~removed
    unknown &= ~removed                  # can overlap: a row can have a wrong value AND an Indian state
    return removed, bad, unknown, fixed & ~bad & ~removed


def column_counts(c, removed):
    """Per file column: how many kept rows are missing/wrong, fixed, and not known to the model."""
    out = {}
    for f, col in c.mapping.items():
        out[col] = (int((np.isin(c.status[f], cl.WRONG) & ~removed).sum()),
                    int((np.isin(c.status[f], cl.CHANGED) & ~removed).sum()),
                    int(((c.status[f] == cl.OTHER) & ~removed).sum()))
    return out


def summary(c):
    """Headline counts for the pop-up: rows and columns with problems."""
    removed, bad, unknown, fixed = masks(c)
    counts = column_counts(c, removed)
    return {"bad_rows": int(bad.sum()), "fixed_rows": int(fixed.sum()), "removed": int(removed.sum()),
            "unknown_rows": int(unknown.sum()),
            "bad_cols": [col for col, (b, _, _) in counts.items() if b],
            "unknown_cols": [col for col, (_, _, u) in counts.items() if u],
            "rows": len(c.drop)}


def messages(c, raw, rows, name_col):
    """One plain-English line per row listing everything that was wrong with it.

    Built a column at a time (not row by row), so it is quick even for hundreds of thousands of rows.
    """
    rows = np.asarray(rows)
    out = pd.Series("", index=range(len(rows)), dtype=object)

    def add(where, text):
        nonlocal out
        cur = out.iloc[where]
        out.iloc[where] = (cur + np.where(cur == "", "", " · ") + text).to_numpy()

    for f, col in c.mapping.items():
        st_ = c.status[f][rows]
        where = np.flatnonzero(st_ != cl.OK)
        if not len(where):
            continue
        code = st_[where]
        orig = raw[col].iloc[rows[where]].astype("string").str.strip()
        shown = ("'" + orig + "'").where(orig.notna() & (orig != ""), "blank").to_numpy(dtype=object)
        new = c.values[f][rows[where]]
        new_txt = np.array(["" if pd.isna(v) else cl._fmt(f, v) for v in new], dtype=object)   # pd.isna: safe for NA, NaN, None
        typical = cl._fmt(f, c.typical[f])
        text = np.select(
            [code == cl.FIXED, code == cl.ADJUSTED, code == cl.BLANK, code == cl.UNREADABLE, code == cl.IMPOSSIBLE],
            [col + ": " + shown + " read as " + new_txt,
             col + ": " + shown + " corrected to " + new_txt,
             np.full(len(code), f"{col}: missing → typical value {typical} used", dtype=object),
             col + ": " + shown + f" can't be read → {typical} used",
             col + ": " + shown + f" is impossible → {typical} used"],
            default=col + ": " + shown + (f" is a place the model doesn't know → scored as {typical}" if f == "Geography"
                                          else f" isn't a category the model learned → {typical} used"))
        add(where, text)
    if c.name_blank is not None:
        where = np.flatnonzero(c.name_blank[rows])
        if len(where):
            add(where, f"{name_col}: blank → ID used instead")
    gone = c.drop[rows]
    for reason, text in DROP_TEXT.items():
        where = np.flatnonzero(gone == reason)
        if len(where):
            out.iloc[where] = text
    where = np.flatnonzero(np.isin(gone, [cl.DUPLICATE, cl.SAME_ID]) & (c.copy_of[rows] >= 0))
    if len(where):                                   # say which row it repeats
        first = (c.copy_of[rows][where] + 1).astype(str)
        kind = np.where(gone[where] == cl.DUPLICATE, "Exact copy of row ", "Same customer ID as row ")
        out.iloc[where] = (pd.Series(kind) + first + " → removed").to_numpy()
    return out.tolist()


def _cell(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def styled_page(c, raw, rows, name_col):
    """The rows to show, as text, with a matching table of cell colours."""
    view = raw.iloc[rows]
    table = pd.DataFrame({col: [_cell(v) for v in view[col]] for col in raw.columns})
    table.insert(0, WHAT, messages(c, raw, rows, name_col))
    table.insert(0, ROW, rows + 1)

    kind = pd.DataFrame("", index=table.index, columns=table.columns)
    for f, col in c.mapping.items():
        s = c.status[f][rows]
        kind.loc[np.isin(s, cl.WRONG), col] = "bad"
        kind.loc[s == cl.OTHER, col] = "unknown"
        kind.loc[np.isin(s, cl.CHANGED), col] = "fixed"
    if c.name_blank is not None:
        kind.loc[c.name_blank[rows], name_col] = "fixed"
    row_kind = np.select([(kind == "bad").any(axis=1), (kind == "unknown").any(axis=1), (kind == "fixed").any(axis=1)],
                         ["bad", "unknown", "fixed"], default="")
    kind[ROW] = row_kind
    kind[WHAT] = row_kind
    gone = c.drop[rows] != cl.KEPT
    kind.loc[gone, :] = "removed"
    css = kind.replace(CSS)
    return table.style.apply(lambda _: css, axis=None)


def problem_rows_csv(c, raw, name_col):
    """Every row with a problem, plus a Problems column, for fixing the data at its source."""
    removed, bad, unknown, fixed = masks(c)
    rows = np.flatnonzero(removed | bad | unknown | fixed)
    out = raw.iloc[rows].copy()
    out.insert(0, WHAT, messages(c, raw, rows, name_col))
    out.insert(0, ROW, rows + 1)
    return out.to_csv(index=False).encode("utf-8")


# ------------------------------------------------------------------ the screen
def chip(css, text):
    return f'<span class="qchip" style="{CSS[css]}">{text}</span>'


def render(c, raw, name_col, file_name, html, token=""):
    removed, bad, unknown, fixed = masks(c)
    counts = column_counts(c, removed)
    n = len(raw)
    unknown_chip = chip("unknown", f"■ Not known to the model · {unknown.sum():,} rows") if unknown.any() else ""
    html(f"""<div style="display:flex;flex-wrap:wrap;gap:.45rem;align-items:center;margin-bottom:.4rem;">
    {chip("bad", f"■ Missing or wrong · {bad.sum():,} rows")}{unknown_chip}
    {chip("fixed", f"■ Fixed automatically · {fixed.sum():,} rows")}
    {chip("removed", f"■ Removed · {removed.sum():,} rows")}
    <span style="color:#5f6f65;font-size:.85rem;margin-left:.3rem;">Yellow cells get a typical value: check them first.
    Blue cells are real data the model never learned (like an Indian state). Hover a column name for its counts.</span></div>""")

    a, b = st.columns([1.9, 1])
    groups = {"All rows": np.ones(n, bool), "Missing or wrong": bad, "Not known to the model": unknown,
              "Fixed automatically": fixed, "Removed": removed}
    shown_groups = [g for g in SHOW if g in ("All rows", "Missing or wrong") or groups[g].any()]
    labels = [f"{g} ({groups[g].sum():,})" for g in shown_groups]
    pick = a.segmented_control("Show", labels, default=labels[1] if bad.any() else labels[0], key=f"rv_show_{token}")
    show = shown_groups[labels.index(pick)] if pick else "All rows"
    problem_cols = [col for col, cnt in counts.items() if any(cnt)]

    def describe(x):
        if x == "Any column":
            return x
        bn, fn, un = counts[x]
        parts = [f"{bn:,} wrong", f"{fn:,} fixed"] + ([f"{un:,} unknown"] if un else [])
        return f"{x}  ({' · '.join(parts)})"
    col_pick = b.selectbox("Column", ["Any column"] + problem_cols, key=f"rv_col_{token}", format_func=describe)

    mask = groups[show]
    if col_pick != "Any column":
        f = next(f for f, col in c.mapping.items() if col == col_pick)
        mask = mask & (c.status[f] != cl.OK)
    rows = np.flatnonzero(mask)
    if not len(rows):
        st.success("No rows match this filter.")
        return

    pages = math.ceil(len(rows) / PAGE_SIZE)
    page = 1
    if pages > 1:
        p1, p2 = st.columns([1, 3])
        page = p1.number_input("Page", 1, pages, 1, key=f"rv_page_{token}_{show}_{col_pick}")
        p2.markdown(f"<div style='padding-top:2.1rem;color:#5f6f65;font-size:.9rem;'>Page {page} of {pages:,} · "
                    f"{len(rows):,} rows · {PAGE_SIZE} per page</div>", unsafe_allow_html=True)
    rows = rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    config = {ROW: st.column_config.NumberColumn(ROW, width="small", format="%d",
                                                 help="Position in your file (first data row = 1)"),
              WHAT: st.column_config.TextColumn(WHAT, width="medium", help="Hover or widen to read the full text")}
    for col, (bn, fn, un) in counts.items():
        if bn or fn or un:
            config[col] = st.column_config.TextColumn(f"⚠ {col}" if bn else (f"◆ {col}" if un else f"✓ {col}"),
                                                      help=f"{bn:,} missing or wrong · {fn:,} fixed automatically"
                                                           + (f" · {un:,} not known to the model" if un else ""))
    st.dataframe(styled_page(c, raw, rows, name_col), hide_index=True, width="stretch", height=480,
                 column_config=config)

    stem = file_name.rsplit(".", 1)[0]
    st.download_button("Download every problem row (CSV)", lambda: problem_rows_csv(c, raw, name_col),
                       f"{stem}_problems.csv", "text/csv", icon=":material/download:")
