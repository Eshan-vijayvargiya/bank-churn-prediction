"""Clean a messy customer table before scoring.

Each model input goes through the same steps:
  1. read what can be read ("₹1,20,000", "50k", "1.2 lakh", "45 yrs", "forty", "Germny", "F.", "Yes")
  2. catch impossible values (age 250, credit score 5000) and fix the obvious ones (a birth year typed as age)
  3. fill anything still blank or unreadable with a typical customer's value
Every change is counted, so the app can show a cleaning report and flag the customers whose scores rest
on guessed values. Work is done once per distinct value, so it stays fast on files with millions of rows.
"""
import difflib
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

from loader import FEATURES, FRIENDLY, INDIAN_PLACES, TO_EURO, TYPICAL, _key, _tokens

# what each cell ended up as
OK, FIXED, BLANK, UNREADABLE, IMPOSSIBLE, ADJUSTED, OTHER = range(7)
GUESSED = [BLANK, UNREADABLE, IMPOSSIBLE, OTHER]      # the model got a typical value, not the real one
WRONG = [BLANK, UNREADABLE, IMPOSSIBLE]               # ...because the data was missing or wrong
# (OTHER is real data the model was never trained on, like an Indian state: shown separately, not as an error)
CHANGED = [FIXED, ADJUSTED]                           # messy, but read or corrected automatically
# why a whole row is removed
KEPT, DUPLICATE, SAME_ID, EMPTY = range(4)

NULLS = {"", "na", "n/a", "n.a.", "nan", "none", "null", "nil", "-", "--", "---", "?", "??", "???", "unknown",
         "missing", "not available", "not known", "#n/a", "<na>", "nat", "blank", "tbd"}
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
MONEY = {"Balance", "EstimatedSalary"}
LIMITS = {"CreditScore": (300, 900), "Age": (18, 100), "Tenure": (0, 50), "NumOfProducts": (1, None),
          "EstimatedSalary": (0, 1e13), "Balance": (0, 1e13)}    # money above 10 trillion is a typing mistake
MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "l": 1e5, "lac": 1e5, "lacs": 1e5, "lakh": 1e5, "lakhs": 1e5, "lpa": 1e5,
               "cr": 1e7, "crore": 1e7, "crores": 1e7, "m": 1e6, "mn": 1e6, "million": 1e6}
WORD_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
WORD_NUMBERS.update({w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)})

COUNTRY_WORDS = {"france": "France", "fr": "France", "fra": "France", "french": "France",
                 "germany": "Germany", "de": "Germany", "deu": "Germany", "ger": "Germany", "german": "Germany",
                 "deutschland": "Germany", "spain": "Spain", "es": "Spain", "esp": "Spain", "spanish": "Spain",
                 "espana": "Spain"}
GENDER_WORDS = {"m": "Male", "male": "Male", "man": "Male", "men": "Male", "boy": "Male", "mr": "Male",
                "gent": "Male", "gents": "Male", "gentleman": "Male", "f": "Female", "female": "Female",
                "woman": "Female", "women": "Female", "girl": "Female", "lady": "Female", "mrs": "Female",
                "ms": "Female", "miss": "Female"}
YES_NO_WORDS = {w: 1 for w in ["1", "yes", "y", "true", "t", "active", "has", "have", "yeah", "yep", "yup", "on"]}
YES_NO_WORDS.update({w: 0 for w in ["0", "no", "n", "false", "f", "inactive", "not active", "nope", "nah", "off",
                                     "no card", "dormant"]})
GENDER_OTHER = {"other", "others", "o", "transgender", "trans", "third gender", "third", "non binary", "nonbinary",
                "nb", "prefer not to say", "not specified", "unspecified", "x"}
OUTCOME_YES = {"1", "yes", "y", "true", "t", "exited", "exit", "churn", "churned", "attrited", "attrited customer",
               "left", "closed", "lost", "leaver", "gone", "lapsed"}
OUTCOME_NO = {"0", "no", "n", "false", "f", "stayed", "stay", "retained", "existing", "existing customer", "active",
              "current", "open", "not churned", "remained", "loyal", "continuing"}
# dates: year first (1990-05-12) is read as year-month-day; anything else day first, as in India (12/05/1990)
ISO_DATE = r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
ANY_DATE = (r"^\s*(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}[-/ ][A-Za-z]{3,9}[-/ ,]+\d{2,4}"
            r"|[A-Za-z]{3,9}[-/ ]\d{1,2},?[-/ ]+\d{2,4})")
UNIT_WORDS = {1e5: {"lpa", "lakh", "lakhs", "lac", "lacs"}, 1e7: {"crore", "crores"},
              1e3: {"thousand", "thousands", "000s", "k"}, 1e6: {"million", "millions", "mn"}}
MONTH_WORDS = {"month", "months", "mon", "mths", "mos", "mo"}
YEAR_WORDS = {"year", "years", "yrs", "yr"}
CATEGORIES = {"Geography": COUNTRY_WORDS, "Gender": GENDER_WORDS,
              "HasCrCard": YES_NO_WORDS, "IsActiveMember": YES_NO_WORDS}


@dataclass
class Cleaned:
    X: pd.DataFrame              # model-ready table
    names: pd.Series             # who each row is
    guessed: pd.Series           # per customer: inputs that were guessed, e.g. "Age, Gender" ("" if none)
    report: pd.DataFrame         # one line per kind of problem found
    summary: dict                # headline counts for the app
    real_share: dict             # per input: share of customers with a real (not guessed) value
    # For the review screen, one entry per row of the ORIGINAL file (removed rows included):
    status: dict = None          # input -> what happened to each cell (OK, FIXED, BLANK, ...)
    values: dict = None          # input -> the value the cleaner read or corrected
    drop: np.ndarray = None      # why each row was removed (KEPT, DUPLICATE, SAME_ID, EMPTY)
    copy_of: np.ndarray = None   # for a removed duplicate: the position of the row it repeats (-1 otherwise)
    name_blank: np.ndarray = None
    mapping: dict = None         # input -> column in the file
    typical: dict = None         # the value used to fill each input when it was missing or wrong
    outcome: np.ndarray = None   # the file's real outcome per scored customer (1 left, 0 stayed, nan unknown)
    outcome_col: str = None
    currency: str = "EUR"        # money in the file; the model works in euros
    places: dict = None          # where customers are from, when the model doesn't know the place
    place: pd.Series = None      # per scored customer: the place as written in the file (for display)


# ------------------------------------------------------------------ small helpers
def _plain(text):
    """Lower case, no accents, letters and spaces only: ' Españа. ' -> 'espana'."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text)).strip()


def _distance(a, b):
    """Edits needed to turn a into b, counting a swap of two neighbouring letters as one (typos like 'frnace')."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _closest(word, words):
    """Fix a typo when the close known words all mean the same thing (1 edit for short words, 2 for long)."""
    if len(word) < 4:
        return None
    allowed = 1 if len(word) <= 5 else 2
    near = difflib.get_close_matches(word, [w for w in words if len(w) >= 3], n=5, cutoff=0.5)
    meanings = {words[w] for w in near if _distance(word, w) <= allowed}
    return meanings.pop() if len(meanings) == 1 else None


def _parse_number(text, money):
    """Read one messy number. Returns a float, or nan if it can't be read."""
    t = str(text).strip().lower()
    words = re.split(r"[\s\-]+", t)
    if words and all(w in WORD_NUMBERS or w == "and" for w in words) and any(w in WORD_NUMBERS for w in words):
        return float(sum(WORD_NUMBERS.get(w, 0) for w in words))          # "thirty two" -> 32
    negative = t.startswith("(") and t.endswith(")")                     # accounting style (500)
    t = re.sub(r"^(less than|more than|up to|upto|under|below|above|over|approx\.?|approximately|around|about|"
               r"min\.?|max\.?|[<>~≈]+)\s*", "", t)                        # "Less than $40K", "> 5 lakh"
    t = re.sub(r"[₹$€£,\s()+]", "", t)
    t = re.sub(r"^(rs\.?|inr|usd|eur|gbp)", "", t)
    rng = re.match(r"^(\d+(?:\.\d+)?)([a-z]*)(?:-|to|–)(?:rs\.?|inr)?(\d+(?:\.\d+)?)([a-z]*)$", t)
    if rng:                                                              # a range: "60k-80k", "6-8lpa" -> midpoint
        lo, lo_unit, hi, hi_unit = float(rng.group(1)), rng.group(2), float(rng.group(3)), rng.group(4)
        unit = hi_unit or lo_unit
        mult = MULTIPLIERS.get(unit, 1) if money else 1
        value = (lo * MULTIPLIERS.get(lo_unit or unit, 1) if money else lo) / 2 + hi * mult / 2
        return -value if negative else value
    if re.fullmatch(r"[\do]+(\.[\do]+)?", t) and re.search(r"\d", t):
        t = t.replace("o", "0")                                          # letter O typed for zero: "7OO"
    m = re.match(r"^(-?\d+(?:\.\d+)?)([a-z]*)", t)
    if not m:
        return np.nan
    value = float(m.group(1))
    if money and m.group(2) in MULTIPLIERS:
        value *= MULTIPLIERS[m.group(2)]                                 # "50k", "1.2lakh", "2cr"
    return -value if negative else value


def _is_blank(text_lower):
    return text_lower.isna() | text_lower.isin(NULLS)


# ------------------------------------------------------------------ one column at a time
def _years_since(dates, today):
    return (today - dates).dt.days.to_numpy(dtype=float, na_value=np.nan) / 365.25


def _parse_dates(text):
    """Turn date strings into timestamps: year-first as Y-M-D, everything else day-first (Indian style)."""
    text = pd.Series(text, dtype=object)
    iso = text.str.match(ISO_DATE, na=False)
    out = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(text[iso], errors="coerce", format="mixed", utc=True).dt.tz_localize(None)
    if (~iso).any():
        out[~iso] = pd.to_datetime(text[~iso], errors="coerce", format="mixed", dayfirst=True,
                                   utc=True).dt.tz_localize(None)
    return out


def _whole_column_or_cells(mask, present, status, notes, problem, action):
    """A fix that applies to most of a column is one report line; a few odd cells are marked one by one."""
    if not mask.any():
        return
    if mask.sum() > 0.5 * max(present.sum(), 1):
        notes.append((problem + " (whole column)", int(mask.sum()), "", action))
    else:
        status[mask] = ADJUSTED


def _numbers(raw, feat, col=""):
    """Read one numeric column. Returns (values, status, notes): notes are column-wide fixes for the report."""
    n = len(raw)
    status = np.full(n, OK, dtype=np.int8)
    notes = []
    today = pd.Timestamp.now().normalize()
    from_date = np.zeros(n, dtype=bool)                  # the cell held a date (of birth, or of joining)
    if pd.api.types.is_datetime64_any_dtype(raw):         # a real date column, e.g. from Excel
        dates = pd.to_datetime(raw, errors="coerce", utc=True).dt.tz_localize(None)
        values = _years_since(dates, today) if feat in ("Age", "Tenure") else np.full(n, np.nan)
        from_date = ~np.isnan(values)
        status[raw.isna().to_numpy()] = BLANK
        status[~from_date & ~raw.isna().to_numpy()] = UNREADABLE
    elif pd.api.types.is_numeric_dtype(raw):
        values = raw.to_numpy(dtype=float, na_value=np.nan, copy=True)   # copy: pandas may hand back a read-only view
        status[np.isnan(values)] = BLANK
    else:
        text = raw.astype("string").str.strip()
        blank = _is_blank(text.str.lower()).to_numpy()
        values = pd.to_numeric(text, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)  # fast path
        values[blank] = np.nan
        status[blank] = BLANK
        if feat in ("Age", "Tenure"):                      # "12/05/1990", "2016-04-01", "05-Dec-1988"
            maybe = np.isnan(values) & ~blank & text.str.match(ANY_DATE, na=False).to_numpy(dtype=bool)
            if maybe.any():
                uniq = pd.unique(text[maybe])
                years = dict(zip(uniq, _years_since(_parse_dates(uniq), today)))
                got = text[maybe].map(years).to_numpy(dtype=float, na_value=np.nan)
                values[maybe] = got
                from_date[maybe] = ~np.isnan(got)
        todo = np.isnan(values) & ~blank & ~from_date
        if todo.any():
            lookup = {u: _parse_number(u, feat in MONEY) for u in pd.unique(text[todo])}
            values[todo] = text[todo].map(lookup).to_numpy(dtype=float, na_value=np.nan)
            status[todo] = np.where(np.isnan(values[todo]), UNREADABLE, FIXED)

    bad_number = np.isinf(values)                        # "inf", 1e400
    values[bad_number] = np.nan
    status[bad_number] = UNREADABLE
    words = _tokens(col) if col else set()
    present = ~np.isnan(values)
    plain = present & (status == OK) & ~from_date
    year = today.year

    if feat in MONEY and plain.any():                     # "Annual_Income_LPA" = 6.5 means 6.5 lakh
        for mult, unit_words in UNIT_WORDS.items():
            hit = words & unit_words
            if hit and np.nanmedian(values[plain]) < 10_000:
                values[plain] *= mult
                notes.append((f"Amounts written in {sorted(hit)[0]} (from the column name)", int(plain.sum()), "",
                              f"Multiplied by {mult:,.0f}"))
                break
    if feat == "Age":                                     # a birth year or date of birth instead of an age
        born = present & ~from_date & (values >= 1900) & (values <= year - 18)
        values[born] = year - values[born]
        values[from_date] = np.floor(values[from_date])
        _whole_column_or_cells(born | from_date, present, status, notes,
                               "Birth dates or years instead of ages", "Turned into ages")
    if feat == "Tenure":
        joined = present & ~from_date & (values >= 1950) & (values <= year)
        values[joined] = year - values[joined]
        converted = joined | from_date
        _whole_column_or_cells(converted, present, status, notes,
                               "Joining dates or years instead of years as a customer", "Turned into years")
        rest = present & ~converted
        named_months = bool(words & MONTH_WORDS) and not words & YEAR_WORDS
        looks_months = (rest.sum() >= 20 and (values[rest] > 24).mean() > 0.25 and np.nanmax(values[rest]) <= 900)
        if rest.any() and (named_months or looks_months):
            values[rest] = values[rest] / 12
            why = "column name says months" if named_months else "many values above 24, so not years"
            notes.append((f"Months instead of years ({why})", int(rest.sum()), "", "Divided by 12 to get years"))
    if feat == "Balance":                                 # overdrawn: the model only knows balances of 0 or more
        neg = values < 0
        values[neg] = 0
        status[neg] = ADJUSTED
    if feat == "NumOfProducts":
        values = np.round(values)
        many = values > 4
        values[many] = 4
        status[many] = ADJUSTED
    lo, hi = LIMITS.get(feat, (None, None))
    bad = np.isinf(values)                                # e.g. a huge number pushed past the float limit
    bad = bad.copy()
    if lo is not None:
        bad |= values < lo
    if hi is not None:
        bad |= values > hi
    values[bad] = np.nan
    status[bad] = IMPOSSIBLE
    return values, status, notes


def _category(raw, feat):
    words = CATEGORIES[feat]
    yes_no = feat in ("HasCrCard", "IsActiveMember")
    text = raw.astype("string").str.strip()
    blank = _is_blank(text.str.lower()).to_numpy()
    lookup = {}
    for u in pd.unique(text[~blank]):
        key = _plain(u)
        if yes_no:
            try:
                num = float(u)
                lookup[u] = (int(num), OK) if num in (0, 1) else (None, UNREADABLE)
                continue
            except ValueError:
                pass
        if key in words:
            lookup[u] = (words[key], OK)                                # "DE", " GERMANY ", "f", "Yes"
        elif (meaning := _closest(key, words)) is not None:
            lookup[u] = (meaning, FIXED)                                # typo: "Germny", "femal"
        elif feat == "Gender" and key in GENDER_OTHER:
            lookup[u] = (None, OTHER)                                   # a real answer the model never saw
        else:
            lookup[u] = (None, OTHER if feat == "Geography" else UNREADABLE)
    found = text.map(lambda u: lookup.get(u, (None, BLANK)) if pd.notna(u) else (None, BLANK))
    status = np.array([s for _, s in found], dtype=np.int8) if len(found) else np.array([], dtype=np.int8)
    status[blank] = BLANK
    values = pd.Series([v for v, _ in found], index=raw.index, dtype=object)
    return values, status


def _fast_category(raw, feat):
    """Same as _category but maps once per distinct value, for big files."""
    words = CATEGORIES[feat]
    text = raw.astype("string").str.strip()
    blank = _is_blank(text.str.lower())
    uniques = pd.unique(text[~blank])
    small = pd.DataFrame({"u": uniques})
    one = _category(small["u"], feat)
    value_of = dict(zip(uniques, one[0]))
    status_of = dict(zip(uniques, one[1]))
    values = text.map(value_of).astype(object)
    values = values.where(values.notna(), None)        # one kind of blank (None), whatever pandas version
    status = text.map(status_of).fillna(BLANK).to_numpy(dtype=np.int8, copy=True)
    status[blank.to_numpy()] = BLANK
    return values, status


# ------------------------------------------------------------------ the report
def _fmt(feat, value):
    if feat in MONEY:
        return f"{value:,.0f}"
    if feat in ("HasCrCard", "IsActiveMember"):
        return "Yes" if value else "No"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value)}"
    return str(value)


def _examples(raw, status, code, new=None, feat=None, k=3):
    idx = np.flatnonzero(status == code)
    seen, out = set(), []
    for i in idx[:500]:
        orig = raw.iat[i]
        orig = "(blank)" if pd.isna(orig) or str(orig).strip() == "" else str(orig).strip()
        if orig in seen:
            continue
        seen.add(orig)
        out.append(f"'{orig}' → {_fmt(feat, new[i])}" if new is not None else f"'{orig}'")
        if len(out) == k:
            break
    return ", ".join(out)


def _describe(feat, code, typical):
    fill = _fmt(feat, typical[feat])
    if code == FIXED:
        kind = "Text or symbols in a number" if feat in NUMERIC else "Spelling mistake"
        return kind, "Read correctly"
    if code == BLANK:
        return "Blank", f"Filled with typical value ({fill})"
    if code == UNREADABLE:
        return "Couldn't be read", f"Filled with typical value ({fill})"
    if code == IMPOSSIBLE:
        lo, hi = LIMITS[feat]
        if feat in MONEY:
            return "Impossible amount (negative or absurdly large)", f"Filled with typical value ({fill})"
        rule = f"{lo}–{hi}" if hi is not None else f"{lo} or more"
        return f"Impossible value (must be {rule})", f"Filled with typical value ({fill})"
    if code == OTHER:
        if feat == "Geography":
            return "Place the model doesn't know", f"Scored as {fill}: treat with care"
        return "A category the model doesn't know (it learned Male/Female)", f"Filled with typical value ({fill})"
    return {"Age": ("Birth year or date instead of age", "Turned into an age"),
            "Tenure": ("Joining year or date instead of years", "Turned into years as a customer"),
            "Balance": ("Negative balance (overdrawn)", "Set to 0, the lowest the model knows"),
            "NumOfProducts": ("More than 4 products", "Capped at 4, the most the model knows")}[feat]


def _outcome(raw):
    """The file's real outcome, if it has one: 1 = left, 0 = stayed, nan = unknown."""
    text = raw.astype("string").str.strip()
    lookup = {}
    for u in pd.unique(text.dropna()):
        try:
            num = float(u)
            lookup[u] = num if num in (0, 1) else np.nan
            continue
        except ValueError:
            pass
        key = _plain(u)
        lookup[u] = 1.0 if key in OUTCOME_YES else 0.0 if key in OUTCOME_NO else np.nan
    return text.map(lookup).to_numpy(dtype=float, na_value=np.nan)


def _fill_values(values, status, keep, currency):
    """What to fill missing or wrong cells with: the file's own typical value when it has enough real ones
    (median for numbers, most common for categories), otherwise the training data's typical customer."""
    to_eur = TO_EURO.get(currency, 1.0)
    typical = {f: (round(TYPICAL[f] / to_eur) if f in MONEY else TYPICAL[f]) for f in FEATURES}
    for f, st in status.items():
        if st is None:
            continue
        real = keep & ~np.isin(st, GUESSED) & pd.notna(values[f])
        if real.sum() < 30:
            continue
        v = values[f][real]
        if f in NUMERIC:
            med = float(np.median(v.astype(float)))
            typical[f] = round(med)
        else:
            typical[f] = pd.Series(v).mode().iloc[0]
    return typical


# ------------------------------------------------------------------ main entry point
def clean(df, mapping, name_col=None, id_col=None, outcome_col=None, currency="EUR"):
    """Clean the uploaded table and build the model-ready inputs. Returns a Cleaned result.

    Money stays in the file's currency here (so people see their own numbers); scoring converts it
    to euros for the model.
    """
    lines, summary = [], {"rows_in": len(df)}
    drop = np.full(len(df), KEPT, dtype=np.int8)

    # 1. every model input, cell by cell (all rows, so the review screen can show every problem)
    values, status, notes = {}, {}, {}
    for feat in FEATURES:
        col = mapping.get(feat)
        if not col:
            status[feat] = None                       # whole column missing: filled later, shown separately
            continue
        if feat in NUMERIC:
            values[feat], status[feat], notes[feat] = _numbers(df[col], feat, col)
        else:
            v, st = _fast_category(df[col], feat)
            values[feat], status[feat] = v.to_numpy(), st
    mapped = [f for f in FEATURES if status[f] is not None]

    # 2. exact duplicate rows
    dup = df.duplicated().to_numpy()
    drop[dup] = DUPLICATE
    copy_of = np.full(len(df), -1, dtype=np.int64)
    if dup.any():                                   # which earlier row each duplicate repeats
        h = pd.util.hash_pandas_object(df, index=False).to_numpy()
        first_pos = pd.Series(np.arange(len(df))).groupby(h).transform("min").to_numpy()
        copy_of[dup] = first_pos[dup]
        lines.append(("Whole row", "Exact duplicate row", int(dup.sum()), "", "Removed (kept the first copy)"))

    # 3. rows where nothing the model uses could be read
    unusable = drop == KEPT
    for f in mapped:
        unusable &= np.isin(status[f], [BLANK, UNREADABLE])
    drop[unusable] = EMPTY
    if unusable.any():
        lines.append(("Whole row", "No usable values for the model", int(unusable.sum()), "", "Removed"))

    # 4. repeated customer IDs, among rows that are still kept (so an empty first copy doesn't win)
    if id_col:
        ids = _ids(df[id_col])
        first = drop == KEPT
        same = first & ids.notna().to_numpy() & ids.where(first).duplicated().to_numpy()
        if first.any() and same.sum() / first.sum() > 0.2:   # too many repeats: not a unique customer ID
            lines.append((id_col, "Many rows share the same ID", int(same.sum()), "",
                          "Not treated as duplicates: this column doesn't look like a unique customer ID"))
        elif same.any():
            drop[same] = SAME_ID
            first_pos = pd.Series(np.arange(len(df))).where(first).groupby(ids.to_numpy()).transform("min").to_numpy()
            copy_of[same] = first_pos[same]
            lines.append((id_col, "Same customer ID more than once", int(same.sum()),
                          ", ".join(f"'{v}'" for v in ids[same].unique()[:3]), "Kept the first row for each ID"))

    keep = drop == KEPT
    summary.update(duplicates=int((drop == DUPLICATE).sum()), same_id=int((drop == SAME_ID).sum()),
                   empty=int(unusable.sum()))
    kept = df[keep]
    typical = _fill_values(values, status, keep, currency)

    # 5. one report line per column and kind of problem (kept rows only)
    for f in mapped:
        for problem, count, examples, action in notes.get(f, []):      # column-wide fixes (units, dates)
            lines.append((FRIENDLY[f], problem, count, examples, action))
        st_k, vals_k, raw_k = status[f][keep], values[f][keep], kept[mapping[f]]
        for code in (FIXED, ADJUSTED, BLANK, UNREADABLE, IMPOSSIBLE, OTHER):
            count = int((st_k == code).sum())
            if count:
                problem, action = _describe(f, code, typical)
                lines.append((FRIENDLY[f], problem, count,
                              _examples(raw_k, st_k, code, vals_k if code in CHANGED else None, f), action))

    # 6. fill guesses and set the types the model expects
    X = pd.DataFrame(index=kept.index)
    guessed_mask, wrong_mask = {}, {}
    for f in FEATURES:
        if status[f] is None:
            X[f] = typical[f]
            continue
        col = values[f][keep].astype(object)
        g = np.isin(status[f][keep], GUESSED)
        col[g] = typical[f]
        guessed_mask[f] = g
        wrong_mask[f] = np.isin(status[f][keep], WRONG)
        X[f] = col
    for f in ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]:
        X[f] = X[f].astype(float)
        X.loc[~np.isfinite(X[f]), f] = float(typical[f])      # last safety net: the model can't take inf/nan
    for f in ["NumOfProducts", "HasCrCard", "IsActiveMember"]:
        X[f] = X[f].astype(float).round().astype(int)
    for f in ["Geography", "Gender"]:
        X[f] = X[f].astype(str)

    # 7. per customer: which inputs were guessed
    guessed = pd.Series("", index=X.index, dtype=object)
    for f, g in wrong_mask.items():
        if g.any():
            guessed = guessed.where(~g, guessed + np.where(guessed == "", "", ", ") + FRIENDLY[f])

    names, name_lines = _names(kept, name_col, id_col)
    lines.extend(name_lines)
    name_blank = None
    if name_col:
        nm = df[name_col].astype("string").str.strip()
        name_blank = (nm.isna() | nm.str.lower().isin(NULLS)).to_numpy(dtype=bool, na_value=True)

    # 8. headline numbers
    n = len(X)
    real_share = {f: 0.0 if status[f] is None else (1 - float(guessed_mask[f].mean()) if n else 0.0)
                  for f in FEATURES}
    any_guess = np.zeros(n, dtype=bool)
    any_change = np.zeros(n, dtype=bool)
    any_unknown = np.zeros(n, dtype=bool)
    for f in mapped:
        any_guess |= wrong_mask[f]
        any_change |= np.isin(status[f][keep], CHANGED)
        any_unknown |= status[f][keep] == OTHER
    summary.update(
        rows_out=n,
        fixed=sum(int(np.isin(status[f][keep], CHANGED).sum()) for f in mapped),
        filled=sum(int(g.sum()) for g in wrong_mask.values()),
        customers_guessed=int(any_guess.sum()),
        customers_unknown=int(any_unknown.sum()),
        rows_fixed_only=int((any_change & ~any_guess).sum()),
        removed=int((~keep).sum()),
        problem_columns=[mapping[f] for f in mapped if wrong_mask[f].any()],
        other_country=int((status["Geography"][keep] == OTHER).sum()) if status["Geography"] is not None else 0)
    places, place = None, None
    if status["Geography"] is not None:
        st_k = status["Geography"][keep]
        raw_k = kept[mapping["Geography"]].astype("string").str.strip()
        shown = np.where(st_k == OTHER, raw_k.str.title().fillna("Unknown").to_numpy(dtype=object),
                         X["Geography"].to_numpy(dtype=object))
        place = pd.Series(np.where(st_k == BLANK, "Unknown", shown), dtype=object)
    if status["Geography"] is not None and n:
        unknown = status["Geography"][keep] == OTHER
        raw_places = kept[mapping["Geography"]].astype("string").str.strip()
        plain_places = raw_places.str.lower()
        places = {"unknown_share": float(unknown.mean()),
                  "examples": raw_places[unknown].value_counts().head(3).index.tolist(),
                  "indian_share": float(plain_places.isin(INDIAN_PLACES).mean())}
    outcome = _outcome(df[outcome_col])[keep] if outcome_col and outcome_col in df.columns else None
    if currency != "EUR":
        lines.append(("Money", f"Amounts are in {currency}", n, "",
                      f"Converted to euros for the model (1 € = {1 / TO_EURO[currency]:,.2f} {currency}); "
                      "shown in your currency"))
    report = pd.DataFrame(lines, columns=["Column", "Problem", "Rows", "Examples", "What the tool did"])
    return Cleaned(X.reset_index(drop=True), names.reset_index(drop=True), guessed.reset_index(drop=True),
                   report, summary, real_share, status=status, values=values, drop=drop, copy_of=copy_of,
                   name_blank=name_blank, mapping={f: mapping[f] for f in mapped}, typical=typical,
                   outcome=outcome, outcome_col=outcome_col if outcome is not None else None,
                   currency=currency, places=places, place=place)


def _ids(series):
    """IDs as text; Excel often turns 15634602 into 15634602.0."""
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def _names(df, name_col, id_col):
    lines = []
    ids = _ids(df[id_col]) if id_col else None
    fallback = ("Customer " + ids) if id_col else pd.Series([f"Row {i + 1}" for i in df.index], index=df.index)
    if fallback.isna().any():
        fallback = fallback.fillna(pd.Series([f"Row {i + 1}" for i in df.index], index=df.index))
    if not name_col:
        return fallback.astype(str), lines
    names = df[name_col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    shouty = (names.str.isupper() | names.str.islower()).fillna(False) & names.str.contains(r"[A-Za-z]", na=False)
    if shouty.any():
        before = names[shouty].head(2).tolist()
        names = names.where(~shouty, names.str.title())
        lines.append(("Name", "ALL CAPS or all lower-case name", int(shouty.sum()),
                      ", ".join(f"'{b}' → '{b.title()}'" for b in before), "Tidied the capitals"))
    if _key(name_col) in ("surname", "lastname") and id_col:
        names = names + " (" + ids + ")"
    blank = names.isna() | (names == "") | names.str.lower().isin(NULLS)
    if blank.any():
        lines.append(("Name", "Blank name", int(blank.sum()), "",
                      "Used the customer ID instead" if id_col else "Used the row number instead"))
        names = names.where(~blank, fallback)
    return names.astype(str), lines
