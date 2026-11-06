"""Read a customer file of almost any type into a table, and match its columns to what the model needs."""
import io
import re
from pathlib import Path

import pandas as pd

FEATURES = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

# Other column names people commonly use for each model input
ALIASES = {
    "CreditScore": ["creditscore", "credit", "score", "cibil", "creditrating"],
    "Geography": ["geography", "country", "region", "location", "nation"],
    "Gender": ["gender", "sex"],
    "Age": ["age", "ageyears"],
    "Tenure": ["tenure", "years", "yearsascustomer", "yearswithbank", "customersince"],
    "Balance": ["balance", "accountbalance", "deposit", "deposits"],
    "NumOfProducts": ["numofproducts", "products", "numberofproducts", "productcount", "noofproducts"],
    "HasCrCard": ["hascrcard", "hascreditcard", "creditcard", "card", "hascard"],
    "IsActiveMember": ["isactivemember", "activemember", "active", "isactive"],
    "EstimatedSalary": ["estimatedsalary", "salary", "income", "annualincome"],
}
FRIENDLY = {"CreditScore": "Credit score", "Geography": "Country", "Gender": "Gender", "Age": "Age",
            "Tenure": "Years as customer", "Balance": "Account balance", "NumOfProducts": "Number of products",
            "HasCrCard": "Has credit card", "IsActiveMember": "Active member", "EstimatedSalary": "Estimated salary"}
NAME_ALIASES = ["customername", "name", "fullname", "customer", "clientname", "client", "surname", "lastname"]
ID_ALIASES = ["customerid", "id", "clientid", "accountid", "accountno", "accountnumber", "clientnum", "custid"]
OUTCOME_ALIASES = ["exited", "churn", "churned", "attrition", "attritionflag", "left", "churnflag"]

SUPPORTED = ["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "json", "pdf", "docx"]

# Used when a file has no column for an input: the typical value in the 10,000-customer
# training data (median for numbers, most common value for categories).
TYPICAL = {"CreditScore": 652, "Geography": "France", "Gender": "Male", "Age": 37, "Tenure": 5,
           "Balance": 97198.54, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1,
           "EstimatedSalary": 100193.92}


def _key(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _known_names():
    names = {n for group in ALIASES.values() for n in group} | set(NAME_ALIASES) | set(ID_ALIASES)
    return names | {_key(f) for f in FEATURES}


def _header_row(rows, limit=30):
    """Index of the real header row. Files often have a title or export note above it."""
    known = _known_names()

    def looks_like_column(cell):
        if _key(cell) in known:
            return True
        return any(_score(cell, RULES[f], set()) >= 60 for f in FEATURES) or \
            any(_score(cell, rules, set()) >= 60 for rules in (NAME_RULES, ID_RULES, OUTCOME_RULES))
    for i, row in enumerate(rows[:limit]):
        if sum(looks_like_column(c) for c in row if isinstance(c, str) and 0 < len(c) < 60) >= 2:
            return i
    return 0


def _promote_header(df):
    """Use the detected header row of a table that was read without one (PDF, Word, Excel)."""
    df = df.dropna(how="all").reset_index(drop=True)
    h = _header_row(df.head(30).astype(str).values.tolist())
    out = df.iloc[h + 1:].copy()
    out.columns = [str(c).strip() for c in df.iloc[h]]
    return out.reset_index(drop=True).infer_objects()


def _delimiter(lines):
    """The delimiter that splits the most lines into the same number of fields."""
    best, best_score = ",", 0
    for d in [",", ";", "\t", "|"]:
        counts = [line.count(d) for line in lines if line.strip()]
        nonzero = [c for c in counts if c]
        if nonzero:
            mode = max(set(nonzero), key=nonzero.count)
            score = nonzero.count(mode) * (1 + min(mode, 50) / 100)
            if score > best_score:
                best, best_score = d, score
    return best


def _read_csv(data):
    import codecs
    import csv
    head = codecs.getincrementaldecoder("utf-8")(errors="replace").decode(data[:65536])
    lines = head.lstrip("\ufeff").splitlines()[:30]
    sep = _delimiter(lines)
    skip = _header_row(list(csv.reader(lines, delimiter=sep)))
    for encoding in ("utf-8-sig", "latin-1"):          # Excel-made CSVs are often not UTF-8
        try:
            return pd.read_csv(io.BytesIO(data), sep=sep, skiprows=skip, encoding=encoding,
                               low_memory=False)       # fast C reader, fine for millions of rows
        except UnicodeDecodeError:
            continue


def _read_excel(data):
    """Read every sheet and keep the one that looks most like a customer table."""
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
    known = _known_names()

    def fit(raw):
        rows = raw.head(30).astype(str).values.tolist()
        return sum(_key(c) in known for c in rows[_header_row(rows)]), len(raw)
    raw = max(sheets.values(), key=fit)
    return _promote_header(raw)


def _read_raw(name, data):
    ext = Path(name).suffix.lower().lstrip(".")
    if ext in ("csv", "tsv", "txt"):
        return _read_csv(data)
    if ext in ("xlsx", "xlsm", "xls"):
        return _read_excel(data)
    if ext == "json":
        return pd.read_json(io.BytesIO(data))
    if ext == "pdf":
        import pdfplumber
        frames = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()          # tables drawn with lines
                if not tables:                          # tables laid out with spacing only
                    tables = page.extract_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})
                for table in tables:
                    frames.append(pd.DataFrame(table))
        if not frames:
            raise ValueError("No table found in this PDF. The PDF needs a real table, not a scanned image.")
        first = _promote_header(frames[0])
        rest = []
        for f in frames[1:]:   # tables that continue on later pages may repeat the header
            f = f.dropna(how="all")
            if [str(c).strip() for c in f.iloc[0]] == list(first.columns):
                f = f.iloc[1:]
            f.columns = first.columns[: f.shape[1]]
            rest.append(f)
        return pd.concat([first, *rest], ignore_index=True)
    if ext == "docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        if not doc.tables:
            raise ValueError("No table found in this Word file. Put the customers in a Word table.")
        rows = [[cell.text for cell in row.cells] for t in doc.tables for row in t.rows]
        return _promote_header(pd.DataFrame(rows))
    raise ValueError(f"'.{ext}' files aren't supported. Use one of: {', '.join(SUPPORTED)}.")


def _tidy_columns(columns):
    """Strip column names, name blank ones, and make repeated names unique."""
    out, seen = [], {}
    for i, c in enumerate(columns):
        c = str(c).strip()
        if c == "" or c.lower() in ("nan", "none") or c.startswith("Unnamed:"):
            c = f"Column {i + 1}"
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else f"{c} ({seen[c]})")
    return out


FRIENDLY_READ_ERROR = {
    "csv": "It doesn't look like a normal text table. Check that it's a real CSV (open it in Notepad to see).",
    "xlsx": "Excel couldn't open it. It may be damaged or password-protected: open it in Excel and save a fresh copy.",
    "json": "It isn't a JSON list of customers (records). Try saving the data as CSV or Excel instead.",
    "pdf": "The PDF couldn't be read. It may be scanned, damaged or password-protected.",
    "docx": "The Word file couldn't be opened. It may be damaged, or an old .doc file: save it as .docx.",
}


def read_any(name, data):
    """Return a DataFrame from the uploaded file's name and bytes, with blank rows and columns removed.

    Any reading problem becomes a ValueError with a plain-English reason, so the app can explain it.
    """
    ext = Path(name).suffix.lower().lstrip(".")
    if not data:
        raise ValueError("The file is empty.")
    try:
        df = _read_raw(name, data)
    except ValueError as err:
        if "No table found" in str(err) or "aren't supported" in str(err):
            raise
        kind = {"tsv": "csv", "txt": "csv", "xlsm": "xlsx", "xls": "xlsx"}.get(ext, ext)
        raise ValueError(FRIENDLY_READ_ERROR.get(kind, "The file couldn't be read.")) from err
    except Exception as err:
        kind = {"tsv": "csv", "txt": "csv", "xlsm": "xlsx", "xls": "xlsx"}.get(ext, ext)
        raise ValueError(FRIENDLY_READ_ERROR.get(kind, "The file couldn't be read.")) from err
    if df is None or df.shape[1] == 0:
        raise ValueError("No columns were found in this file.")
    df.columns = _tidy_columns(df.columns)
    df = df.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
    empty = [c for c in df.columns if c.startswith("Column ") and df[c].isna().all()]
    return df.drop(columns=empty).reset_index(drop=True)


# ------------------------------------------------------------------ matching columns by name
def _tokens(name):
    """The words in a column name: 'CIBIL_Score' -> {cibil, score}, 'NumOfProducts' -> {num, of, products}."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}


# Each rule is (score, groups, blocked): a column fits when its name has a word from EVERY group
# and none of the blocked words. Higher scores win. This catches names real files use
# ("CIBIL_Score", "Account_Balance (INR)", "Months_on_book") and avoids traps
# ("Months_Inactive_12_mon" is not an active-member flag, "Credit_Limit" is not a credit card flag).
MONEY_BLOCK = {"revolving", "loan", "due", "outstanding", "emi", "limit", "card"}
CARD_BLOCK = {"limit", "spend", "spending", "utilization", "number", "no", "type", "category", "bill", "balance",
              "bal", "count", "amount", "expiry", "issuer", "network"}
RULES = {
    "CreditScore": [(90, [{"cibil", "fico", "creditscore"}], set()),
                    (85, [{"credit", "cr"}, {"score", "rating", "rank"}], {"risk", "churn"}),
                    (30, [{"score"}], {"risk", "churn", "satisfaction", "nps", "csat", "probability", "propensity",
                                       "engagement", "loyalty"})],
    "Geography": [(90, [{"country", "geography", "nation", "nationality"}], {"code"}),
                  (70, [{"state", "province", "region", "zone"}], {"marital", "account", "statement"}),
                  (60, [{"city", "district", "location", "residence", "town", "area"}], {"code", "pin", "pincode"})],
    "Gender": [(90, [{"gender", "sex"}], set())],
    "Age": [(90, [{"age"}], {"account", "relationship", "card", "loan", "vintage"}),
            (85, [{"dob", "birthdate", "dateofbirth", "yob", "birthday"}], set()),
            (80, [{"birth"}, {"date", "year", "dt", "day"}], set())],
    "Tenure": [(90, [{"tenure", "vintage"}], set()),
               (85, [{"account", "relationship"}, {"age"}], set()),
               (80, [{"years", "year", "yrs", "months", "month", "mon"},
                     {"bank", "customer", "relationship", "book", "with", "association", "membership"}],
                {"inactive", "contacts", "contact", "birth"}),
               (75, [{"customer", "member", "client", "account", "relationship"}, {"since", "start", "opened", "open"}],
                {"balance", "bal"}),
               (70, [{"joining", "joined", "join", "opening", "onboarding", "onboarded", "since"}],
                {"fee", "bonus", "channel", "balance"})],
    "Balance": [(90, [{"balance", "bal"}], MONEY_BLOCK),
                (80, [{"amb", "aqb", "deposit", "deposits", "casa", "savings"}], MONEY_BLOCK | {"account", "type"})],
    "NumOfProducts": [(90, [{"products", "numofproducts", "noofproducts"}], {"type", "category", "name", "code"}),
                      (80, [{"product"}, {"count", "num", "number", "no", "total", "held", "holding", "holdings"}],
                       {"type", "category", "name", "code"}),
                      (70, [{"relationship"}, {"count"}], set())],
    "HasCrCard": [(90, [{"hascrcard", "creditcard"}], CARD_BLOCK),
                  (85, [{"card"}, {"credit", "cr", "has", "holder", "have", "owns", "own"}], CARD_BLOCK)],
    "IsActiveMember": [(90, [{"active", "isactive", "activemember"}],
                        {"inactive", "month", "months", "mon", "days", "day", "count", "last", "period", "since"}),
                       (70, [{"activity"}, {"status", "flag"}], set())],
    "EstimatedSalary": [(90, [{"salary", "income", "ctc", "earnings", "earning", "wage", "wages", "lpa"}], {"tax"})],
}
NAME_RULES = [(90, [{"customer", "client", "account", "holder", "member", "full", "cust"}, {"name"}],
               {"bank", "branch", "product", "father", "mother", "nominee", "user"}),
              (80, [{"name", "customername", "fullname", "clientname"}],
               {"bank", "branch", "product", "card", "file", "company", "father", "mother", "nominee", "employer",
                "manager", "city", "state", "scheme", "user", "username", "plan", "agent", "sheet"}),
              (70, [{"surname", "lastname"}], set())]
ID_RULES = [(90, [{"customer", "client", "cust", "account", "member", "ac", "acct"}, {"id", "no", "number", "num", "code"}],
             {"branch", "product", "type"}),
            (60, [{"id"}], {"branch", "product", "row", "transaction", "txn", "session"})]
OUTCOME_RULES = [(95, [{"exited", "churn", "churned", "attrition", "attrited", "churnflag", "attritionflag"}],
                  {"date", "reason", "score", "probability", "risk", "prediction", "predicted", "propensity",
                   "category", "month", "months"}),
                 (85, [{"left", "closed", "closure"}], {"date", "reason", "balance", "hand", "side"}),
                 (60, [{"target", "label"}], set())]


def _score(col, rules, aliases):
    if _key(col) in aliases:
        return 100
    words = _tokens(col)
    return max((score for score, groups, blocked in rules
                if all(words & g for g in groups) and not words & blocked), default=0)


def match_columns(columns):
    """Match file columns to the model's inputs, a name, an ID and (if present) the real outcome.

    Every candidate gets a score, the best ones are taken first, and no column is used twice.
    Returns {"mapping": {feature: column or None}, "name": ..., "id": ..., "outcome": ...}.
    """
    columns = list(columns)
    roles = {f: (RULES[f], set(ALIASES[f])) for f in FEATURES}
    roles.update({"name": (NAME_RULES, set(NAME_ALIASES)), "id": (ID_RULES, set(ID_ALIASES)),
                  "outcome": (OUTCOME_RULES, set(OUTCOME_ALIASES))})
    candidates = []
    for position, col in enumerate(columns):
        for role, (rules, aliases) in roles.items():
            score = _score(col, rules, aliases)
            if score:
                candidates.append((-score, position, role, col))
    taken, used = {}, set()
    for _, _, role, col in sorted(candidates):        # best score first, then left-most column
        if role not in taken and col not in used:
            taken[role] = col
            used.add(col)
    return {"mapping": {f: taken.get(f) for f in FEATURES},
            "name": taken.get("name"), "id": taken.get("id"), "outcome": taken.get("outcome")}


def guess_mapping(columns):
    """Match file columns to model inputs by name. Returns {feature: column or None}."""
    return match_columns(columns)["mapping"]


def find_column(columns, names):
    keyed = {_key(c): c for c in columns}
    return next((keyed[n] for n in names if n in keyed), None)


# ------------------------------------------------------------------ money: which currency is the file in?
TO_EURO = {"EUR": 1.0, "INR": 1 / 106.1, "USD": 1 / 1.12}     # fixed rates, early November 2026
CURRENCY_LABEL = {"EUR": "Euro (€), same as the model", "INR": "Indian rupee (₹)", "USD": "US dollar ($)"}
CURRENCY_SYMBOL = {"EUR": "€", "INR": "₹", "USD": "$"}
INDIAN_PLACES = {
    "india", "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat", "haryana",
    "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "orissa", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal", "delhi", "new delhi", "jammu and kashmir", "ladakh", "chandigarh",
    "puducherry", "mumbai", "bengaluru", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur",
    "lucknow", "surat", "indore", "bhopal", "nagpur", "noida", "gurgaon", "gurugram", "kanpur", "patna", "vadodara",
    "coimbatore", "kochi", "visakhapatnam", "thane", "ludhiana", "agra", "nashik", "rajkot", "varanasi", "ranchi"}


def detect_currency(df, mapping):
    """Guess the file's currency from column names, symbols in the values, and the places listed."""
    votes = {"INR": 0, "USD": 0, "EUR": 0}
    for feat in ("Balance", "EstimatedSalary"):
        col = mapping.get(feat)
        if not col:
            continue
        words = _tokens(col)
        if words & {"inr", "rs", "rupee", "rupees", "lpa", "lakh", "lakhs", "lac", "lacs", "crore", "crores"}:
            votes["INR"] += 3
        if words & {"usd", "dollar", "dollars"}:
            votes["USD"] += 3
        if words & {"eur", "euro", "euros"}:
            votes["EUR"] += 3
        sample = df[col].dropna().astype(str).head(2000).str.lower()
        votes["INR"] += int(sample.str.contains(r"₹|\brs\.?|inr|lakh|lac\b|crore|lpa|\d\s*l\b", regex=True).mean() > 0.05) * 2
        votes["USD"] += int(sample.str.contains(r"\$|usd", regex=True).mean() > 0.05) * 2
        votes["EUR"] += int(sample.str.contains(r"€|eur", regex=True).mean() > 0.05) * 2
    geo = mapping.get("Geography")
    if geo:
        places = df[geo].dropna().astype(str).head(5000).str.strip().str.lower()
        if len(places) and places.isin(INDIAN_PLACES).mean() > 0.5:
            votes["INR"] += 2
    best = max(votes, key=votes.get)
    return best if votes[best] > 0 else "EUR"
