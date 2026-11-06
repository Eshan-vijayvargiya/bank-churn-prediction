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
ID_ALIASES = ["customerid", "id", "clientid", "accountid", "accountno", "accountnumber"]

SUPPORTED = ["csv", "txt", "xlsx", "xls", "json", "pdf", "docx"]

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
    for i, row in enumerate(rows[:limit]):
        if sum(_key(c) in known for c in row) >= 2:
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
    if ext in ("csv", "txt"):
        return _read_csv(data)
    if ext in ("xlsx", "xls"):
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


def read_any(name, data):
    """Return a DataFrame from the uploaded file's name and bytes, with blank rows and columns removed."""
    df = _read_raw(name, data)
    df.columns = _tidy_columns(df.columns)
    df = df.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
    empty = [c for c in df.columns if c.startswith("Column ") and df[c].isna().all()]
    return df.drop(columns=empty).reset_index(drop=True)


def guess_mapping(columns):
    """Match file columns to model inputs by name. Returns {feature: column or None}."""
    keyed = {_key(c): c for c in columns}
    mapping = {}
    for feat, names in ALIASES.items():
        mapping[feat] = next((keyed[n] for n in names if n in keyed), None)
    return mapping


def find_column(columns, names):
    keyed = {_key(c): c for c in columns}
    return next((keyed[n] for n in names if n in keyed), None)
