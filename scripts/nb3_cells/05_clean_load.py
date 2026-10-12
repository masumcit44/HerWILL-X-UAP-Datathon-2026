# Notebook cell 05 - code: text cleaning, metrics, load CSVs
# --- text cleaning and metrics ------------------------------------------------
URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def clean_text(t):
    """URL -> ' URL ', @mention -> ' USER ', collapse whitespace, strip."""
    s = "" if t is None else str(t)
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    return WS_RE.sub(" ", s).strip()


def macro_f1(y_true, proba):
    return float(f1_score(y_true, np.asarray(proba).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def per_class_f1(y_true, proba):
    return [round(float(v), 6) for v in
            f1_score(y_true, np.asarray(proba).argmax(1), average=None,
                     labels=CLASSES, zero_division=0)]


train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
test = pd.read_csv(TEST_CSV, encoding="utf-8")
sample = pd.read_csv(SAMPLE_CSV)
assert list(train.columns) == ["id", "text", "y"], list(train.columns)
assert list(test.columns) == ["id", "text"], list(test.columns)
assert list(sample.columns) == ["id", "y_pred"], list(sample.columns)
for df in (train, test):
    df["text"] = df["text"].astype("string").fillna("").astype(str)
train["clean"] = train["text"].map(clean_text)
test["clean"] = test["text"].map(clean_text)

y = train["y"].to_numpy(dtype=int)
ids = train["id"].to_numpy(dtype=np.int64)
test_ids = test["id"].to_numpy(dtype=np.int64)
clean_tr = train["clean"].to_numpy()
clean_te = test["clean"].to_numpy()
n_tr, n_te = len(train), len(test)
print("train {} test {}".format(n_tr, n_te))
