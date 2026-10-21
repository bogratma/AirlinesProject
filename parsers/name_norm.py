import re

import polars as pl
from unidecode import unidecode

TITLES = r"\b(MR|MRS|MS|MISS|MSTR|MX)\b"


def name_skeleton(raw, max_tokens=None):
    if raw is None:
        return None
    s = unidecode(str(raw)).upper()
    s = re.sub(r"[’'`\-]", "", s)
    s = re.sub(TITLES, " ", s)
    toks = [t for t in re.split(r"[^A-Z]+", s) if len(t) > 1]
    if max_tokens:
        toks = toks[:max_tokens]
    out = []
    for t in toks:
        t = t.replace("X", "KS").replace("Y", "I")
        t = re.sub(r"(.)\1+", r"\1", t)
        out.append(t)
    return " ".join(sorted(out)) if out else None


def with_name_key(df, raw_expr, max_tokens=None):
    e = raw_expr.cast(pl.String).str.strip_chars()
    df = df.with_columns(
        pl.when(e.is_in(["", "N/A", "Not presented"])).then(None).otherwise(e).alias("_raw_name"))
    uniq = df["_raw_name"].drop_nulls().unique().to_list()
    mapping = {n: name_skeleton(n, max_tokens) for n in uniq}
    mapping = {k: v for k, v in mapping.items() if v}
    return df.with_columns(
        pl.col("_raw_name").replace_strict(mapping, default=None, return_dtype=pl.String).alias("_name_key"))