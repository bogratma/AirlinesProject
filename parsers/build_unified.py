import os
import polars as pl

from .unify import (unify_tab, unify_yaml, unify_json, unify_xml, unify_csv, unify_boarding)

DEFAULT_PATHS = {
    "tab": "clean_tab.parquet", "yaml": "clean_yaml.parquet", "json": "clean_json.parquet",
    "xml": "clean_xml.parquet", "csv": "clean_csv.parquet", "boarding": "clean_zip.parquet",
}


def learn_program_codes(xml_df, code_parts):
    cards = (pl.scan_parquet(code_parts)
               .select("card_program", "card_number").drop_nulls().unique().collect())
    x = (xml_df.select(pl.col("bonus_program").alias("name"), pl.col("card_number"))
               .drop_nulls().unique())
    j = x.join(cards, on="card_number")
    if j.is_empty():
        return {}
    cnt = j.group_by("name", "card_program").len().sort("len", descending=True)
    best = cnt.group_by("name", maintain_order=True).first()
    total = cnt.group_by("name").agg(pl.col("len").sum().alias("total"))
    rep = best.join(total, on="name").with_columns(share=(pl.col("len") / pl.col("total")).round(2))
    print("Программы xml -> коды (по совпадению номеров карт):")
    print(rep.select("name", "card_program", "len", "share"))
    return dict(zip(best["name"], best["card_program"]))

def build_unified_file(process_dir="process", out_name="unified.parquet", paths=None):
    paths = {**DEFAULT_PATHS, **(paths or {})}
    full = {k: f"{process_dir}/{v}" for k, v in paths.items()}
    parts = {}

    def write_part(name, df):
        p = f"{process_dir}/_unified_{name}.parquet"
        df.write_parquet(p)
        parts[name] = p
        print(f"{name}: {len(df)} строк")
    for name, fn in [("tab", unify_tab), ("yaml", unify_yaml), ("json", unify_json),
                     ("boarding", unify_boarding), ("csv", unify_csv)]:
        write_part(name, fn(pl.read_parquet(full[name])))

    xml_df = pl.read_parquet(full["xml"])
    codes = learn_program_codes(xml_df, [parts[n] for n in ("tab", "yaml", "json", "boarding")])
    write_part("xml", unify_xml(xml_df, codes))

    out = f"{process_dir}/{out_name}"
    pl.scan_parquet(list(parts.values())).sink_parquet(out)
    for p in parts.values():
        os.remove(p)
    print(f"Готово: {out}")
    return out
