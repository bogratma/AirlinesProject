import os

import polars as pl

from .unify import (unify_tab, unify_yaml, unify_json, unify_xml, unify_csv, unify_boarding,
                    airports_from_schedule)

DEFAULT_PATHS = {
    "tab": "clean_tab.parquet", "yaml": "clean_yaml.parquet", "json": "clean_json.parquet",
    "xml": "clean_xml.parquet", "csv": "clean_csv.parquet", "boarding": "clean_zip.parquet",
}


def pdf_schedule_path(process_dir="process"):
    """Путь к распарсенному расписанию или None, если файла нет."""
    p = f"{process_dir}/clean_pdf.parquet/skyteam_schedule.parquet"
    return p if os.path.exists(p) else None


def learn_program_codes(xml_df, code_parts):
    # читаем файлы по отдельности: схемы частей могут различаться
    cards = (pl.concat([pl.scan_parquet(p).select("card_program", "card_number") for p in code_parts])
             .drop_nulls().unique().collect())
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


def _city_map(pdf_parquet):
    """город -> IATA только для городов с одним аэропортом (у Москвы их несколько, её не угадать)."""
    airports = airports_from_schedule(pdf_parquet).drop_nulls("city")
    uniq = (airports.group_by(pl.col("city").str.to_uppercase().str.strip_chars().alias("city"))
            .agg(pl.col("airport").unique().alias("a"))
            .filter(pl.col("a").list.len() == 1))
    return {c: a[0].strip().upper() for c, a in uniq.iter_rows() if c and a[0]}


def build_unified_file(process_dir="process", out_name="unified_raw.parquet", paths=None):
    """Собирает общую таблицу из clean_*.parquet. Сверка с PDF и person_id делаются отдельно (main)."""
    paths = {**DEFAULT_PATHS, **(paths or {})}
    full = {k: f"{process_dir}/{v}" for k, v in paths.items()}
    parts = {}

    def write_part(name, df):
        p = f"{process_dir}/_unified_{name}.parquet"
        df.write_parquet(p)
        parts[name] = p
        print(f"{name}: {len(df)} строк")

    pdf_parquet = pdf_schedule_path(process_dir)
    city_map = {}
    if pdf_parquet:
        print("Загружаем справочник аэропортов из расписания PDF...")
        city_map = _city_map(pdf_parquet)
        print(f"Городов с однозначным аэропортом: {len(city_map)}")

    for name, fn in [("tab", unify_tab), ("yaml", unify_yaml), ("json", unify_json),
                     ("boarding", unify_boarding)]:
        write_part(name, fn(pl.read_parquet(full[name])))
    if os.path.exists(full["csv"]):
        write_part("csv", unify_csv(pl.read_parquet(full["csv"]), city_map))

    xml_df = pl.read_parquet(full["xml"])
    codes = learn_program_codes(xml_df, [parts[n] for n in ("tab", "yaml", "json", "boarding")])
    write_part("xml", unify_xml(xml_df, codes))

    out = f"{process_dir}/{out_name}"
    pl.concat([pl.scan_parquet(p) for p in parts.values()], how="diagonal_relaxed").sink_parquet(out)
    for p in parts.values():
        os.remove(p)
    print(f"Готово: {out}")
    return out