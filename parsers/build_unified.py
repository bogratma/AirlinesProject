import os
import polars as pl
from transliterate import translit

from .unify import (unify_tab, unify_yaml, unify_json, unify_xml, unify_csv, unify_boarding, airports_from_schedule)

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

    pdf_parquet = f"{process_dir}/clean_pdf.parquet/skyteam_schedule.parquet"

    city_map = {}
    if os.path.exists(pdf_parquet):
        print("Загружаем справочник аэропортов из расписания PDF...")
        airports_df = airports_from_schedule(pdf_parquet).drop_nulls("city")
        for row in airports_df.select("city", "airport").iter_rows():
            city, iata = row
            if city and iata:
                city_map[city.strip().upper()] = iata.strip().upper()
        print(f"Загружено городов в справочник: {len(city_map)}")
    for name, fn in [("tab", unify_tab), ("yaml", unify_yaml), ("json", unify_json),
                     ("boarding", unify_boarding)]:
        write_part(name, fn(pl.read_parquet(full[name])))
    if os.path.exists(full["csv"]):
        csv_df = pl.read_parquet(full["csv"])
        write_part("csv", unify_csv(csv_df, city_map))

    xml_df = pl.read_parquet(full["xml"])
    codes = learn_program_codes(xml_df, [parts[n] for n in ("tab", "yaml", "json", "boarding")])
    write_part("xml", unify_xml(xml_df, codes))

    out = f"{process_dir}/{out_name}"

    # Собираем общую базу из частей
    pl.scan_parquet(list(parts.values())).sink_parquet(out)
    for p in parts.values():
        os.remove(p)
    print(f"Готово: {out}")
    pdf_parquet = f"{process_dir}/clean_pdf.parquet/skyteam_schedule.parquet"
    if os.path.exists(pdf_parquet):
        print("Запуск сверки с официальным расписанием из PDF...")
        unified_df = pl.read_parquet(out)
        enriched_df = enrich_and_validate_with_pdf(unified_df, pdf_parquet)
        enriched_df.write_parquet(out)
        print("Сверка с PDF успешно завершена, колонка route_mismatch добавлена!")
    else:
        print(f"Файл расписания PDF не найден по пути: {pdf_parquet} (сверка пропущена)")

    return out


def enrich_and_validate_with_pdf(unified_df, pdf_schedule_path):
    pdf_schedule = pl.read_parquet(pdf_schedule_path).select(
        pl.col("flight").alias("flight_code"),
        pl.col("orig_iata").alias("pdf_dep"),
        pl.col("dest_iata").alias("pdf_arr"),
        pl.col("valid_from"),
        pl.col("valid_to"),
        pl.col("days"),
        pl.col("aircraft"),
        pl.col("operated_by")
    ).unique(subset=["flight_code"])
    enriched = unified_df.join(
        pdf_schedule,
        on="flight_code",
        how="left"
    )
    checked = enriched.with_columns(
        route_mismatch=(
                               (pl.col("dep_airport").is_not_null()) &
                               (pl.col("pdf_dep").is_not_null()) &
                               (pl.col("dep_airport") != pl.col("pdf_dep"))
                       ) | (
                               (pl.col("arr_airport").is_not_null()) &
                               (pl.col("pdf_arr").is_not_null()) &
                               (pl.col("arr_airport") != pl.col("pdf_arr"))
                       )
    )

    return checked


def export_suspicious_passengers(df_with_ids, output_prefix="process/suspicious_spies"):
    print("\n=== ЭТАП 5: Анализ аномалий и экспорт отчета по шпионам ===")

    # 1. Динамически вычисляем конфликты аэропортов по источникам из df_with_ids
    conflicts = (
        df_with_ids.filter(
            pl.col("person_id").is_not_null() &
            pl.col("flight_code").is_not_null() &
            pl.col("date").is_not_null()
        )
        .group_by(["person_id", "flight_code", "date"])
        .agg(
            dep_conflict=(pl.col("dep_airport").drop_nulls().n_unique() > 1),
            arr_conflict=(pl.col("arr_airport").drop_nulls().n_unique() > 1)
        )
    )

    df_conf = df_with_ids.join(conflicts, on=["person_id", "flight_code", "date"], how="left")

    # 2. Инициализируем список аномалий
    anomalies_list = []

    # Аномалия А: Расхождение с официальным расписанием PDF
    if "route_mismatch" in df_conf.columns:
        pdf_mismatches = df_conf.filter(
            pl.col("route_mismatch") == True
        ).select(
            "person_id", "flight_code", "date",
            pl.lit("Маршрут расходится с официальным расписанием PDF").alias("reason"),
            pl.lit(3).alias("severity")
        )
        anomalies_list.append(pdf_mismatches)

    # Аномалия Б: Конфликт аэропортов между источниками
    source_conflicts = df_conf.filter(
        pl.col("arr_conflict") | pl.col("dep_conflict")
    ).select(
        "person_id", "flight_code", "date",
        pl.lit("Конфликт данных между источниками (разные аэропорты)").alias("reason"),
        pl.lit(3).alias("severity")
    )
    anomalies_list.append(source_conflicts)
    shared_cards = (
        df_conf.filter(pl.col("card_number").is_not_null())
        .group_by(["card_program", "card_number"])
        .agg(
            # Считаем уникальных владельцев по ИМЕНИ, а не по внутреннему person_id
            unique_owners=pl.col("name_key").n_unique(),
            owners=pl.col("name_key").drop_nulls().unique().sort()
        )
        .filter(pl.col("unique_owners") > 1)
    )
    card_sharing = (
        df_conf.join(shared_cards, on=["card_program", "card_number"], how="inner")
        .select(
            "person_id", "flight_code", "date",
            pl.concat_str([
                pl.lit("Шеринг карты лояльности (программа: "),
                pl.col("card_program").fill_null(""),
                pl.lit("). Владельцы: ["),
                pl.col("owners").list.join(", "),
                pl.lit("]")
            ]).alias("reason"),
            pl.lit(2).alias("severity")
        )
    )
    anomalies_list.append(card_sharing)
    multi_passports = (
        df_conf.filter(pl.col("passport").is_not_null())
        .group_by("person_id")
        .agg(pl.col("passport").n_unique().alias("pass_count"))
        .filter(pl.col("pass_count") > 1)
    )

    passport_anomalies = (
        df_conf.join(multi_passports, on="person_id", how="inner")
        .select(
            "person_id", "flight_code", "date",
            pl.concat_str([
                pl.lit("Использование нескольких паспортов (всего: "),
                pl.col("pass_count").cast(pl.String),
                pl.lit(")")
            ]).alias("reason"),
            pl.lit(3).alias("severity")
        )
    )
    anomalies_list.append(passport_anomalies)

    unique_flights_for_teleport = (
        df_conf.filter(
            pl.col("person_id").is_not_null() &
            pl.col("dep_airport").is_not_null() &
            pl.col("arr_airport").is_not_null() &
            pl.col("dep_ts").is_not_null() &
            pl.col("arr_ts").is_not_null()
        )
        .with_columns(
            dep_ts=pl.col("dep_ts").cast(pl.Datetime("us")),
            arr_ts=pl.col("arr_ts").cast(pl.Datetime("us"))
        )
        .unique(subset=["person_id", "flight_code", "date", "dep_ts", "arr_ts"])
        .sort(["person_id", "dep_ts"])
    )

    teleport_anomalies = (
        unique_flights_for_teleport
        .with_columns(
            prev_arr_ts=pl.col("arr_ts").shift(1).over("person_id"),
            prev_arr_airport=pl.col("arr_airport").shift(1).over("person_id")
        )
        .filter(
            pl.col("prev_arr_ts").is_not_null() &
            (pl.col("prev_arr_airport") != pl.col("dep_airport")) &
            (pl.col("dep_ts") < pl.col("prev_arr_ts") + pl.duration(minutes=30))
        )
        .select(
            "person_id", "flight_code", "date",
            pl.lit("Физически невозможное перемещение (телепортация / overlap рейсов)").alias("reason"),
            pl.lit(3).alias("severity")
        )
    )

    print(f"Найдено телепортаций: {len(teleport_anomalies)}")
    anomalies_list.append(teleport_anomalies)
    raw_anomalies = pl.concat(anomalies_list, how="diagonal").unique()

    aggregated_reasons = (
        raw_anomalies
        .group_by(["person_id", "flight_code", "date"])
        .agg(
            pl.col("reason").unique().str.join(" | ").alias("all_suspicion_reasons"),
            pl.col("reason").n_unique().alias("total_reasons_count"),
            pl.col("severity").max().alias("max_severity")
        )
    )

    person_profile = (
        df_conf.filter(pl.col("person_id").is_not_null())
        .group_by("person_id")
        .agg(
            pl.col("name_key").drop_nulls().unique().str.join(" | ").alias("name_key"),
            pl.col("passport").drop_nulls().unique().str.join(" | ").alias("passport")
        )
    )

    final_report = (
        aggregated_reasons
        .join(person_profile, on="person_id", how="left")
        .join(
            df_conf.select([
                "person_id", "flight_code", "date",
                "card_program", "card_number", "dep_airport", "arr_airport", "ticket"
            ]).unique(subset=["person_id", "flight_code", "date"]),
            on=["person_id", "flight_code", "date"],
            how="left"
        )
        .sort(["total_reasons_count", "max_severity", "person_id"], descending=[True, True, False])
    )

    parquet_path = f"{output_prefix}.parquet"
    final_report.write_parquet(parquet_path)
    print(f" Успешно сохранено в Parquet: {parquet_path}")

    try:
        final_report.write_excel(f"{output_prefix}.xlsx")
        print(f" Успешно сохранено в Excel: {output_prefix}.xlsx")
    except Exception:
        pass

    return final_report