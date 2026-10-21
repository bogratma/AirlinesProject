import os

import polars as pl

from parsers.parse_json import parse_json
from parsers.parse_pdf import parse_pdf
from parsers.parse_tab import parse_tab
from parsers.parse_yaml import parse_yaml
from parsers.parse_xml import parse_xml
from parsers.parse_csv import parse_csv
from parsers.parse_boarding_passes import parse_boardingPass_zip

from parsers.build_unified import build_unified_file, pdf_schedule_path
from parsers.merge import merge_events
from parsers import spy_detect as sd

RAW_DIR = r"C:\Users\Айдын\Универ\DS\Airlines-All\Airlines"
PROCESS_DIR = r"C:\Users\Айдын\PycharmProjects\AirlinesProject\process"

json_path = rf"{RAW_DIR}\FrequentFlyerForum-Profiles.json"
tab_path = rf"{RAW_DIR}\Sirena-export-fixed.tab"
yaml_path = rf"{RAW_DIR}\SkyTeam-Exchange.yaml"
xml_path = rf"{RAW_DIR}\PointzAggregator-AirlinesData.xml"
csv_path = rf"{RAW_DIR}\BoardingData.csv"
pdf_path = rf"{RAW_DIR}\Skyteam_Timetable.pdf"
zip_path = rf"{RAW_DIR}\YourBoardingPassDotAero.zip"

parquet_json_path = rf"{PROCESS_DIR}\clean_json.parquet"
parquet_tab_path = rf"{PROCESS_DIR}\clean_tab.parquet"
parquet_yaml_path = rf"{PROCESS_DIR}\clean_yaml.parquet"
parquet_xml_path = rf"{PROCESS_DIR}\clean_xml.parquet"
parquet_csv_path = rf"{PROCESS_DIR}\clean_csv.parquet"
parquet_pdf_path = rf"{PROCESS_DIR}\clean_pdf.parquet"
parquet_zip_path = rf"{PROCESS_DIR}\clean_zip.parquet"

RUN_PARSERS = False


def run_parsers():
    print("=== ЭТАП 1: Парсинг сырых файлов в Parquet ===")
    parse_json(json_path, parquet_json_path)
    parse_tab(tab_path, parquet_tab_path)
    parse_yaml(yaml_path, parquet_yaml_path)
    parse_xml(xml_path, parquet_xml_path)
    parse_csv(csv_path, parquet_csv_path)
    parse_pdf(pdf_path, parquet_pdf_path)
    parse_boardingPass_zip(zip_path, parquet_zip_path)


def dossier(df_with_ids, person_id):
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=300):
        print(df_with_ids.filter(pl.col("person_id") == person_id)
              .sort("date", "dep_ts")
              .select("source", "flight_code", "date", "dep_airport", "arr_airport", "dep_ts",
                      "name_key", "birth_date", "passport", "ticket", "card_key", "nickname"))


def main():
    if RUN_PARSERS:
        run_parsers()

    print("\n=== ЭТАП 2: Унификация схем и сборка общей базы ===")
    unified_raw_path = build_unified_file(process_dir=PROCESS_DIR, out_name="unified_raw.parquet")

    print("\n=== ЭТАП 3: Достройка рейсов, сверка с PDF, person_id ===")
    df = pl.read_parquet(unified_raw_path)
    df = sd.fill_flight_reference(df)
    pdf_parquet = pdf_schedule_path(PROCESS_DIR)
    if pdf_parquet:
        df = sd.enrich_and_validate_with_pdf(df, pdf_parquet)
        print(df.group_by("pdf_status").len().sort("len", descending=True))
    else:
        print("PDF-расписание не найдено, сверка пропущена")
    df_with_ids = sd.assign_person_id(df)
    df_with_ids.write_parquet(rf"{PROCESS_DIR}\unified_with_ids.parquet")

    print("\n=== ЭТАП 4: Слияние событий (merge_events) ===")
    final_master = merge_events(df_with_ids)
    final_master.write_parquet(rf"{PROCESS_DIR}\unified_master.parquet")
    print(f"Всего событий в unified_master: {len(final_master)}")

    print("\n=== ЭТАП 5: Поиск подозрительных ===")
    try:
        info = sd.load_airport_info()
    except ImportError:
        info = None
        print("airportsdata не установлен: проверки перемещений без часовых поясов")
    events = sd.export_suspicious_passengers(
        df_with_ids, output_prefix=rf"{PROCESS_DIR}\suspicious_spies", airport_info=info)

    persons = pl.read_parquet(rf"{PROCESS_DIR}\suspicious_spies_persons.parquet")
    print("\nТоп подозреваемых:")
    with pl.Config(tbl_rows=25, tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=80):
        print(persons.select("person_id", "score", "n_kinds", "kinds", "names", "passports").head(25))
    if len(persons):
        print("\nДосье первого в списке:")
        dossier(df_with_ids, persons["person_id"][0])


if __name__ == "__main__":
    main()