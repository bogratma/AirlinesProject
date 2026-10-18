import polars as pl

# Импорт парсеров
from parsers.parse_json import parse_json
from parsers.parse_tab import parse_tab
from parsers.parse_yaml import parse_yaml
from parsers.parse_xml import parse_xml
from parsers.parse_csv import parse_csv
from parsers.parse_boarding_passes import parse_boardingPass_zip
from parsers.build_unified import build_unified_file
from parsers.merge import assign_person_id, merge_events

json_path = r'C:\Users\Vladislav\Desktop\Airlines\FrequentFlyerForum-Profiles.json'
parquet_json_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_json.parquet'

tab_path = r'C:\Users\Vladislav\Desktop\Airlines\Sirena-export-fixed.tab'
parquet_tab_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_tab.parquet'

yaml_path = r'C:\Users\Vladislav\Desktop\Airlines\SkyTeam-Exchange.yaml'
parquet_yaml_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_yaml.parquet'

xml_path = r'C:\Users\Vladislav\Desktop\Airlines\PointzAggregator-AirlinesData.xml'
parquet_xml_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_xml.parquet'

csv_path = r'C:\Users\Vladislav\Desktop\Airlines\BoardingData.csv'
parquet_csv_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_csv.parquet'

pdf_path = r'C:\Users\Vladislav\Desktop\Airlines\Skyteam_Timetable.pdf'
parquet_pdf_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_pdf.parquet'

zip_path = r'C:\Users\Vladislav\Desktop\Airlines\YourBoardingPassDotAero.zip'
parquet_zip_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_zip.parquet'

def main():
    print("=== ЭТАП 1: Парсинг сырых файлов в промежуточные Parquet ==pec===")
    parse_json(json_path, parquet_json_path)
    parse_tab(tab_path, parquet_tab_path)
    parse_yaml(yaml_path, parquet_yaml_path)
    parse_xml(xml_path, parquet_xml_path)
    parse_csv(csv_path, parquet_csv_path)
    # parse_pdf(pdf_path, parquet_pdf_path)  # Раскомментируй, когда понадобится
    parse_boardingPass_zip(zip_path, parquet_zip_path)
    print("\n=== ЭТАП 2: Унификация схем и сборка сырой общей базы ===")
    unified_raw_path = build_unified_file(
        process_dir="process",
        out_name="unified_raw.parquet"
    )

    print("\n=== ЭТАП 3: Entity Resolution (Назначение сквозных ID пассажиров) ===")
    df_raw = pl.read_parquet(unified_raw_path)
    df_with_ids = assign_person_id(df_raw)
    print("\n=== ЭТАП 4: Слияние событий (Merge Events) в мастер-базу ===")
    final_master = merge_events(df_with_ids)
    final_output_path = r"C:\Users\Vladislav\PycharmProjects\Parser\process\unified_master.parquet"
    final_master.write_parquet(final_output_path)
    print(f"\nГотово! Финальная мастер-база сохранена: {final_output_path}")
    print(f"Всего строк в unified_master: {len(final_master)}")

    print("\n=== ДЕТЕКТИВНОЕ РАССЛЕДОВАНИЕ КОНКРЕТНОГО КОНФЛИКТА ===")

    conflicted_row = final_master.filter(
        pl.col("arr_conflict") | pl.col("dep_conflict")
    ).head(1)
    if len(conflicted_row) > 0:
        p_id = conflicted_row["person_id"][0]
        f_code = conflicted_row["flight_code"][0]
        f_date = conflicted_row["date"][0]
        print(f"Ищем расхождения для Person ID: {p_id} на рейсе {f_code} ({f_date})")
        raw_clash = df_with_ids.filter(
            (pl.col("person_id") == p_id) &
            (pl.col("flight_code") == f_code) &
            (pl.col("date") == f_date)
        ).select("source", "flight_code", "date", "dep_airport", "arr_airport", "booking_code", "ticket")
        print("\nЧто по этому рейсу сообщали разные источники:")
        with pl.Config(tbl_cols=-1, tbl_width_chars=200):
            print(raw_clash)
    else:
        print("Конфликтов не обнаружено!")
if __name__ == "__main__":
    main()