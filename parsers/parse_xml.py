import datetime as dt
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq

INPUT_PATH = r"C:\Users\Айдын\Универ\DS\Airlines-All\Airlines\PointzAggregator-AirlinesData.xml"
OUTPUT_PATH = r"C:\Users\Айдын\PycharmProjects\AirlinesProject\process\xml_data.parquet"
BATCH_SIZE = 100_000

SCHEMA = pa.schema([
    ("uid", pa.string()),
    ("first_name", pa.string()),
    ("last_name", pa.string()),
    ("card_number", pa.string()),
    ("bonus_program", pa.string()),
    ("flight_code", pa.string()),
    ("flight_date", pa.date32()),
    ("departure", pa.string()),
    ("arrival", pa.string()),
    ("fare", pa.string()),
])


def text(elem, tag):
    """Достаёт текст дочернего тега."""
    child = elem.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def parse_date(value):
    try:
        return dt.date.fromisoformat(value) if value else None
    except ValueError:
        return None


def user_to_rows(user):
    name = user.find("name")
    first = name.get("first") if name is not None else None
    last = name.get("last") if name is not None else None
    uid = user.get("uid")

    for card in user.findall("./cards/card"):
        card_number = card.get("number")
        program = text(card, "bonusprogramm")

        for act in card.findall("./activities/activity"):
            yield {
                "uid": uid,
                "first_name": first,
                "last_name": last,
                "card_number": card_number,
                "bonus_program": program,
                "flight_code": text(act, "Code"),
                "flight_date": parse_date(text(act, "Date")),
                "departure": text(act, "Departure"),
                "arrival": text(act, "Arrival"),
                "fare": text(act, "Fare"),
            }


def main():
    rows = []
    total = 0
    writer = pq.ParquetWriter(OUTPUT_PATH, SCHEMA, compression="snappy")

    def flush():
        nonlocal rows, total
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            total += len(rows)
            rows = []

    try:
        context = ET.iterparse(INPUT_PATH, events=("start", "end"))
        _, root = next(context)  # корневой элемент, чтобы чистить память

        for event, elem in context:
            if event == "end" and elem.tag == "user":
                rows.extend(user_to_rows(elem))
                root.clear()  # освобождаем уже обработанных пользователей
                if len(rows) >= BATCH_SIZE:
                    flush()
        flush()
    finally:
        writer.close()

    print(f"Готово: записано {total} строк в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()