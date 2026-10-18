import csv
import re
from collections import Counter
from datetime import date

from pathlib import Path

import polars as pl

from parse_pdf import OUT_DIR, SCHEDULE_NAME, UNMATCHED_NAME

OUT = Path(OUT_DIR)
df = pl.read_parquet(OUT / SCHEDULE_NAME)
print("строк после дедупликации:", df.height)

with open(OUT / UNMATCHED_NAME, encoding="utf-8") as f:
    unmatched = list(csv.DictReader(f))
bad = [u for u in unmatched if re.search(r"\d\d:\d\d", u["line"])]
print(f"нераспознанных строк: {len(unmatched)}, из них похожих на расписание: {len(bad)}")
for u in bad[:20]:
    print("  ", u)
for pat, n in Counter(re.sub(r"\d", "#", u["line"]) for u in unmatched).most_common(15):
    print("  ", n, pat)

t = r"^([01]\d|2[0-3]):[0-5]\d$"
checks = {
    "нет маршрута": df.filter(pl.col("orig_iata").is_null() | pl.col("dest_iata").is_null()),
    "origin == dest": df.filter(pl.col("orig_iata") == pl.col("dest_iata")),
    "valid_from > valid_to": df.filter(pl.col("valid_from") > pl.col("valid_to")),
    "период вне ноя'18-янв'19": df.filter((pl.col("valid_from") < date(2018, 11, 1))
                                          | (pl.col("valid_to") > date(2019, 1, 31))),
    "days не из цифр 1-7": df.filter(~pl.col("days").str.contains(r"^[1-7]+$")),
    "формат времени": df.filter(~pl.col("dep_time").str.contains(t)
                                | ~pl.col("arr_time").str.contains(t)),
    "duration вне [15; 1500] мин": df.filter(~pl.col("duration_min").is_between(15, 1500)),
    "arr_plus_days вне 0..2": df.filter(~pl.col("arr_plus_days").is_between(0, 2)),
}
for name, bad_df in checks.items():
    print(f"{name:30}{bad_df.height}")

routes = df.select("orig_iata", "dest_iata").unique()
rev = routes.rename({"orig_iata": "dest_iata", "dest_iata": "orig_iata"})
one_way = routes.join(rev, on=["orig_iata", "dest_iata"], how="anti")
print("односторонние маршруты (нужно посмотреть вручную):", one_way.height)
print(one_way)

def one(**kw):
    q = df
    for k, v in kw.items():
        q = q.filter(pl.col(k) == v)
    return q

if one(orig_iata="AAL", dest_iata="AMS", flight="KL1328").height:
    r = one(orig_iata="AAL", dest_iata="AMS", flight="KL1328").row(0, named=True)
    assert (r["dep_time"], r["arr_time"], r["aircraft"], r["duration_min"], r["days"]) == \
           ("06:00", "07:25", "73W", 85, "1234567")
    r = one(orig_iata="AES", dest_iata="AMS", flight="KL1326", dep_time="16:50").row(0, named=True)
    assert r["days"] == "13457" and r["duration_min"] == 125
    r = one(orig_iata="DMM", dest_iata="AHB", flight="SV1683", dep_time="22:45").row(0, named=True)
    assert r["arr_time"] == "01:00" and r["arr_plus_days"] == 1 and r["days"] == "12345"
    assert one(orig_iata="ABZ", dest_iata="CDG", flight="AF1473")["operated_by"].unique().to_list() == ["HOP!"]
    assert one(orig_iata="ABZ", dest_iata="CDG", flight="KQ3747")["operated_by"].unique().to_list() == ["Air France"]
    assert one(orig_iata="AAL", dest_iata="GRZ").height == 0     # "Consult your travel agent"
    print("эталонные строки: OK")