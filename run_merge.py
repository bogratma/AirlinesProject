import polars as pl
from parsers.merge import assign_person_id, merge_events

UNIFIED = r"C:\Users\Vladislav\PycharmProjects\Parser\process\unified.parquet"
MERGED = r"C:\Users\Vladislav\PycharmProjects\Parser\process\merged.parquet"

u = pl.read_parquet(UNIFIED)
merged = merge_events(assign_person_id(u))
merged.write_parquet(MERGED)

print(f"unified: {len(u)} строк -> merged: {len(merged)} строк")
print(f"людей (person_id): {merged['person_id'].n_unique()}")
print("строк, подтверждённых 2+ источниками:", merged.filter(pl.col("n_sources") >= 2).height)

# самые "большие" люди: слишком большое число рейсов может означать ошибочную склейку двух людей
with pl.Config(tbl_rows=10, tbl_cols=-1, tbl_width_chars=250):
    print(merged.group_by("person_id").len().sort("len", descending=True).head(10))