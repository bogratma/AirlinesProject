import polars as pl

u = pl.read_parquet("process/unified.parquet")   # путь, куда вы сохранили таблицу

card = (u.filter((pl.col("card_program") == "SU") & (pl.col("card_number") == "38116280"))
          .sort("date", "source"))

with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(card)