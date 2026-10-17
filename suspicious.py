import polars as pl

MERGED = "process/merged.parquet"
JSON_PARQUET = "process/clean_json.parquet"   # нужен только для справочника аэропортов (страны); можно None


def airports_from_json(path):
    j = pl.read_parquet(path)
    return (pl.concat([
        j.select(pl.col("departure_airport").alias("airport"), pl.col("departure_country").alias("country")),
        j.select(pl.col("arrival_airport").alias("airport"), pl.col("arrival_country").alias("country")),
    ]).drop_nulls("airport").unique(subset="airport", keep="first"))


def person_features(m, airports=None):
    """Одна строка на человека: насколько часто летает и что в его поездках необычного."""
    m = (m.sort("person_id", "date", "dep_ts", nulls_last=True)
           .with_columns(
               prev_arr=pl.col("arr_airport").shift(1).over("person_id"),
               route=pl.concat_str([pl.col("dep_airport"), pl.lit("->"), pl.col("arr_airport")]),
               json_only=pl.col("sources").list.join(",") == "json",
           ))
    if airports is not None:
        m = m.join(airports.select(pl.col("airport").alias("arr_airport"),
                                   pl.col("country").alias("arr_country")),
                   on="arr_airport", how="left")
    else:
        m = m.with_columns(arr_country=pl.lit(None, dtype=pl.String))

    feats = m.group_by("person_id").agg(
        pl.col("name_key").drop_nulls().first().alias("name"),
        pl.col("nickname").drop_nulls().first().alias("nickname"),
        pl.col("card_program").drop_nulls().first().alias("card_program"),
        pl.col("card_number").drop_nulls().first().alias("card_number"),
        pl.len().alias("n_flights"),
        pl.col("date").n_unique().alias("n_days"),
        pl.col("arr_airport").n_unique().alias("n_destinations"),
        pl.col("route").n_unique().alias("n_routes"),
        pl.col("arr_country").drop_nulls().n_unique().alias("n_countries"),
        pl.col("date").min().alias("first_date"),
        pl.col("date").max().alias("last_date"),
        (pl.col("n_sources") >= 2).mean().round(2).alias("confirmed_share"),
        pl.col("json_only").sum().alias("json_only"),
        (pl.col("prev_arr").is_not_null() & (pl.col("dep_airport") != pl.col("prev_arr")))
            .sum().alias("route_gaps"),
        (pl.col("dep_conflict") | pl.col("arr_conflict")).sum().alias("conflicts"),
    )

    # регулярность: доля самого частого маршрута
    top_route = (m.group_by("person_id", "route").len()
                  .group_by("person_id").agg((pl.col("len").max() / pl.col("len").sum()).round(2).alias("top_route_share")))
    return feats.join(top_route, on="person_id", how="left")


def attention_list(feats, top_share=0.05):
    """Частые летчики (верхние 5% по числу рейсов), у которых есть хотя бы один тревожный признак."""
    threshold = feats["n_flights"].quantile(1 - top_share)
    return (feats.filter(pl.col("n_flights") >= threshold)
                 .filter((pl.col("route_gaps") > 0) | (pl.col("json_only") > 0) | (pl.col("conflicts") > 0))
                 .sort("n_flights", descending=True))


def timeline(m, person_id):
    """Хронология одного человека: куда и когда летал и откуда подтверждено."""
    return (m.filter(pl.col("person_id") == person_id)
             .sort("date", "dep_ts", nulls_last=True)
             .select("date", "flight_code", "dep_airport", "arr_airport", "sources"))


if __name__ == "__main__":
    pl.Config.set_tbl_rows(25)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)

    m = pl.read_parquet(MERGED)
    airports = airports_from_json(JSON_PARQUET) if JSON_PARQUET else None
    feats = person_features(m, airports)

    print("=== Самые частые летчики ===");              print(feats.sort("n_flights", descending=True).head(15))
    print("\n=== Больше всего стран ===");                print(feats.sort("n_countries", descending=True).head(15))
    print("\n=== Самый регулярный маршрут ===");          print(feats.filter(pl.col("n_flights") >= 5).sort("top_route_share", descending=True).head(15))
    print("\n=== Рейсы только по словам форума ===");     print(feats.sort("json_only", descending=True).head(15))
    print("\n=== Разрывы маршрута ===");                  print(feats.sort("route_gaps", descending=True).head(15))
    print("\n=== Список на внимание ===");                print(attention_list(feats))