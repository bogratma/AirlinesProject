import polars as pl
from unidecode import unidecode

SCHEMA = {
    "source": pl.String,
    "flight_code": pl.String,
    "date": pl.String,
    "dep_airport": pl.String,
    "arr_airport": pl.String,
    "dep_ts": pl.Datetime,
    "arr_ts": pl.Datetime,
    "name_key": pl.String,
    "birth_date": pl.String,
    "passport": pl.String,
    "card_program": pl.String,
    "card_number": pl.String,
    "nickname": pl.String,
    "booking_code": pl.String,
    "ticket": pl.String,
    "agent": pl.String,
}


def _c(df, names):
  """Ищет первую попавшуюся колонку из списка возможных вариантов названия"""
  if isinstance(names, str):
    names = [names]
  for name in names:
    if name in df.columns:
      return pl.col(name)
  return pl.lit(None, dtype=pl.String)


def _norm(e):
  e = e.cast(pl.String).str.strip_chars()
  return pl.when(e.is_in(["", "N/A", "None", "null"])).then(None).otherwise(e)


def _id(e):
  return _norm(_norm(e).str.replace_all(r"[^0-9A-Za-zА-Яа-я]", ""))


def _flight(e):
  return _norm(e).str.to_uppercase().str.extract(r"^([A-Z0-9]{2}\d+)", 1)


def _date(e):
  e = _norm(e)
  d = pl.coalesce(
      e.str.to_date("%Y-%m-%d", strict=False),
      e.str.to_date("%d.%m.%Y", strict=False),
      e.str.to_date("%d/%m/%Y", strict=False),
  )
  return d.dt.to_string("%Y-%m-%d")


def _ts(date_e, time_e):
  return pl.concat_str([_norm(date_e), _norm(time_e)], separator=" ").str.to_datetime(
      "%Y-%m-%d %H:%M", strict=False
  )


def _name_key(e):
  """Нормализует ФИО: переводит в латиницу (транслит) или приводит к верхнему регистру,

  чтобы русские и английские имена матчились между собой.
  """
  # Очищаем и переводим в верхний регистр
  cleaned = _norm(e).str.to_uppercase().str.replace_all(r"\s+", " ")
  # Оставляем Фамилию и Имя (или все слова, если нужно)
  return cleaned


def _finish(df, source, mapping):
  return df.select([
      pl.lit(source).alias("source")
      if c == "source"
      else (mapping[c] if c in mapping else pl.lit(None, dtype=t)).alias(c)
      for c, t in SCHEMA.items()
  ])


def unify_tab(df):
  return _finish(
      df,
      "tab",
      {
          "flight_code": _flight(_c(df, ["FlightCodeSh", "flight_code", "flight"])),
          "date": _date(_c(df, ["DepartDate", "date"])),
          "dep_airport": _norm(_c(df, ["From", "dep_airport", "from"])),
          "arr_airport": _norm(_c(df, ["Dest", "arr_airport", "to"])),
          "dep_ts": _ts(
              _c(df, ["DepartDate", "date"]), _c(df, ["DepartTime", "dep_time"])
          ),
          "arr_ts": _ts(
              _c(df, ["ArrivalDate", "date"]),
              _c(df, ["ArrivalTime", "arr_time"]),
          ),
          "name_key": _name_key(
              _c(df, ["PaxName", "name_key", "passenger_name"])
          ),
          "birth_date": _date(_c(df, ["PaxBirthDate", "birth_date"])),
          "passport": _id(_c(df, ["TravelDoc", "passport", "passports"])),
          "card_program": _norm(_c(df, ["loyalty_program", "card_program"])),
          "card_number": _norm(_c(df, ["loyalty_number", "card_number"])),
          "booking_code": _norm(_c(df, ["Code", "booking_code"])),
          "ticket": _norm(_c(df, ["e-Ticket", "ticket"])),
          "agent": _norm(_c(df, ["AgentInfo", "agent"])),
      },
  )


def unify_yaml(df):
  return _finish(
      df,
      "yaml",
      {
          "flight_code": _flight(_c(df, ["flight_code", "flight"])),
          "date": _date(_c(df, ["date", "DepartDate"])),
          "dep_airport": _norm(_c(df, ["from", "dep_airport", "From"])),
          "arr_airport": _norm(_c(df, ["to", "arr_airport", "Dest"])),
          "card_program": _norm(_c(df, ["loyalty_program", "card_program"])),
          "card_number": _norm(_c(df, ["loyalty_number", "card_number"])),
          "name_key": _name_key(_c(df, ["name_key", "passenger_name"])),
      },
  )


def unify_json(df):
  cards = _norm(_c(df, "loyalty_programs")).str.split(" | ").alias("_cards")
  passports = _norm(_c(df, "passports")).str.split(", ").alias("_pass")

  base = df.with_columns(cards, passports).select(
      _flight(_c(df, "flight_id")).alias("flight_code"),
      _date(_c(df, "date")).alias("date"),
      _norm(_c(df, "departure_airport")).alias("dep_airport"),
      _norm(_c(df, "arrival_airport")).alias("arr_airport"),
      _name_key(
          pl.concat_str(
              [_c(df, "last_name"), _c(df, "first_name")], separator=" "
          )
      ).alias("name_key"),
      _norm(_c(df, "nickname")).alias("nickname"),
      pl.col("_cards"),
      pl.col("_pass"),
  )

  def fin(d, extra):
    m = {
        c: pl.col(c)
        for c in [
            "flight_code",
            "date",
            "dep_airport",
            "arr_airport",
            "name_key",
            "nickname",
        ]
    }
    m.update(extra)
    return _finish(d, "json", m)

  by_card = base.explode("_cards").filter(pl.col("_cards").is_not_null())
  by_card = fin(
      by_card,
      {
          "card_program": pl.col("_cards").str.extract(
              r"^([A-Za-z0-9]+)[:\s]", 1
          ),
          "card_number": pl.col("_cards").str.extract(r"(\d+)\s*$", 1),
      },
  )

  by_pass = base.explode("_pass").filter(pl.col("_pass").is_not_null())
  by_pass = fin(by_pass, {"passport": _id(pl.col("_pass"))})

  no_ids = fin(
      base.filter(pl.col("_cards").is_null() & pl.col("_pass").is_null()), {}
  )

  return pl.concat([by_card, by_pass, no_ids])


def build_unified(df_tab, df_yaml, df_json, output_path=None):
  u = pl.concat([unify_tab(df_tab), unify_yaml(df_yaml), unify_json(df_json)]).unique()
  if output_path:
    u.write_parquet(output_path)
  return u

def build_unified_smart(df_tab, df_yaml, df_json, output_path=None):
  raw_concat = pl.concat(
      [unify_tab(df_tab), unify_yaml(df_yaml), unify_json(df_json)]
  )

  group_keys = ["date", "flight_code", "card_number", "passport"]
  agg_cols = [
      c for c in raw_concat.columns if c not in group_keys and c != "source"
  ]

  agg_expressions = [
      pl.col(c).drop_nulls().first().alias(c) for c in agg_cols
  ]

  # Вот здесь добавляем .implode() перед .list.join()
  agg_expressions.append(
      pl.col("source")
      .drop_nulls()
      .unique()
      .implode()
      .list.join(", ")
      .alias("source")
  )

  u = raw_concat.group_by(group_keys).agg(agg_expressions)

  if output_path:
    u.write_parquet(output_path)
  return u