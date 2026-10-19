import re

import polars as pl
from unidecode import unidecode
import pandas as pd

SCHEMA = {
    "source": pl.String,
    "flight_code": pl.String, "date": pl.String,
    "dep_airport": pl.String, "arr_airport": pl.String,
    "dep_ts": pl.Datetime("us"), "arr_ts": pl.Datetime("us"),
    "name_key": pl.String, "birth_date": pl.String,
    "passport": pl.String, "ticket": pl.String, "booking_code": pl.String,
    "card_program": pl.String, "card_number": pl.String,
    "nickname": pl.String,
    "travel_class": pl.String, "fare": pl.String, "agent": pl.String,
}


def _has(df, name):
    return name in df.columns


def _c(df, name):
    if not _has(df, name):
        return pl.lit(None, dtype=pl.String)
    return pl.col(name).cast(pl.String)


def _norm(e):
    e = e.cast(pl.String).str.strip_chars()
    return pl.when(e.is_in(["", "N/A", "Not presented"])).then(None).otherwise(e)


def _id(e):
    return _norm(_norm(e).str.replace_all(r"[^0-9A-Za-zА-Яа-я]", ""))


def _ticket(e):
    return _norm(_norm(e).str.replace_all(r"\D", "").str.strip_chars_start("0"))


def _flight(e):
    s = _norm(e).str.to_uppercase().str.replace_all(r"[\s\-]", "")
    air = s.str.extract(r"^([A-Z0-9]{2})\d", 1)
    num = s.str.extract(r"^[A-Z0-9]{2}(\d+)", 1).str.strip_chars_start("0")
    return pl.when(air.is_not_null() & (num != "")).then(pl.concat_str([air, num]))


def _iata(e):
    return _norm(e).str.to_uppercase()


def _date(df, name):
    if not _has(df, name):
        return pl.lit(None, dtype=pl.String)
    dt = df.schema[name]
    col = pl.col(name)
    if dt == pl.Date or isinstance(dt, pl.Datetime):
        return col.dt.to_string("%Y-%m-%d")
    e = _norm(col)
    iso = e.str.extract(r"^(\d{4}-\d{2}-\d{2})", 1)
    dmy = e.str.to_date("%d.%m.%Y", strict=False).dt.to_string("%Y-%m-%d")
    return pl.coalesce(iso, dmy)


def _ts_from_parts(df, date_name, time_name):
    if not (_has(df, date_name) and _has(df, time_name)):
        return pl.lit(None, dtype=pl.Datetime("us"))
    d = _date(df, date_name)
    t = _norm(pl.col(time_name).cast(pl.String))
    return pl.concat_str([d, t], separator=" ").str.to_datetime("%Y-%m-%d %H:%M", strict=False)


def _ts_col(df, name):
    if not _has(df, name):
        return pl.lit(None, dtype=pl.Datetime("us"))
    return pl.col(name).cast(pl.Datetime("us"))


def canon_name(raw):
        if not raw:
            return None
        s = unidecode(raw).upper()
        s = re.sub(r"[’'`\-]", "", s)

        # --- УНИВЕРСАЛЬНАЯ НОРМАЛИЗАЦИЯ ДЛЯ ЛЮБЫХ ИМЕН ---
        # 1. Гласные и дифтонги (разница систем ICAO / ГОСТ)
        s = s.replace("IU", "YU")  # Liudmila -> Lyudmila, Iurii -> Yuriy
        s = s.replace("IA", "YA")  # Ksenia -> Kseniya, Maria -> Mariya
        s = s.replace("YE", "E")  # Evgeny -> Evgeni (по желанию, но осторожно)

        # 2. Шипящие и сложные согласные (Ш, Щ, Ж, Х, Ц)
        s = s.replace("SCH", "SH")  # Scherbatoff -> Sherbatoff
        s = s.replace("KH", "H")  # Khokhlov -> Hohlov (или наоборот)
        s = s.replace("CZ", "TS")  # Cvetkova -> Tsvetkova
        s = s.replace("ZH", "Z")  # Zhukov -> Zukov (иногда сокращают)

        # 3. Окончания (ИЙ / ИИ / ЫЙ)
        s = re.sub(r"II\b", "IY", s)  # Petrovskii -> Petrovskiy
        s = re.sub(r"YY\b", "Y", s)
        s = re.sub(r"Y\b", "I", s)  # Иногда на конце путают Y и I
        # --------------------------------------------------

        toks = [t for t in re.split(r"[^A-Z]+", s) if len(t) > 1]
        return " ".join(sorted(toks)) or None


def _with_name_key(df, raw_expr):
    df = df.with_columns(_norm(raw_expr).alias("_raw_name"))
    uniq = df["_raw_name"].drop_nulls().unique().to_list()
    mapping = {n: canon_name(n) for n in uniq}
    return df.with_columns(
        pl.col("_raw_name").replace_strict(mapping, default=None, return_dtype=pl.String).alias("_name_key"))


def _two_tokens(e):
    return _norm(e).str.replace_all(r"\s+", " ").str.extract(r"^(\S+(?: \S+)?)", 1)


def _finish(df, source, mapping):
    return df.select([
        pl.lit(source).alias("source") if c == "source"
        else (mapping[c] if c in mapping else pl.lit(None, dtype=t)).cast(t).alias(c)
        for c, t in SCHEMA.items()
    ])


def unify_tab(df):
    df = _with_name_key(df, _two_tokens(_c(df, "PaxName")))
    return _finish(df, "tab", {
        "flight_code": _flight(_c(df, "FlightCodeSh")),
        "date": _date(df, "DepartDate"),
        "dep_airport": _iata(_c(df, "From")), "arr_airport": _iata(_c(df, "Dest")),
        "dep_ts": _ts_from_parts(df, "DepartDate", "DepartTime"),
        "arr_ts": _ts_from_parts(df, "ArrivalDate", "ArrivalTime"),
        "name_key": pl.col("_name_key"),
        "birth_date": _date(df, "PaxBirthDate"),
        "passport": _id(_c(df, "TravelDoc")),
        "ticket": _ticket(_c(df, "e-Ticket")),
        "booking_code": _norm(_c(df, "Code")),
        "card_program": _norm(_c(df, "loyalty_program")),
        "card_number": _norm(_c(df, "loyalty_number")),
        "travel_class": _norm(_c(df, "TrvCls")), "fare": _norm(_c(df, "Fare")),
        "agent": _norm(_c(df, "AgentInfo")),
    })


def unify_yaml(df):
    return _finish(df, "yaml", {
        "flight_code": _flight(_c(df, "flight")),
        "date": _date(df, "date"),
        "dep_airport": _iata(_c(df, "from")),
        "arr_airport": _iata(_c(df, "to")),
        "card_program": _norm(_c(df, "loyalty_program")),
        "card_number": _norm(_c(df, "loyalty_number")),
        "travel_class": _norm(_c(df, "class")), "fare": _norm(_c(df, "fare")),
    })


def unify_xml(df, program_codes=None):
    df = _with_name_key(df, pl.concat_str(
        [_c(df, "last_name").fill_null(""), pl.lit(" "), _c(df, "first_name").fill_null("")]))
    prog = _norm(_c(df, "bonus_program"))
    if program_codes:
        prog = prog.replace_strict(program_codes, default=prog, return_dtype=pl.String)
    return _finish(df, "xml", {
        "flight_code": _flight(_c(df, "flight_code")),
        "date": _date(df, "flight_date"),
        "dep_airport": _iata(_c(df, "departure")), "arr_airport": _iata(_c(df, "arrival")),
        "name_key": pl.col("_name_key"),
        "card_program": prog, "card_number": _norm(_c(df, "card_number")),
        "fare": _norm(_c(df, "fare")),
    })


def unify_csv(df, city_map=None):
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    df = _with_name_key(df, pl.concat_str(
        [_c(df, "last_name").fill_null(""), pl.lit(" "), _c(df, "first_name").fill_null("")]))
    def _iata_dyn(e):
        norm_e = _norm(e).str.to_uppercase()
        if city_map:
            return norm_e.replace_strict(city_map, default=norm_e, return_dtype=pl.String)
        return norm_e
    dep_col = _iata_dyn(_c(df, "departure")) if _has(df, "departure") else _iata_dyn(_c(df, "origin"))
    return _finish(df, "csv", {
        "flight_code": _flight(_c(df, "flight_number")),
        "date": _date(df, "flight_date"),
        "dep_airport": pl.lit(None, dtype=pl.String),
        "arr_airport": pl.lit(None, dtype=pl.String),
        "dep_ts": _ts_col(df, "flight_datetime"),
        "name_key": pl.col("_name_key"),
        "birth_date": _date(df, "birth_date"),
        "passport": _id(_c(df, "document")),
        "ticket": _ticket(_c(df, "ticket_number")),
        "booking_code": _norm(_c(df, "booking_code")),
    })


def unify_boarding(df):
    df = _with_name_key(df, _c(df, "passenger_name_raw"))
    ff = _norm(_c(df, "ff_raw")).str.to_uppercase()
    prog = ff.str.extract(r"^([A-Z0-9]{2})[\s\-]*\d+$", 1)
    num = ff.str.extract(r"^[A-Z0-9]{2}[\s\-]*(\d+)$", 1)
    return _finish(df, "boarding", {
        "flight_code": _flight(_c(df, "flight_no")),
        "date": _date(df, "flight_date"),
        "dep_airport": _iata(_c(df, "from_iata")), "arr_airport": _iata(_c(df, "to_iata")),
        "dep_ts": _ts_col(df, "departure_dt"),
        "name_key": pl.col("_name_key"),
        "ticket": _ticket(_c(df, "ticket_no")),
        "booking_code": _norm(_c(df, "pnr")),
        "card_program": prog, "card_number": num,
        "travel_class": _norm(_c(df, "travel_class")),
    })


def unify_json(df):
    df = _with_name_key(df, pl.concat_str(
        [_c(df, "last_name").fill_null(""), pl.lit(" "), _c(df, "first_name").fill_null("")]))
    cards = _norm(_c(df, "loyalty_programs")).str.split(" | ").alias("_cards")
    passports = _norm(_c(df, "passports")).str.split(", ").alias("_pass")
    base = df.with_columns(cards, passports).select(
        _flight(_c(df, "flight_id")).alias("flight_code"),
        _date(df, "date").alias("date"),
        _iata(_c(df, "departure_airport")).alias("dep_airport"),
        _iata(_c(df, "arrival_airport")).alias("arr_airport"),
        pl.col("_name_key").alias("name_key"),
        _norm(_c(df, "nickname")).alias("nickname"),
        pl.col("_cards"), pl.col("_pass"),
    )

    def fin(d, extra):
        m = {c: pl.col(c) for c in ["flight_code", "date", "dep_airport", "arr_airport", "name_key", "nickname"]}
        m.update(extra)
        return _finish(d, "json", m)

    by_card = base.explode("_cards").filter(pl.col("_cards").is_not_null())
    by_card = fin(by_card, {
        "card_program": pl.col("_cards").str.extract(r"^([A-Za-z0-9]+)[:\s]", 1),
        "card_number": pl.col("_cards").str.extract(r"(\d+)\s*$", 1),
    })
    by_pass = base.explode("_pass").filter(pl.col("_pass").is_not_null())
    by_pass = fin(by_pass, {"passport": _id(pl.col("_pass"))})
    no_ids = fin(base.filter(pl.col("_cards").is_null() & pl.col("_pass").is_null()), {})
    return pl.concat([by_card, by_pass, no_ids])


def airports_from_schedule(path):
    s = pl.read_parquet(path)
    return (pl.concat([
        s.select(pl.col("orig_iata").alias("airport"), pl.col("orig_city").alias("city"),
                 pl.col("orig_country").alias("country")),
        s.select(pl.col("dest_iata").alias("airport"), pl.col("dest_city").alias("city"),
                 pl.col("dest_country").alias("country")),
    ]).drop_nulls("airport").unique(subset="airport", keep="first"))
