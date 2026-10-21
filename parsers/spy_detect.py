import math

import polars as pl

SOURCE_PRIORITY = {"pdf": 0, "tab": 1, "csv": 2, "boarding": 3, "yaml": 4, "xml": 5, "json": 6}

PLACEHOLDERS = ["", "N/A", "NA", "NONE", "NULL", "-", "NOTPRESENTED", "UNKNOWN"]
IATA_RE = r"^[A-Z]{3}$"
MAX_GAP_H = 72
MIN_GAP_NO_COORDS_H = 3.0
AIRPORT_ALIASES = {"REA": "ICN"}


def _alias_airports(u):
    return u.with_columns(pl.col("dep_airport").replace(AIRPORT_ALIASES),
                          pl.col("arr_airport").replace(AIRPORT_ALIASES))


def _clean_key(col, remove_spaces=True):
    x = pl.col(col).cast(pl.String).str.strip_chars().str.to_uppercase()
    if remove_spaces:
        x = x.str.replace_all(r"\s+", "")
    junk = (
        x.is_null()
        | x.is_in(PLACEHOLDERS)
        | (x.str.len_chars() < 4)
        | (x.str.replace_all(r"[0\-./]", "") == "")
    )
    return pl.when(junk).then(None).otherwise(x)


def prepare(u):
    return _alias_airports(u.with_columns(passport=_clean_key("passport"), ticket=_clean_key("ticket")))


def add_card_key(u):
    if "card_key" in u.columns:
        return u
    num = pl.col("card_number").cast(pl.String).str.replace_all(r"\s+", "")
    u = u.with_columns(
        _prog=pl.col("card_program").cast(pl.String).str.strip_chars().str.to_uppercase(),
        card_num_n=pl.when(num == "").then(None).otherwise(num),
    )
    fill = (
        u.filter(pl.col("_prog").is_not_null() & pl.col("card_num_n").is_not_null())
        .group_by("card_num_n")
        .agg(pl.col("_prog").n_unique().alias("_n"), pl.col("_prog").first().alias("_prog_fill"))
        .filter(pl.col("_n") == 1)
        .select("card_num_n", "_prog_fill")
    )
    u = (
        u.with_row_index("_r0")
        .join(fill, on="card_num_n", how="left")
        .sort("_r0")
        .with_columns(card_prog_f=pl.coalesce("_prog", "_prog_fill"))
        .with_columns(
            card_key=pl.when(pl.col("card_prog_f").is_not_null() & pl.col("card_num_n").is_not_null())
            .then(pl.concat_str(["card_prog_f", "card_num_n"], separator=" "))
        )
        .drop("_prog", "_prog_fill", "_r0")
    )
    return u


def fill_flight_reference(df):
    if "ts_est" in df.columns:
        return df
    prio = pl.col("source").replace_strict(SOURCE_PRIORITY, default=9, return_dtype=pl.Int64)
    d = df.with_row_index("_r0").with_columns(
        _prio=prio,
        _dep0=pl.col("dep_ts").is_null(), _arr0=pl.col("arr_ts").is_null(),
        _date_dt=pl.col("date").cast(pl.String).str.to_date("%Y-%m-%d", strict=False).cast(pl.Datetime("us")),
    )
    d = _alias_airports(d)
    ok = pl.col("flight_code").is_not_null() & pl.col("date").is_not_null()

    k1 = ["flight_code", "date"]
    ap = (d.filter(ok).group_by(k1).agg(
        pl.col("dep_airport").drop_nulls().n_unique().alias("_nd"),
        pl.col("dep_airport").drop_nulls().first().alias("_dep_r"),
        pl.col("arr_airport").drop_nulls().n_unique().alias("_na"),
        pl.col("arr_airport").drop_nulls().first().alias("_arr_r"))
        .select(*k1,
                pl.when(pl.col("_nd") == 1).then(pl.col("_dep_r")).alias("_dep_r"),
                pl.when(pl.col("_na") == 1).then(pl.col("_arr_r")).alias("_arr_r")))
    d = (d.join(ap, on=k1, how="left")
         .with_columns(dep_airport=pl.coalesce("dep_airport", "_dep_r"),
                       arr_airport=pl.coalesce("arr_airport", "_arr_r"))
         .drop("_dep_r", "_arr_r"))

    k2 = ["flight_code", "date", "dep_airport", "arr_airport"]
    full = ok & pl.col("dep_airport").is_not_null() & pl.col("arr_airport").is_not_null()
    t = (d.filter(full).sort("_prio", maintain_order=True).group_by(k2)
         .agg(pl.col("dep_ts").drop_nulls().first().alias("_dts"),
              pl.col("arr_ts").drop_nulls().first().alias("_ats")))
    d = (d.join(t, on=k2, how="left")
         .with_columns(dep_ts=pl.coalesce("dep_ts", "_dts"), arr_ts=pl.coalesce("arr_ts", "_ats"))
         .drop("_dts", "_ats"))

    k3 = ["flight_code", "dep_airport", "arr_airport"]
    both = (pl.col("flight_code").is_not_null() & pl.col("dep_airport").is_not_null()
            & pl.col("arr_airport").is_not_null() & pl.col("dep_ts").is_not_null() & pl.col("arr_ts").is_not_null())
    typ = d.filter(both).group_by(k3).agg(
        (pl.col("dep_ts") - pl.col("dep_ts").dt.truncate("1d")).mode().first().alias("_tod"),
        (pl.col("arr_ts") - pl.col("dep_ts")).mode().first().alias("_dur"))
    d = (d.join(typ, on=k3, how="left")
         .with_columns(dep_ts=pl.coalesce("dep_ts", pl.col("_date_dt") + pl.col("_tod")))
         .with_columns(arr_ts=pl.coalesce("arr_ts", pl.col("dep_ts") + pl.col("_dur")))
         .with_columns(ts_est=(pl.col("_dep0") & pl.col("dep_ts").is_not_null())
                       | (pl.col("_arr0") & pl.col("arr_ts").is_not_null()))
         .sort("_r0")
         .drop("_r0", "_prio", "_dep0", "_arr0", "_date_dt", "_tod", "_dur"))
    n = d.filter(pl.col("ts_est")).height
    print(f"[fill_flight_reference] достроено времён у {n} записей из {d.height}")
    return d


def _pref(prefix, col):
    return pl.when(pl.col(col).is_not_null()).then(pl.concat_str([pl.lit(prefix), pl.col(col).cast(pl.String)]))


def _components(u, cols):
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in u.select(cols).unique().iter_rows():
        ids = [x for x in row if x is not None]
        for x in ids:
            parent.setdefault(x, x)
        for x in ids[1:]:
            ra, rb = find(ids[0]), find(x)
            if ra != rb:
                parent[rb] = ra

    roots, idents, cids = {}, [], []
    for k in parent:
        idents.append(k)
        cids.append(roots.setdefault(find(k), len(roots)))
    return pl.DataFrame({"ident": idents, "cid": cids}, schema={"ident": pl.String, "cid": pl.Int64})


def assign_person_id(u, verbose=True):
    u = add_card_key(prepare(u)).with_row_index("_row")
    u = u.with_columns(
        _i_pass=_pref("pass:", "passport"),
        _i_tick=_pref("tick:", "ticket"),
        _i_nick=_pref("nick:", "nickname"),
        _i_nb=pl.when(pl.col("name_key").is_not_null() & pl.col("birth_date").is_not_null())
        .then(pl.concat_str([pl.lit("nb:"), pl.col("name_key"), pl.lit("|"),
                             pl.col("birth_date").cast(pl.String)])),
    )

    strong = ["_i_pass", "_i_tick", "_i_nick", "_i_nb"]
    comp = _components(u, strong)
    for c in strong:
        u = u.join(comp.rename({"ident": c, "cid": f"_cid{c}"}), on=c, how="left")
    cid = pl.coalesce([f"_cid{c}" for c in strong])
    u = u.with_columns(
        person_id=pl.when(cid.is_not_null()).then(pl.concat_str([pl.lit("P"), cid.cast(pl.String)]))
    ).drop([f"_cid{c}" for c in strong])

    for key in ("name_key", "card_key", "name_key", "card_key"):
        m = (
            u.filter(pl.col(key).is_not_null() & pl.col("person_id").is_not_null())
            .group_by(key)
            .agg(pl.col("person_id").n_unique().alias("_n"), pl.col("person_id").first().alias("_v"))
            .filter(pl.col("_n") == 1)
            .select(key, "_v")
        )
        eligible = pl.col("name_key").is_null() if key == "card_key" else pl.lit(True)
        u = (
            u.join(m, on=key, how="left")
            .with_columns(
                person_id=pl.when(pl.col("person_id").is_null() & eligible)
                .then(pl.col("_v")).otherwise(pl.col("person_id"))
            )
            .drop("_v")
        )

    u = u.with_columns(_i_card=_pref("card:", "card_key"))
    weak = u.filter(pl.col("person_id").is_null())
    if weak.height:
        cw = _components(weak, ["_i_card", "_i_nick"])
        for c in ("_i_card", "_i_nick"):
            u = u.join(cw.rename({"ident": c, "cid": f"_w{c}"}), on=c, how="left")
        wid = pl.coalesce(["_w_i_card", "_w_i_nick"])
        u = u.with_columns(
            person_id=pl.coalesce(
                pl.col("person_id"),
                pl.when(wid.is_not_null()).then(pl.concat_str([pl.lit("W"), wid.cast(pl.String)])),
            )
        ).drop(["_w_i_card", "_w_i_nick"])
    u = u.with_columns(
        person_id=pl.coalesce(pl.col("person_id"),
                              pl.concat_str([pl.lit("row_"), pl.col("_row").cast(pl.String)]))
    )
    u = u.sort("_row").drop("_row", "_i_pass", "_i_tick", "_i_nb", "_i_card", "_i_nick")

    if verbose:
        _report_persons(u)
    return u


def _report_persons(u):
    t = (u.select(pl.col("person_id").str.slice(0, 1).alias("t"))
         .group_by("t").len().sort("t"))
    print("person_id по типам (P сильные, W слабые, r одиночные записи):")
    print(t)
    top = (u.filter(pl.col("person_id").str.starts_with("P"))
           .group_by("person_id").len().sort("len", descending=True).head(5))
    print("самые крупные P-компоненты (если они огромные, где-то есть общий мусорный ключ):")
    print(top)


def enrich_and_validate_with_pdf(unified_df, pdf_schedule_path):
    old = ["pdf_status", "pdf_legs", "route_mismatch", "pdf_dep", "pdf_arr", "valid_from", "valid_to",
           "days", "aircraft", "operated_by"]
    unified_df = _alias_airports(unified_df.drop([c for c in old if c in unified_df.columns]))
    norm_fc = lambda c: pl.col(c).cast(pl.String).str.replace_all(r"\s+", "").str.to_uppercase()
    norm_ap = lambda c: pl.col(c).cast(pl.String).str.strip_chars().str.to_uppercase()

    sched = pl.read_parquet(pdf_schedule_path).select(
        norm_fc("flight").alias("_fc"),
        norm_ap("orig_iata").alias("_pdf_dep"),
        norm_ap("dest_iata").alias("_pdf_arr"),
    ).drop_nulls(["_fc", "_pdf_dep", "_pdf_arr"]).unique()

    known = sched.select("_fc").unique().with_columns(_known=pl.lit(True))
    pairs = sched.rename({"_pdf_dep": "dep_airport", "_pdf_arr": "arr_airport"}).with_columns(
        _pair_ok=pl.lit(True))
    aps = pl.concat([
        sched.select("_fc", pl.col("_pdf_dep").alias("_ap")),
        sched.select("_fc", pl.col("_pdf_arr").alias("_ap")),
    ]).unique()
    dep_ok = aps.rename({"_ap": "dep_airport"}).with_columns(_dep_ok=pl.lit(True))
    arr_ok = aps.rename({"_ap": "arr_airport"}).with_columns(_arr_ok=pl.lit(True))
    legs = sched.group_by("_fc").agg(
        pl.concat_str(["_pdf_dep", "_pdf_arr"], separator="-").unique().sort().str.join("; ").alias("pdf_legs"))

    u = unified_df.with_row_index("_r0").with_columns(_fc=norm_fc("flight_code"))
    u = (
        u.join(known, on="_fc", how="left")
        .join(pairs, on=["_fc", "dep_airport", "arr_airport"], how="left")
        .join(dep_ok, on=["_fc", "dep_airport"], how="left")
        .join(arr_ok, on=["_fc", "arr_airport"], how="left")
        .join(legs, on="_fc", how="left")
        .sort("_r0")
    )
    dep, arr = pl.col("dep_airport"), pl.col("arr_airport")
    status = (
        pl.when(pl.col("flight_code").is_null()).then(pl.lit("no_data"))
        .when(pl.col("_known").is_null()).then(pl.lit("not_in_pdf"))
        .when(dep.is_not_null() & pl.col("_dep_ok").is_null()).then(pl.lit("airport_mismatch"))
        .when(arr.is_not_null() & pl.col("_arr_ok").is_null()).then(pl.lit("airport_mismatch"))
        .when(dep.is_not_null() & arr.is_not_null() & pl.col("_pair_ok").is_null())
        .then(pl.lit("pair_mismatch"))
        .otherwise(pl.lit("ok"))
    )
    return (
        u.with_columns(pdf_status=status)
        .with_columns(route_mismatch=pl.col("pdf_status").is_in(["airport_mismatch", "pair_mismatch"]))
        .drop("_r0", "_fc", "_known", "_pair_ok", "_dep_ok", "_arr_ok")
    )


def load_airport_info():
    import airportsdata
    rows = [{"airport": k, "lat": v["lat"], "lon": v["lon"], "tz": v["tz"]}
            for k, v in airportsdata.load("IATA").items()]
    return pl.DataFrame(rows)


def _haversine_km(lat1, lon1, lat2, lon2):
    k = math.pi / 180
    a = (((lat2 - lat1) * k / 2).sin() ** 2
         + (lat1 * k).cos() * (lat2 * k).cos() * (((lon2 - lon1) * k / 2).sin() ** 2))
    return 2 * 6371.0 * a.sqrt().arcsin()


def _to_utc(ev, ts, ap, out, info):
    if info is None or "tz" not in info.columns:
        return ev.with_columns(pl.col(ts).alias(out))
    tzs = info.select(pl.col("airport").alias(ap), pl.col("tz").alias("_tz")).unique(ap)
    ev = ev.join(tzs, on=ap, how="left")
    parts = []
    for (tz,), part in ev.partition_by("_tz", as_dict=True, include_key=True).items():
        expr = pl.col(ts)
        if tz is not None:
            try:
                expr = (pl.col(ts).dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null")
                        .dt.convert_time_zone("UTC").dt.replace_time_zone(None)
                        .fill_null(pl.col(ts)))
            except Exception:
                expr = pl.col(ts)
        parts.append(part.with_columns(expr.alias(out)))
    return pl.concat(parts, how="vertical_relaxed").drop("_tz")


def _events(df):
    prio = pl.col("source").replace_strict(SOURCE_PRIORITY, default=9, return_dtype=pl.Int64)
    first = lambda c: pl.col(c).drop_nulls().first().alias(c)
    return (
        df.filter(pl.col("person_id").is_not_null() & pl.col("flight_code").is_not_null()
                  & pl.col("date").is_not_null() & pl.col("dep_airport").is_not_null()
                  & pl.col("arr_airport").is_not_null() & pl.col("dep_ts").is_not_null()
                  & pl.col("arr_ts").is_not_null())
        .with_columns(_prio=prio).sort(["ts_est", "_prio"], maintain_order=True)
        .group_by("person_id", "flight_code", "date", maintain_order=True)
        .agg(first("dep_airport"), first("arr_airport"), first("dep_ts"), first("arr_ts"),
             pl.col("ts_est").first())
        .unique(subset=["person_id", "date", "dep_ts", "arr_ts"], keep="first", maintain_order=True)
    )


def det_movement(df, info=None):
    if "ts_est" not in df.columns:
        df = df.with_columns(ts_est=pl.lit(False))
    ev = _events(df)
    if ev.is_empty():
        return []
    tz_known = info is not None and "tz" in info.columns
    if not tz_known:
        print("[det_movement] нет справочника таймзон: сравнение по местному времени, "
              "степень серьёзности понижена (load_airport_info()).")
    ev = _to_utc(ev, "dep_ts", "dep_airport", "dep_utc", info)
    ev = _to_utc(ev, "arr_ts", "arr_airport", "arr_utc", info)
    ev = ev.sort("person_id", "dep_utc").with_columns(
        prev_airport=pl.col("arr_airport").shift(1).over("person_id"),
        prev_arr=pl.col("arr_utc").shift(1).over("person_id"),
        prev_code=pl.col("flight_code").shift(1).over("person_id"),
        prev_est=pl.col("ts_est").shift(1).over("person_id"),
        max_prev_arr=pl.col("arr_utc").shift(1).cum_max().over("person_id"),
    )
    if info is not None and {"lat", "lon"} <= set(info.columns):
        g = info.select("airport", "lat", "lon").unique("airport")
        ev = (ev.join(g.rename({"airport": "prev_airport", "lat": "lat1", "lon": "lon1"}),
                      on="prev_airport", how="left")
                .join(g.rename({"airport": "dep_airport", "lat": "lat2", "lon": "lon2"}),
                      on="dep_airport", how="left"))
        dist = _haversine_km(pl.col("lat1"), pl.col("lon1"), pl.col("lat2"), pl.col("lon2"))
        req = pl.min_horizontal(1.5 + dist / 80, 2.0 + dist / 800).fill_null(MIN_GAP_NO_COORDS_H)
        ev = ev.with_columns(dist_km=dist)
    else:
        req = pl.lit(MIN_GAP_NO_COORDS_H)
        ev = ev.with_columns(dist_km=pl.lit(None, dtype=pl.Float64))
    ev = ev.with_columns(
        gap_h=(pl.col("dep_utc") - pl.col("prev_arr")).dt.total_minutes() / 60,
        req_h=req,
    ).with_columns(
        differs=pl.col("prev_airport").is_not_null() & (pl.col("prev_airport") != pl.col("dep_airport")),
        overlap=pl.col("max_prev_arr").is_not_null() & (pl.col("dep_utc") < pl.col("max_prev_arr")),
    )
    fmt = lambda c: pl.col(c).round(1).cast(pl.String)
    hard = 3 if tz_known else 2
    # если время достроено по другим датам (ts_est), серьёзность на единицу ниже
    sev_hard = pl.when(pl.col("ts_est") | pl.col("prev_est").fill_null(False)).then(hard - 1).otherwise(hard)
    base = [pl.col("person_id"), pl.col("flight_code"), pl.col("date")]

    ov = ev.filter(pl.col("overlap")).select(
        *base, pl.lit("overlap").alias("kind"),
        pl.concat_str([pl.lit("Вылет "), pl.col("dep_airport"), pl.lit(" ("), pl.col("flight_code"),
                       pl.lit(") до прилёта предыдущего рейса "), pl.col("prev_code")]).alias("detail"),
        sev_hard.alias("severity"))
    tr = ev.filter(~pl.col("overlap") & pl.col("differs") & (pl.col("gap_h") < pl.col("req_h"))).select(
        *base, pl.lit("impossible_transfer").alias("kind"),
        pl.concat_str([pl.lit("Прилёт "), pl.col("prev_airport"), pl.lit(" ("), pl.col("prev_code"),
                       pl.lit("), вылет "), pl.col("dep_airport"), pl.lit(" ("), pl.col("flight_code"),
                       pl.lit(") через "), fmt("gap_h"), pl.lit(" ч, нужно >= "), fmt("req_h"),
                       pl.lit(" ч")]).alias("detail"),
        sev_hard.alias("severity"))
    rg = ev.filter(~pl.col("overlap") & pl.col("differs") & (pl.col("gap_h") >= pl.col("req_h"))
                   & (pl.col("gap_h") <= MAX_GAP_H)).select(
        *base, pl.lit("route_gap").alias("kind"),
        pl.concat_str([pl.lit("Прилёт "), pl.col("prev_airport"), pl.lit(", затем вылет "),
                       pl.col("dep_airport"), pl.lit(" через "), fmt("gap_h"), pl.lit(" ч")]).alias("detail"),
        pl.lit(1).alias("severity"))
    print(f"[det_movement] overlap={ov.height}, impossible_transfer={tr.height}, route_gap={rg.height}")
    return [ov, tr, rg]


def export_suspicious_passengers(df_with_ids, output_prefix="process/suspicious_spies", airport_info=None,
                                 include_not_in_pdf=False):
    print("\n=== Анализ аномалий и экспорт отчёта ===")
    df = add_card_key(prepare(df_with_ids))
    if "ts_est" not in df.columns:
        df = df.with_columns(ts_est=pl.lit(False))
    sch = df.schema
    anomalies = []

    def out(d, kind, detail, severity, event=True):
        anomalies.append(d.select(
            pl.col("person_id").cast(pl.String),
            (pl.col("flight_code") if event else pl.lit(None, dtype=sch["flight_code"]).alias("flight_code")),
            (pl.col("date") if event else pl.lit(None, dtype=sch["date"]).alias("date")),
            pl.lit(kind).alias("kind"),
            detail.cast(pl.String).alias("detail"),
            (severity if isinstance(severity, pl.Expr) else pl.lit(severity)).cast(pl.Int64).alias("severity"),
        ))

    ev_ok = pl.col("person_id").is_not_null() & pl.col("flight_code").is_not_null() & pl.col("date").is_not_null()
    is_p = pl.col("person_id").str.starts_with("P")

    if "pdf_status" in df.columns:
        flagged = ["airport_mismatch", "pair_mismatch"] + (["not_in_pdf"] if include_not_in_pdf else [])
        d = df.filter(ev_ok & pl.col("pdf_status").is_in(flagged))
        sev = (pl.when(pl.col("pdf_status") == "airport_mismatch").then(3)
               .when(pl.col("pdf_status") == "pair_mismatch").then(2).otherwise(1))
        out(d, "pdf_route",
            pl.concat_str([pl.col("pdf_status"), pl.lit(": факт "), pl.col("dep_airport").fill_null("?"),
                           pl.lit("-"), pl.col("arr_airport").fill_null("?"), pl.lit(", расписание: "),
                           pl.col("pdf_legs").fill_null("рейса нет")]), sev)

    c = (df.filter(ev_ok)
         .group_by("person_id", "flight_code", "date")
         .agg(pl.col("dep_airport").filter(pl.col("dep_airport").str.contains(IATA_RE)).unique().sort().alias("_dep"),
              pl.col("arr_airport").filter(pl.col("arr_airport").str.contains(IATA_RE)).unique().sort().alias("_arr"),
              pl.col("source").n_unique().alias("_ns"))
         .filter((pl.col("_ns") > 1) & ((pl.col("_dep").list.len() > 1) | (pl.col("_arr").list.len() > 1))))
    out(c, "source_conflict",
        pl.concat_str([pl.lit("вылет: "), pl.col("_dep").list.join("/"), pl.lit(", прилёт: "),
                       pl.col("_arr").list.join("/")]), 3)

    shared = (df.filter(pl.col("card_key").is_not_null())
              .group_by("card_key")
              .agg(pl.col("name_key").drop_nulls().unique().sort().alias("_names"),
                   pl.col("person_id").filter(is_p).unique().sort().alias("_pids"))
              .with_columns(_nn=pl.col("_names").list.len(), _np=pl.col("_pids").list.len())
              .filter((pl.col("_nn") > 1) | (pl.col("_np") > 1)))
    d = df.filter(pl.col("person_id").is_not_null() & pl.col("flight_code").is_not_null()
                  & pl.col("date").is_not_null()).join(shared, on="card_key", how="inner")
    out(d, "card_sharing",
        pl.concat_str([pl.lit("карта "), pl.col("card_key"), pl.lit(", владельцы: "),
                       pl.col("_names").list.join(", ")]),
        pl.when(pl.col("_np") > 1).then(3).otherwise(2))

    mc = (df.filter(is_p & pl.col("card_prog_f").is_not_null() & pl.col("card_num_n").is_not_null())
          .group_by("person_id", "card_prog_f")
          .agg(pl.col("card_num_n").n_unique().alias("_n"), pl.col("card_num_n").unique().sort().str.join(", ").alias("_nums"))
          .filter(pl.col("_n") > 1))
    out(mc, "multi_cards_same_program",
        pl.concat_str([pl.col("card_prog_f"), pl.lit(": "), pl.col("_nums")]), 1, event=False)

    pp = (df.filter(is_p & pl.col("passport").is_not_null())
          .select("person_id", "passport").unique()
          .with_columns(_t=pl.when(pl.col("passport").str.contains(r"^\d{10}$")).then(pl.lit("внутренний"))
                        .when(pl.col("passport").str.contains(r"^\d{9}$")).then(pl.lit("загран"))
                        .otherwise(pl.lit("иной")))
          .group_by("person_id", "_t")
          .agg(pl.len().alias("_n"), pl.col("passport").sort().str.join(", ").alias("_nums"))
          .filter(pl.col("_n") > 1))
    out(pp, "multi_passports",
        pl.concat_str([pl.col("_t"), pl.lit(": "), pl.col("_nums")]),
        pl.when(pl.col("_t") == "иной").then(2).otherwise(3), event=False)

    idn = (df.filter(is_p).group_by("person_id")
           .agg(pl.col("name_key").drop_nulls().unique().sort().alias("_names"),
                pl.col("birth_date").drop_nulls().cast(pl.String).unique().sort().alias("_births")))
    out(idn.filter(pl.col("_births").list.len() > 1), "multi_birth",
        pl.col("_births").list.join(", "), 3, event=False)
    out(idn.filter(pl.col("_names").list.len() > 1), "multi_names",
        pl.col("_names").list.join(" | "), 2, event=False)

    anomalies.extend(det_movement(df, airport_info))

    anomalies = [a for a in anomalies if a.height]
    if not anomalies:
        print("Аномалий не найдено")
        return pl.DataFrame()
    anoms = pl.concat(anomalies, how="vertical_relaxed").unique()
    print(anoms.group_by("kind").len().sort("len", descending=True))

    per_kind = anoms.group_by("person_id", "kind").agg(
        pl.col("severity").max().alias("sev"), pl.len().alias("n"),
        pl.col("detail").first().alias("example"))
    persons = per_kind.group_by("person_id").agg(
        pl.col("sev").sum().alias("score"),
        pl.col("kind").n_unique().alias("n_kinds"),
        pl.col("kind").sort().str.join(", ").alias("kinds"),
        pl.col("n").sum().alias("n_hits"),
        pl.col("example").str.join(" || ").alias("examples"))
    profile = (df.filter(pl.col("person_id").is_not_null()).group_by("person_id")
               .agg(pl.col("name_key").drop_nulls().unique().sort().str.join(" | ").alias("names"),
                    pl.col("passport").drop_nulls().unique().sort().str.join(" | ").alias("passports"),
                    pl.col("card_key").drop_nulls().unique().sort().str.join(" | ").alias("cards"),
                    pl.col("nickname").drop_nulls().unique().sort().str.join(" | ").alias("nicknames"),
                    pl.col("source").unique().sort().str.join(",").alias("sources"),
                    pl.len().alias("n_rows")))
    persons = (persons.join(profile, on="person_id", how="left")
               .sort(["score", "n_kinds"], descending=True))

    flights_info = (df.select("person_id", "flight_code", "date", "dep_airport", "arr_airport", "dep_ts",
                              "arr_ts", "card_key", "ticket")
                    .unique(subset=["person_id", "flight_code", "date"]))
    events = (anoms.filter(pl.col("flight_code").is_not_null())
              .group_by("person_id", "flight_code", "date")
              .agg(pl.concat_str([pl.col("kind"), pl.lit(": "), pl.col("detail")]).unique().str.join(" | ")
                   .alias("all_suspicion_reasons"),
                   pl.col("kind").n_unique().alias("total_reasons_count"),
                   pl.col("severity").max().alias("max_severity"))
              .join(persons.select("person_id", pl.col("kinds").alias("person_flags"), "score", "names", "passports"),
                    on="person_id", how="left")
              .join(flights_info, on=["person_id", "flight_code", "date"], how="left")
              .sort(["score", "total_reasons_count", "max_severity"], descending=True))

    for name, frame in ((output_prefix, events), (f"{output_prefix}_persons", persons)):
        frame.write_parquet(f"{name}.parquet")
        print(f"сохранено: {name}.parquet")
        try:
            frame.write_excel(f"{name}.xlsx")
            print(f"сохранено: {name}.xlsx")
        except Exception as e:
            print(f"xlsx не записан ({e})")
    return events