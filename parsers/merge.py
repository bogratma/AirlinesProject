import polars as pl

SOURCE_PRIORITY = {"tab": 0, "csv": 1, "boarding": 2, "yaml": 3, "xml": 4, "json": 5}
EVENT_FIELDS = ["dep_airport", "arr_airport", "dep_ts", "arr_ts", "name_key", "birth_date",
                "passport", "ticket", "booking_code", "card_program", "card_number",
                "nickname", "travel_class", "fare", "agent"]


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _ids(u):
    def mk(prefix, *cols):
        parts = [pl.lit(prefix + ":")]
        for i, c in enumerate(cols):
            if i:
                parts.append(pl.lit(":"))
            parts.append(pl.col(c).fill_null(""))
        return pl.when(pl.col(cols[-1]).is_not_null()).then(pl.concat_str(parts))

    return u.with_columns(
        mk("card", "card_program", "card_number").alias("_i_card"),
        mk("pass", "passport").alias("_i_pass"),
        mk("tick", "ticket").alias("_i_tick"),
        mk("nick", "nickname").alias("_i_nick"),
        pl.when(pl.col("name_key").is_not_null() & pl.col("birth_date").is_not_null())
          .then(pl.concat_str([pl.lit("nb:"), pl.col("name_key"), pl.lit("|"), pl.col("birth_date")]))
          .alias("_i_nb"),
    )


ID_COLS = ["_i_card", "_i_pass", "_i_tick", "_i_nick", "_i_nb"]


def assign_person_id(u):
    u = _ids(u.with_row_index("_row"))
    parent = {}
    for row in u.select(ID_COLS).unique().iter_rows():
        ids = [x for x in row if x is not None]
        for x in ids:
            parent.setdefault(x, x)
        for x in ids[1:]:
            ra, rb = _find(parent, ids[0]), _find(parent, x)
            if ra != rb:
                parent[rb] = ra

    root_to_int, idents, pids = {}, [], []
    for k in parent:
        idents.append(k)
        pids.append(root_to_int.setdefault(_find(parent, k), len(root_to_int)))
    idmap = pl.DataFrame({"ident": idents, "person_id": pids},
                         schema={"ident": pl.String, "person_id": pl.Int64})

    for col in ID_COLS:
        u = u.join(idmap.rename({"ident": col, "person_id": f"pid{col}"}), on=col, how="left")
    u = (u.sort("_row")
          .with_columns(person_id=pl.coalesce([f"pid{c}" for c in ID_COLS]))
          .drop(["_row"] + ID_COLS + [f"pid{c}" for c in ID_COLS]))
    return u


def merge_events(u):
    ev = u.filter(pl.col("person_id").is_not_null()
                  & pl.col("flight_code").is_not_null() & pl.col("date").is_not_null())
    prio = pl.col("source").replace_strict(SOURCE_PRIORITY, default=9, return_dtype=pl.Int64)
    ev = ev.with_columns(_prio=prio).sort("_prio", maintain_order=True)

    first = lambda c: pl.col(c).drop_nulls().first().alias(c)
    return (ev.group_by("person_id", "flight_code", "date", maintain_order=True).agg(
                pl.col("source").unique().sort().alias("sources"),
                *[first(c) for c in EVENT_FIELDS],
                (pl.col("dep_airport").drop_nulls().n_unique() > 1).alias("dep_conflict"),
                (pl.col("arr_airport").drop_nulls().n_unique() > 1).alias("arr_conflict"),
            ).with_columns(n_sources=pl.col("sources").list.len()))
