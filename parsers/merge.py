import polars as pl

SOURCE_PRIORITY = {"tab": 0, "yaml": 1, "json": 2}   # при конфликте полей верим tab, потом yaml, потом json

def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def assign_person_id(u):
    u = u.with_row_index("_row").with_columns(
        pl.when(pl.col("card_number").is_not_null())
          .then(pl.concat_str([pl.lit("card:"), pl.col("card_program").fill_null(""),
                               pl.lit(":"), pl.col("card_number")])).alias("_c"),
        pl.when(pl.col("passport").is_not_null())
          .then(pl.concat_str([pl.lit("pass:"), pl.col("passport")])).alias("_p"),
        pl.when(pl.col("nickname").is_not_null())
          .then(pl.concat_str([pl.lit("nick:"), pl.col("nickname")])).alias("_n"),
    )
    parent = {}
    for c, p, n in u.select("_c", "_p", "_n").unique().iter_rows():
        ids = [x for x in (c, p, n) if x is not None]
        for x in ids:
            parent.setdefault(x, x)
        for x in ids[1:]:
            ra, rb = _find(parent, ids[0]), _find(parent, x)
            if ra != rb:
                parent[rb] = ra
    root_to_int = {}
    idents, pids = [], []
    for k in parent:
        r = _find(parent, k)
        idents.append(k)
        pids.append(root_to_int.setdefault(r, len(root_to_int)))
    idmap = pl.DataFrame({"ident": idents, "person_id": pids},
                         schema={"ident": pl.String, "person_id": pl.Int64})
    for col in ("_c", "_p", "_n"):
        u = u.join(idmap.rename({"ident": col, "person_id": f"pid{col}"}), on=col, how="left")
    u = u.sort("_row").with_columns(person_id=pl.coalesce("pid_c", "pid_p", "pid_n"))
    offset = len(root_to_int)
    weak = pl.concat_str([pl.col("name_key"), pl.lit("|"), pl.col("birth_date").fill_null("")])
    u = u.with_columns(
        pl.when(pl.col("person_id").is_null() & pl.col("name_key").is_not_null())
          .then(weak).alias("_weak"))
    weak_ids = (u.filter(pl.col("_weak").is_not_null()).select("_weak").unique()
                  .with_row_index("_w_i").with_columns(_w_pid=pl.col("_w_i").cast(pl.Int64) + offset))
    u = u.join(weak_ids.select("_weak", "_w_pid"), on="_weak", how="left")
    u = u.with_columns(
        person_id=pl.coalesce("person_id", "_w_pid", -(pl.col("_row").cast(pl.Int64) + 1)))
    return u.drop("_row", "_c", "_p", "_n", "pid_c", "pid_p", "pid_n", "_weak", "_w_pid")


def merge_events(u):
    prio = pl.col("source").replace_strict(SOURCE_PRIORITY, default=9)
    u = u.with_columns(_prio=prio).sort("_prio", maintain_order=True)
    first = lambda c: pl.col(c).drop_nulls().first().alias(c)
    return (u.group_by("person_id", "flight_code", "date", maintain_order=True).agg(
                pl.col("source").unique().sort().alias("sources"),
                first("dep_airport"), first("arr_airport"),
                first("dep_ts"), first("arr_ts"),
                first("name_key"), first("birth_date"), first("passport"),
                first("card_program"), first("card_number"), first("nickname"),
                first("booking_code"), first("ticket"), first("agent"),
                (pl.col("dep_airport").drop_nulls().n_unique() > 1).alias("dep_conflict"),
                (pl.col("arr_airport").drop_nulls().n_unique() > 1).alias("arr_conflict"),
            ).with_columns(n_sources=pl.col("sources").list.len()))