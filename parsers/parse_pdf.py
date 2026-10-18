import csv
import re
from datetime import date
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf


FIRST_PAGE = 4
LAST_PAGE = None
BATCH_PAGES = 500
LINE_TOL = 3.0

RAW_NAME = "skyteam_raw.parquet"
SCHEDULE_NAME = "skyteam_schedule.parquet"
UNMATCHED_NAME = "skyteam_unmatched.csv"
AIRCRAFT_NAME = "skyteam_aircraft.parquet"
AIRCRAFT_PAGE = 3

SCHEMA = pa.schema([
    ('page', pa.int32()), ('side', pa.string()),
    ('orig_iata', pa.string()), ('orig_city', pa.string()), ('orig_country', pa.string()),
    ('dest_iata', pa.string()), ('dest_city', pa.string()), ('dest_country', pa.string()),
    ('valid_from', pa.date32()), ('valid_to', pa.date32()),
    ('days', pa.string()),               # как в PDF: 1=Вс ... 7=Сб, например "13457"
    ('dep_time', pa.string()), ('arr_time', pa.string()),   # локальное время аэропорта
    ('arr_plus_days', pa.int8()),
    ('flight', pa.string()), ('airline', pa.string()), ('codeshare', pa.bool_()),
    ('aircraft', pa.string()), ('duration_min', pa.int32()),
    ('operated_by', pa.string()),
])
DEDUP_COLS = [c for c in SCHEMA.names if c not in ('page', 'side')]

MONTHS = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}


def to_date(s: str) -> date:
    d, m = s.split()
    m = MONTHS[m]
    return date(2018 if m >= 11 else 2019, m, int(d))


FROM = re.compile(r'^FROM:\s*(?P<city>.+?)\s*,\s*(?P<country>.+?)\s+(?P<iata>[A-Z]{3})$')
TO = re.compile(r'^TO:\s*(?P<city>.+?)\s*,\s*(?P<country>.+?)\s+(?P<iata>[A-Z]{3})$')
ROW = re.compile(
    r'^(?P<vf>\d{2} \w{3})\s*-\s*(?P<vt>\d{2} \w{3})\s+'
    r'(?P<days>[1-7 ]+?)\s+'
    r'(?P<dep>\d{2}:\d{2})\s+(?P<arr>\d{2}:\d{2})(?P<plus>\+\d)?\s+'
    r'(?P<flight>[A-Z0-9]{2}\d{1,4}\*?)\s+(?P<ac>[A-Z0-9]{3})\s+'
    r'(?P<h>\d+)H(?P<m>\d+)M$')
OPER = re.compile(r'^Operated by:\s*(.+)$')
NOISE = re.compile(r'^(Validity|Time(\s+Time)*$|Dep\b|Arr\b|Consult your travel agent)')


def page_lines(page, clip, tol=LINE_TOL):
    words = page.get_text("words", clip=clip)
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines, cur, cur_y = [], [], None
    for w in words:
        yc = (w[1] + w[3]) / 2
        if cur_y is None or abs(yc - cur_y) <= tol:
            cur.append(w)
            if cur_y is None:
                cur_y = yc
        else:
            lines.append(cur)
            cur, cur_y = [w], yc
    if cur:
        lines.append(cur)
    return [" ".join(x[4] for x in sorted(l, key=lambda w: w[0])) for l in lines]


def parse_aircraft(doc, page_idx=AIRCRAFT_PAGE):
    text = doc[page_idx].get_text("text")
    pairs = re.findall(r'^([A-Z0-9]{3})\s*:\s*(.+)$', text, flags=re.M)
    return pl.DataFrame(pairs, schema=['code', 'name'], orient='row').unique(maintain_order=True)


def parse(path, out_raw, out_unmatched, first_page, last_page, batch_pages):
    doc = pymupdf.open(path)
    last_page = last_page or len(doc)
    route = {'L': {}, 'R': {}}
    last = {'L': None, 'R': None}
    rows, cur_page = [], [0]
    n_rows = 0

    def process(side, line, csv_w):
        line = line.strip()
        if not line:
            return
        if m := FROM.match(line):
            route[side] = {'orig_iata': m['iata'], 'orig_city': m['city'],
                           'orig_country': m['country']}
            last[side] = None
        elif m := TO.match(line):
            route[side] |= {'dest_iata': m['iata'], 'dest_city': m['city'],
                            'dest_country': m['country']}
        elif m := ROW.match(line):
            fl = m['flight']
            r = {**route[side], 'page': cur_page[0], 'side': side,
                 'valid_from': to_date(m['vf']), 'valid_to': to_date(m['vt']),
                 'days': re.sub(r'\s', '', m['days']),
                 'dep_time': m['dep'], 'arr_time': m['arr'],
                 'arr_plus_days': int(m['plus'][1]) if m['plus'] else 0,
                 'flight': fl.rstrip('*'), 'airline': fl[:2],
                 'codeshare': fl.endswith('*'), 'aircraft': m['ac'],
                 'duration_min': int(m['h']) * 60 + int(m['m']),
                 'operated_by': None}
            rows.append(r)
            last[side] = r
        elif m := OPER.match(line):
            if last[side]:
                last[side]['operated_by'] = m[1]
        elif not NOISE.match(line):
            csv_w.writerow([cur_page[0], side, line])

    def flush(writer, keep):
        nonlocal n_rows
        ready = [r for r in rows if not any(r is k for k in keep)]
        pending = [r for r in rows if any(r is k for k in keep)]
        if ready:
            writer.write_table(pa.Table.from_pylist(ready, schema=SCHEMA))
            n_rows += len(ready)
        rows[:] = pending

    with open(out_unmatched, 'w', newline='', encoding='utf-8') as f_un, \
            pq.ParquetWriter(out_raw, SCHEMA, compression='zstd') as writer:
        csv_w = csv.writer(f_un)
        csv_w.writerow(['page', 'side', 'line'])
        for i in range(first_page, last_page):
            page = doc[i]
            cur_page[0] = i + 1
            w, h = page.rect.width, page.rect.height
            for side, clip in (('L', pymupdf.Rect(0, 0, w / 2, h)),
                               ('R', pymupdf.Rect(w / 2, 0, w, h))):
                for line in page_lines(page, clip):
                    process(side, line, csv_w)
            if (i - first_page + 1) % batch_pages == 0:
                flush(writer, (last['L'], last['R']))
                print(f"страница {i + 1}/{last_page}, строк записано: {n_rows}", flush=True)
        flush(writer, ())
    print(f"готово: {n_rows} строк -> {out_raw}")
    return doc


def dedup(raw, out):
    (pl.scan_parquet(raw)
       .unique(subset=DEDUP_COLS, keep='first', maintain_order=True)
       .sink_parquet(out))


def parse_pdf(pdf_path,output_path):
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    raw, sched = out / RAW_NAME, out / SCHEDULE_NAME

    doc = parse(pdf_path, raw, out / UNMATCHED_NAME, FIRST_PAGE, LAST_PAGE, BATCH_PAGES)
    dedup(raw, sched)
    parse_aircraft(doc).write_parquet(out / AIRCRAFT_NAME)

    n_raw = pl.scan_parquet(raw).select(pl.len()).collect().item()
    n_ded = pl.scan_parquet(sched).select(pl.len()).collect().item()
    print(f"строк до дедупликации: {n_raw}, после: {n_ded}")
