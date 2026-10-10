#!/usr/bin/env python3
"""Exit 0 only if a parquet dir has at least one READABLE table with rows.

Presence is not success. A killed converter leaves all 47 table headers on disk -- about 20 KB --
and every reader then fails with InvalidInputException. Counting files called that a success once
and produced a report built on nothing, so the check is: can a reader actually open a table, and
does it contain rows.
"""
import glob
import os
import sys

import duckdb

d = sys.argv[1]
ok = rows = 0
for f in glob.glob(os.path.join(d, "*.parquet")):
    try:
        n = duckdb.connect().execute("SELECT count(*) FROM read_parquet(?)", [f]).fetchone()[0]
        if n:
            ok += 1
            rows += n
    except Exception:                                              # noqa: BLE001
        pass
print("    readable tables with rows: %d  (%s rows total)" % (ok, "{:,}".format(rows)))
sys.exit(0 if ok else 1)
