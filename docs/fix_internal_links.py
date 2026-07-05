#!/usr/bin/env python

import re
import sys

if len(sys.argv) <= 1:
    sys.exit(0)


def subber(m):
    return m.group(0).replace("``", "`")


prog = re.compile(r":(.+):``(.+)``")

for fn in sys.argv[1:]:
    print(f"Fixing links in {fn}")
    with open(fn) as f:
        txt = f.read()
    txt = prog.sub(subber, txt)
    with open(fn, "w") as f:
        f.write(txt)
