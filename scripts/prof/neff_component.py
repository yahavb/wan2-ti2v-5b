"""Guess which model component a NEFF belongs to, for naming profiles in Neuron Explorer.

A NEFF is a binary header followed by a gzip'd tar of the compiled graph, whose HLO metadata
carries the python source files and op names it was traced from. This scans those strings and
maps them to a component slug: dit / vae / t5 / rope / ... plus "_cc" when the graph contains
collectives (all-reduce / all-gather), which also means `capture --single-io` cannot run it.

It is a NAMING HINT, not proof: it prints "unk" when nothing matches.

Usage: python3 neff_component.py <file.neff> [more.neff ...]   -> "<path>\t<slug>" per line
"""
import collections
import re
import sys
import zlib

# (slug, substrings of source paths / op names), first match by hit count wins
RULES = [
    ("vae", ("vae2_2", "vae_conv", "conv3d", "vae_static", "vae_attention", "causalconv")),
    ("dit", ("fast_dit", "wan/modules/model", "dit_attention", "self_attention", "cross_attention",
             "constmax", "const_max", "tp_utils")),
    ("t5", ("wan/modules/t5", "t5.py", "umt5")),
    ("rope", ("rope",)),
    ("sched", ("fm_solvers", "unipc")),
]
CC = (b"all-reduce", b"all_reduce", b"all-gather", b"all_gather", b"reduce-scatter", b"collective-permute")


def payload(raw):
    """Decompressed bytes of every gzip member in the file (plus the raw bytes as a fallback)."""
    out = [raw]
    i = raw.find(b"\x1f\x8b\x08")
    while i != -1:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            out.append(d.decompress(raw[i:], 512 << 20))
        except zlib.error:
            pass
        i = raw.find(b"\x1f\x8b\x08", i + 1)
    return b"".join(out)


def classify(path):
    data = payload(open(path, "rb").read()).lower()
    pys = collections.Counter(m.decode("latin1") for m in re.findall(rb"[\w./-]{3,}\.py", data))
    hits = collections.Counter()
    for slug, keys in RULES:
        for k in keys:
            kb = k.encode()
            hits[slug] += data.count(kb)
    slug = hits.most_common(1)[0][0] if hits and hits.most_common(1)[0][1] > 0 else None
    if slug is None and pys:
        top = pys.most_common(1)[0][0].rsplit("/", 1)[-1][:-3]
        slug = re.sub(r"[^a-z0-9]", "", top)[:12] or "unk"
    slug = slug or "unk"
    if any(c in data for c in CC):
        slug += "_cc"
    return slug


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            print(f"{p}\t{classify(p)}")
        except Exception as e:  # never break the caller over a naming hint
            print(f"{p}\tunk", file=sys.stdout)
            print(f"neff_component: {p}: {e}", file=sys.stderr)
