"""Contact sheet: rows = dumps, columns = frames. python3 contact_sheet.py out.png frame,frame,.. dir [dir ...]"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


def main():
    out, frames, dirs = sys.argv[1], [int(f) for f in sys.argv[2].split(",")], sys.argv[3:]
    scale = 0.5
    rows = []
    for d in dirs:
        v = np.load(os.path.join(d, "frames.npy"))
        tiles = []
        for f in frames:
            im = Image.fromarray(v[f])
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
            ImageDraw.Draw(im).text((4, 4), f"{os.path.basename(d.rstrip('/'))} f{f}", fill=(255, 0, 0))
            tiles.append(np.asarray(im))
        rows.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    print("wrote", out, Image.open(out).size)


if __name__ == "__main__":
    main()
