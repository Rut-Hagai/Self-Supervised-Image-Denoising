"""Download the data and the published checkpoints. Needs git and a network
connection; run once:

    python scripts/get_data.py

  data/BSD500_train  the 200 training images of BSDS500
  data/Kodak         the 24 Kodak test images
  weights/           the Neighbor2Neighbor checkpoints released by its authors
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = [
    dict(url="https://github.com/TaoHuang2018/Neighbor2Neighbor.git",
         copies=[("validation/Kodak", "data/Kodak"),
                 ("pretrained_model", "weights")]),
    dict(url="https://github.com/BIDS/BSDS500.git",
         copies=[("BSDS500/data/images/train", "data/BSD500_train")]),
]


def clone(url, dest):
    print(f"  cloning {url}")
    subprocess.run(["git", "clone", "--depth", "1", url, dest],
                   check=True, stdout=subprocess.DEVNULL)


KEEP = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pth"}


def copy_tree(src, dst):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(src)):
        s = os.path.join(src, name)
        if os.path.isfile(s) and os.path.splitext(name)[1].lower() in KEEP:
            shutil.copy2(s, os.path.join(dst, name))
            n += 1
    return n


def main():
    for spec in SOURCES:
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repo")
            try:
                clone(spec["url"], repo)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                print(f"  FAILED: {exc}\n  Is git installed and online?")
                return 1
            for rel_src, rel_dst in spec["copies"]:
                src = os.path.join(repo, *rel_src.split("/"))
                dst = os.path.join(ROOT, *rel_dst.split("/"))
                n = copy_tree(src, dst)
                print(f"  {rel_dst}: {n} files")
    print("\nready. try:  python -m experiments.exp_b_zs_n2n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
