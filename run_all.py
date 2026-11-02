"""
Runs the project: python run_all.py --data-dir data/ml-1m [--out results]
"""
import argparse
import logging
import sys
import time
from pathlib import Path


def setup_logging(out_dir: Path):
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out_dir / "logs" / "run.log", mode="w")])
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder containing ratings.dat and movies.dat")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    setup_logging(out_dir)
    log = logging.getLogger("run_all")

    data_dir = Path(args.data_dir)
    missing = [f for f in ("ratings.dat", "movies.dat") if not (data_dir / f).exists()]
    if missing:
        log.error("missing %s in %s — point --data-dir at the unzipped ml-1m folder", missing, data_dir)
        sys.exit(1)

    from src.experiments.run_part1 import run_part1
    t0 = time.time()
    log.info("===== Part 1 =====")
    run_part1(data_dir, out_dir)
    log.info("Part 1 done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("run_all").exception("run failed")
        sys.exit(1)
