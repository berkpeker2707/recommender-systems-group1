"""
run_all.py — the single documented command (brief §4, requirement 1).

    python run_all.py --data-dir data/ml-1m [--out results]

1. Parse args (data dir is configurable; no hard-coded paths).
2. Set up logging to results/logs/run.log and print environment info
   (Python + package versions) so the log proves which environment produced the numbers.
3. Run src.experiments.part1 (loads data, builds the split, runs the baselines).
4. The experiment writes rows via src.results -> results/results.csv
   in the course template schema; figures go to results/figures/.
5. src.report rewrites the results section of README.md from those files.
6. Exit non-zero on any failure, so a broken run is never silently "complete".
"""
import argparse
import logging
import platform
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


def log_environment(log):
    import matplotlib, numpy, pandas, scipy
    log.info("python %s on %s", platform.python_version(), platform.platform())
    log.info("numpy %s, pandas %s, scipy %s, matplotlib %s",
             numpy.__version__, pandas.__version__, scipy.__version__, matplotlib.__version__)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder containing ratings.dat and movies.dat")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    setup_logging(out_dir)
    log = logging.getLogger("run_all")
    log_environment(log)

    data_dir = Path(args.data_dir)
    missing = [f for f in ("ratings.dat", "movies.dat") if not (data_dir / f).exists()]
    if missing:
        log.error("missing %s in %s — point --data-dir at the unzipped ml-1m folder", missing, data_dir)
        sys.exit(1)

    from src import report
    from src.experiments import part1
    t0 = time.time()
    log.info("===== Part 1 =====")
    part1.run(data_dir, out_dir)
    log.info("Part 1 done in %.1f s", time.time() - t0)

    readme = Path(__file__).parent / "README.md"
    report.update_readme(readme, out_dir)
    log.info("results written into %s", readme)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("run_all").exception("run failed")
        sys.exit(1)
