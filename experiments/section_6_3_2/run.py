#!/usr/bin/env python3
"""Print or execute the Section 6.3.2 FourierPINN sweep from the repository root."""

import argparse
import math
import re
import shlex
import subprocess
from pathlib import Path


def verify_run(log_path):
    text = log_path.read_text()
    metrics = re.search(r"Saving Arg CSV line: final_error=([^,]+), final_loss=([^,]+)", text)
    if not metrics or not all(math.isfinite(float(value)) for value in metrics.groups()):
        raise RuntimeError("Missing/non-finite final metrics; inspect nonlinear LS updates in " + str(log_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute commands; otherwise only print them.")
    parser.add_argument("--smoke", action="store_true", help="Short run exercising two nonlinear LS/pruning blocks and LBFGS.")
    parser.add_argument("--cases", nargs="+", choices=["single_scale", "multiscale", "product"],
                        default=["single_scale", "multiscale", "product"])
    parser.add_argument("--ks", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--binary", default="build/bin/fourierpinns")
    parser.add_argument("--output-root", default="runs/section_6_3_2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    ks = args.ks if args.ks is not None else ([110] if args.smoke else [100, 150, 200, 250, 300])
    seeds = args.seeds if args.seeds is not None else ([1234] if args.smoke else [1234, 1235, 1236, 1237, 1238])
    if any(k < 1 for k in ks):
        parser.error("K must be positive")
    for case in args.cases:
        for k in ks:
            for seed in seeds:
                expid = "s632_{}_K{}_s{}{}".format(case, k, seed, "_smoke" if args.smoke else "")
                command = [args.binary, "train",
                           "--config=experiments/section_6_3_2/configs/{}.json".format(case),
                           "--dim=1", "--ell={}".format(k), "--seed={}".format(seed),
                           "--expid=" + expid,
                           "--datadir=" + str(Path(args.output_root) / expid) + "/"]
                if args.smoke:
                    command += ["--Ncol_pts=512", "--Nbnd_pts=16", "--Ntest_pts=1024",
                                "--width=8", "--adam_max_epochs=4", "--lstsq_upd_interval=2",
                                "--lstsq_stop_epoch=4",
                                "--n_lstsq_upds=10", "--lbfgs_max_epochs=2", "--lbfgs_max_iter=3",
                                "--print_interval=1"]
                print(" ".join(shlex.quote(part) for part in command), flush=True)
                if args.run:
                    subprocess.run(command, cwd=str(root), check=True)
                    log_path = (root / args.output_root / expid / "logs" /
                                ("logs_" + expid + "_fouriernn_forwardpde.txt"))
                    verify_run(log_path)


if __name__ == "__main__":
    main()
