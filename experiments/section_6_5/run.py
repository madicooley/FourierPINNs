#!/usr/bin/env python3
"""Print or execute Section 6.5 one-way wave FourierPINN runs and ablations."""

import argparse
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys

CASES = {
    "reg_prune": (True, True),
    "no_reg_prune": (False, True),
    "reg_no_prune": (True, False),
    "no_reg_no_prune": (False, False),
}


def basis_columns(k):
    return (2*k + 1)**2


def verify_run(log_path):
    metrics = re.search(r"Saving Arg CSV line: final_error=([^,]+), final_loss=([^,]+)", log_path.read_text())
    if not metrics or not all(math.isfinite(float(value)) for value in metrics.groups()):
        raise RuntimeError("Missing/non-finite final metrics in " + str(log_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute; otherwise only print commands.")
    parser.add_argument("--smoke", action="store_true", help="Tiny pipeline check, not an accuracy test.")
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=["reg_prune"])
    parser.add_argument("--ks", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--binary", default="build/bin/fourierpinns")
    parser.add_argument("--output-root", default="runs/section_6_5")
    parser.add_argument("--lstsq-rcond", type=float)
    parser.add_argument("--lstsq-stop-epoch", type=int)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--allow-large-basis", action="store_true",
                        help="Allow execution above 4096 Fourier columns; may exhaust memory.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    ks = args.ks if args.ks is not None else ([2] if args.smoke else [200])
    seeds = args.seeds if args.seeds is not None else ([1234] if args.smoke else list(range(1234, 1239)))
    if any(k < 1 for k in ks):
        parser.error("K must be positive")
    if args.test_batch_size <= 0:
        parser.error("test-batch-size must be positive")
    if args.lstsq_rcond is not None and not (args.lstsq_rcond == -1 or 0 <= args.lstsq_rcond < 1):
        parser.error("lstsq-rcond must be -1 or in [0, 1)")
    if args.lstsq_stop_epoch is not None and args.lstsq_stop_epoch < -1:
        parser.error("lstsq-stop-epoch must be -1 or nonnegative")
    if args.run and any(basis_columns(k) > 4096 for k in ks) and not args.allow_large_basis:
        parser.error("Tensor-product basis exceeds 4096 columns. Inspect a dry run first; "
                     "use --allow-large-basis only after checking memory, or --smoke.")
    config_path = Path("experiments/section_6_5/configs/wave.json")
    cfg = json.loads((root/config_path).read_text())
    print("Fourier periods [t,x]=[1,2]: all target modes are included for K>=10.", file=sys.stderr)
    for case in args.cases:
        regularize, prune = CASES[case]
        for k in ks:
            columns = basis_columns(k)
            ntest = 121 if args.smoke else cfg["Ntest_pts"]
            gib = min(ntest, args.test_batch_size)*columns*4/1024**3
            print(f"{case}: K={k}, Fourier columns={columns}, test basis chunk ~{gib:.3f} GiB "
                  "(float32; not peak memory)", file=sys.stderr)
            if k < 10:
                print(f"WARNING: K={k} also omits the temporal frequency 10 (20*pi*t).", file=sys.stderr)
            for seed in seeds:
                expid = f"s65_{case}_K{k}_s{seed}" + ("_smoke" if args.smoke else "")
                directory = Path(args.output_root)/expid
                coefficient = "0.0001" if regularize else "0"
                truncation = "true" if prune else "false"
                command = [args.binary, "train", "--config="+str(config_path),
                           "--dim=2", f"--ell={k}", f"--seed={seed}",
                           f"--test_batch_size={args.test_batch_size}",
                           f"--nn_truncate={truncation}", f"--pn_truncate={truncation}",
                           f"--nn_l2_reg_coeff={coefficient}", f"--pn_l2_reg_coeff={coefficient}",
                           "--expid="+expid, "--datadir="+str(directory)+"/"]
                if args.smoke:
                    command += ["--Ncol_pts=64", "--Nbnd_pts=16", "--Nic_pts=16", "--Ntest_pts=121",
                                "--width=4", "--adam_max_epochs=4", "--lstsq_upd_interval=2",
                                "--lstsq_stop_epoch=4", "--n_lstsq_upds=10",
                                "--lbfgs_max_epochs=2", "--lbfgs_max_iter=3", "--print_interval=1"]
                if args.lstsq_rcond is not None:
                    command.append(f"--lstsq_rcond={args.lstsq_rcond}")
                if args.lstsq_stop_epoch is not None:
                    command.append(f"--lstsq_stop_epoch={args.lstsq_stop_epoch}")
                print(" ".join(shlex.quote(part) for part in command), flush=True)
                if args.run:
                    subprocess.run(command, cwd=root, check=True)
                    verify_run(root/directory/"logs"/f"logs_{expid}_fouriernn_forwardpde.txt")


if __name__ == "__main__":
    main()
