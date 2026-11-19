"""Configuration, ablation, dry-run and safety checks (no training)."""

import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from run import CASES, basis_columns, verify_run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class Section65Tests(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run([sys.executable, str(HERE/"run.py"), *args],
                              cwd=ROOT, capture_output=True, text=True)

    def test_table6_settings(self):
        cfg = json.loads((HERE/"configs/wave.json").read_text())
        self.assertEqual((cfg["model"], cfg["dim"], cfg["pde_problem"], cfg["test_func"]),
                         ("fouriernn", 2, "onewaywave_inhomo", "test7"))
        self.assertEqual((cfg["width"], cfg["depth"], cfg["adam_max_epochs"]), (128, 3, 40000))
        self.assertEqual((cfg["Ncol_pts"], cfg["Nbnd_pts"], cfg["Nic_pts"], cfg["Ntest_pts"]),
                         (5000, 200, 256, 100000))
        self.assertEqual((cfg["dim1_min"], cfg["dim1_max"], cfg["dim2_min"], cfg["dim2_max"]), (0, 1, 0, 1))
        self.assertTrue(cfg["initial_cond"])
        self.assertFalse(cfg["homogeneous_pde"])
        self.assertEqual(cfg["bnd_typ"], "dirichlet")
        self.assertNotIn("normalize_fourier_basis", cfg)
        self.assertNotIn("input_norm_type", cfg)
        self.assertEqual(cfg["fourier_periods"], [1, 2])
        self.assertEqual(cfg["lstsq_stop_epoch"], 20000)
        self.assertNotIn("lstsq_initial", cfg)
        self.assertNotIn("lstsq_column_scaling", cfg)

    def test_default_and_ablation_sweep(self):
        default = self.run_script()
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(len(default.stdout.splitlines()), 5)
        self.assertIn("all target modes are included for K>=10", default.stderr)
        result = self.run_script("--cases", *CASES)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(commands), 20)
        self.assertEqual(len({next(x for x in c if x.startswith("--datadir=")) for c in commands}), 20)
        for index, (case, (regularize, prune)) in enumerate(CASES.items()):
            cmd = commands[index*5]
            coefficient = "0.0001" if regularize else "0"
            for branch in ("nn", "pn"):
                self.assertIn(f"--{branch}_truncate={str(prune).lower()}", cmd)
                self.assertIn(f"--{branch}_l2_reg_coeff={coefficient}", cmd)
        self.assertEqual(basis_columns(200), 160801)

    def test_memory_guard_before_launch(self):
        result = self.run_script("--run", "--ks", "2", "200", "--binary=/nonexistent/binary")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--allow-large-basis", result.stderr)
        self.assertNotIn("FileNotFoundError", result.stderr)

    def test_smoke_overrides(self):
        result = self.run_script("--smoke", "--test-batch-size=7", "--lstsq-rcond=1e-4", "--lstsq-stop-epoch=2")
        self.assertEqual(result.returncode, 0, result.stderr)
        cmd = shlex.split(result.stdout.strip())
        for option in ("--dim=2", "--ell=2", "--Nic_pts=16", "--test_batch_size=7", "--lstsq_rcond=0.0001"):
            self.assertIn(option, cmd)
        self.assertEqual([x for x in cmd if x.startswith("--lstsq_stop_epoch=")][-1], "--lstsq_stop_epoch=2")

    def test_invalid_parameters(self):
        for argument in ("--ks=0", "--lstsq-rcond=nan", "--lstsq-rcond=inf", "--lstsq-rcond=-2",
                         "--lstsq-stop-epoch=-2", "--test-batch-size=0", "--test-batch-size=-1"):
            self.assertNotEqual(self.run_script(argument).returncode, 0)

    def test_nonfinite_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"log.txt"
            path.write_text("Saving Arg CSV line: final_error=0.1, final_loss=0.2")
            verify_run(path)
            for value in ("", "Saving Arg CSV line: final_error=nan, final_loss=1",
                          "Saving Arg CSV line: final_error=1, final_loss=inf"):
                path.write_text(value)
                with self.assertRaises(RuntimeError):
                    verify_run(path)


if __name__ == "__main__":
    unittest.main()
