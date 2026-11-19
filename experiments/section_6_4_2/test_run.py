"""Config, dry-run, and memory-guard checks; never launch training."""

import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from run import basis_columns, verify_run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class Section642Tests(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run([sys.executable, str(HERE / "run.py"), *args],
                              cwd=ROOT, capture_output=True, text=True)

    def test_settings(self):
        cfg = json.loads((HERE / "configs/multiscale.json").read_text())
        self.assertEqual((cfg["model"], cfg["dim"], cfg["pde_problem"], cfg["test_func"]),
                         ("fouriernn", 2, "allencahn", "test6"))
        self.assertEqual((cfg["width"], cfg["depth"]), (100, 3))
        self.assertEqual((cfg["Ncol_pts"], cfg["Nbnd_pts"], cfg["Ntest_pts"]), (12000, 400, 100000))
        self.assertEqual((cfg["adam_max_epochs"], cfg["lstsq_stop_epoch"]), (40000, 20000))
        self.assertEqual((cfg["dim1_min"], cfg["dim2_min"]), (0, 0))
        self.assertAlmostEqual(cfg["dim1_max"], 6.283185307179586)
        self.assertEqual(cfg["dim1_max"], cfg["dim2_max"])
        self.assertEqual(cfg["point_sampling"], "random")
        self.assertTrue(cfg["lstsq_safeguard"])
        self.assertFalse(cfg["initial_cond"])
        self.assertFalse(cfg["homogeneous_pde"])
        self.assertNotIn("normalize_fourier_basis", cfg)
        self.assertNotIn("input_norm_type", cfg)
        self.assertNotIn("lstsq_initial", cfg)
        self.assertNotIn("lstsq_column_scaling", cfg)

    def test_default_dry_sweep(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(commands), 5)
        self.assertEqual(len({next(x for x in c if x.startswith("--datadir=")) for c in commands}), 5)
        self.assertEqual(basis_columns(200), 160801)
        self.assertIn("160801", result.stderr)

    def test_guard_checks_all_sizes_before_launch(self):
        result = self.run_script("--run", "--ks", "2", "200", "--binary=/nonexistent/binary")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--allow-large-basis", result.stderr)
        self.assertNotIn("FileNotFoundError", result.stderr)

    def test_smoke_and_overrides(self):
        result = self.run_script("--smoke", "--lstsq-rcond=1e-4", "--lstsq-stop-epoch=2", "--test-batch-size=7")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(commands), 1)
        cmd = commands[0]
        for option in ("--dim=2", "--ell=2", "--test_batch_size=7", "--n_lstsq_upds=10", "--lstsq_rcond=0.0001"):
            self.assertIn(option, cmd)
        self.assertEqual([x for x in cmd if x.startswith("--lstsq_stop_epoch=")][-1], "--lstsq_stop_epoch=2")
        self.assertIn("omits exact frequencies", result.stderr)

    def test_invalid_parameters(self):
        for argument in ("--ks=0", "--lstsq-rcond=nan", "--lstsq-rcond=inf", "--lstsq-rcond=-2",
                         "--lstsq-stop-epoch=-2", "--test-batch-size=0", "--test-batch-size=-1"):
            self.assertNotEqual(self.run_script(argument).returncode, 0)

    def test_nonfinite_logs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            path.write_text("Saving Arg CSV line: final_error=0.9, final_loss=0.2")
            verify_run(path)
            for text in ("", "Saving Arg CSV line: final_error=nan, final_loss=0.2",
                         "Saving Arg CSV line: final_error=0.9, final_loss=inf"):
                path.write_text(text)
                with self.assertRaises(RuntimeError):
                    verify_run(path)


if __name__ == "__main__":
    unittest.main()
