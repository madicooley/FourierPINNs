"""Configuration, dry-run, and memory guard checks; never launch training."""

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


class Section641Tests(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run([sys.executable, str(HERE / "run.py"), *args],
                              cwd=ROOT, capture_output=True, text=True)

    def test_paper_settings_and_axis_constants(self):
        for name, selector in (("single_scale", "test4"), ("multiscale", "test5")):
            cfg = json.loads((HERE / "configs" / (name + ".json")).read_text())
            self.assertEqual((cfg["dim"], cfg["pde_problem"], cfg["test_func"]), (2, "poisson", selector))
            self.assertEqual((cfg["width"], cfg["depth"]), (100, 3))
            self.assertEqual((cfg["Ncol_pts"], cfg["Nbnd_pts"], cfg["Ntest_pts"]), (12000, 400, 100000))
            self.assertTrue(cfg["include_zero_mode"])
            self.assertTrue(cfg["scale_to_period"])
            self.assertEqual(cfg["dim1_max"], cfg["dim2_max"])
            self.assertNotIn("input_norm_type", cfg)
            self.assertNotIn("normalize_fourier_basis", cfg)
            self.assertEqual(cfg["point_sampling"], "random")
            self.assertNotIn("lstsq_initial", cfg)
            self.assertNotIn("lstsq_column_scaling", cfg)
            self.assertEqual(cfg["lstsq_stop_epoch"], 20000)

    def test_default_dry_sweep(self):
        res = self.run_script()
        self.assertEqual(res.returncode, 0, res.stderr)
        commands = [shlex.split(line) for line in res.stdout.splitlines()]
        self.assertEqual(len(commands), 10)
        self.assertEqual(len({next(x for x in c if x.startswith("--datadir=")) for c in commands}), 10)
        self.assertEqual(basis_columns(200), 160801)
        self.assertIn("160801", res.stderr)

    def test_large_requests_fail_before_launch(self):
        res = self.run_script("--run", "--binary=/nonexistent/binary")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--allow-large-basis", res.stderr)
        self.assertNotIn("FileNotFoundError", res.stderr)

    def test_smoke_and_overrides(self):
        res = self.run_script("--smoke", "--lstsq-rcond=1e-4", "--lstsq-stop-epoch=2", "--test-batch-size=7")
        self.assertEqual(res.returncode, 0, res.stderr)
        commands = [shlex.split(line) for line in res.stdout.splitlines()]
        self.assertEqual(len(commands), 2)
        for cmd in commands:
            self.assertIn("--dim=2", cmd)
            self.assertIn("--ell=2", cmd)
            self.assertIn("--test_batch_size=7", cmd)
            self.assertIn("--n_lstsq_upds=10", cmd)
            self.assertEqual([x for x in cmd if x.startswith("--lstsq_stop_epoch=")][-1], "--lstsq_stop_epoch=2")

    def test_invalid_parameters(self):
        for argument in ("--ks=0", "--lstsq-rcond=nan", "--lstsq-rcond=-2", "--lstsq-stop-epoch=-2",
                         "--test-batch-size=0", "--test-batch-size=-1"):
            self.assertNotEqual(self.run_script(argument).returncode, 0)

    def test_nonfinite_logs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "log.txt"
            p.write_text("Saving Arg CSV line: final_error=0.1, final_loss=0.2")
            verify_run(p)
            for text in ("", "Saving Arg CSV line: final_error=nan, final_loss=0.2"):
                p.write_text(text)
                with self.assertRaises(RuntimeError):
                    verify_run(p)


if __name__ == "__main__":
    unittest.main()
