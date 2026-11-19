"""Lightweight configuration and dry-run checks; no training is launched."""

import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from run import verify_run


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class ReproductionTests(unittest.TestCase):
    def test_final_metrics_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            for metrics in ("final_error=0.01, final_loss=0.02",):
                log.write_text("Saving Arg CSV line: " + metrics)
                verify_run(log)
            for text in ("", "Saving Arg CSV line: final_error=-nan, final_loss=-nan",
                         "Saving Arg CSV line: final_error=0.1, final_loss=inf"):
                log.write_text(text)
                with self.assertRaises(RuntimeError):
                    verify_run(log)

    def commands(self, *args):
        result = subprocess.run([sys.executable, str(HERE / "run.py"), *args],
                                cwd=ROOT, check=True, capture_output=True, text=True)
        return [shlex.split(line) for line in result.stdout.splitlines()]

    def test_configs(self):
        for case, selector in (("single_scale", "test1_k100"),
                               ("multiscale", "test2"), ("product", "test3")):
            with self.subTest(case=case):
                cfg = json.loads((HERE / "configs" / (case + ".json")).read_text())
                self.assertEqual(cfg["pde_problem"], "allencahn")
                self.assertEqual(cfg["test_func"], selector)
                self.assertEqual(cfg["Ntest_pts"], 20000)
                self.assertNotIn("input_norm_type", cfg)
                self.assertNotIn("normalize_fourier_basis", cfg)
                self.assertNotIn("lstsq_initial", cfg)
                self.assertNotIn("lstsq_column_scaling", cfg)
                self.assertTrue(cfg["lstsq_safeguard"])
                self.assertEqual(cfg["lstsq_rcond"], -1 if case == "single_scale" else 1e-4)
                self.assertEqual(cfg["lstsq_stop_epoch"], 20000)
                blocks = min(cfg["adam_max_epochs"], cfg["lstsq_stop_epoch"]) // cfg["lstsq_upd_interval"]
                self.assertEqual(blocks * cfg["n_lstsq_upds_per_epoch"], 100)
                self.assertGreaterEqual(cfg["n_lstsq_upds"], 100)

    def test_single_command(self):
        commands = self.commands("--cases", "multiscale", "--ks", "200", "--seeds", "1234")
        self.assertEqual(len(commands), 1)
        self.assertIn("--config=experiments/section_6_3_2/configs/multiscale.json", commands[0])
        self.assertIn("--expid=s632_multiscale_K200_s1234", commands[0])

    def test_smoke_budget(self):
        commands = self.commands("--smoke")
        self.assertEqual(len(commands), 3)
        for command in commands:
            self.assertIn("--n_lstsq_upds=10", command)
            self.assertIn("--adam_max_epochs=4", command)
            self.assertIn("--lstsq_upd_interval=2", command)
            self.assertIn("--lstsq_stop_epoch=4", command)

    def test_default_sweep(self):
        commands = self.commands()
        self.assertEqual(len(commands), 75)
        outputs = [next(arg for arg in command if arg.startswith("--datadir="))
                   for command in commands]
        self.assertEqual(len(set(outputs)), 75)


if __name__ == "__main__":
    unittest.main()
