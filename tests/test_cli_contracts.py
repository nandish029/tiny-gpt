import unittest
import subprocess
import os

class TestCLIContracts(unittest.TestCase):
    """
    Tests to ensure that the CLI contracts for each script match documentation
    and handle basic expected/unexpected argument behaviors.
    """
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
    def run_script(self, script_path, args):
        cmd = ["python", os.path.join(self.project_root, script_path)] + args
        return subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
    def test_train_gpt1_cli(self):
        # Missing required arguments isn't strictly enforced if defaults exist, but let's test --help
        res = self.run_script("scripts/train_gpt1.py", ["--help"])
        self.assertEqual(res.returncode, 0)
        self.assertIn("--config", res.stdout)
        self.assertIn("--device", res.stdout)
        
        # Test invalid device explicitly rejected by argparse
        res_invalid = self.run_script("scripts/train_gpt1.py", ["--device", "invalid_device"])
        self.assertNotEqual(res_invalid.returncode, 0)
        self.assertIn("invalid choice", res_invalid.stderr)
        
    def test_evaluate_gpt1_cli(self):
        res = self.run_script("scripts/evaluate_gpt1.py", ["--help"])
        self.assertEqual(res.returncode, 0)
        self.assertIn("--checkpoint", res.stdout)
        # Verify --device is NOT in evaluate_gpt1.py (it doesn't have it currently)
        self.assertNotIn("--device", res.stdout)
        
    def test_generate_gpt1_cli(self):
        # generate_gpt1.py has NO argparse currently.
        pass

    def test_chat_cli(self):
        res = self.run_script("scripts/chat.py", ["--help"])
        self.assertEqual(res.returncode, 0)
        self.assertIn("--checkpoint", res.stdout)
        self.assertIn("--prompt", res.stdout)
        
        # Test required arguments missing
        res_missing = self.run_script("scripts/chat.py", [])
        self.assertNotEqual(res_missing.returncode, 0)
        self.assertIn("required", res_missing.stderr.lower())
        
    def test_prepare_cli(self):
        res = self.run_script("src/data/prepare.py", ["--help"])
        self.assertEqual(res.returncode, 0)
        self.assertIn("--num_samples", res.stdout)
        self.assertIn("--seed", res.stdout)

if __name__ == "__main__":
    unittest.main()
