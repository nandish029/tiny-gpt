import unittest
import os

class TestStaleReferences(unittest.TestCase):
    """
    Scans the repository to ensure no stale references like `valid_data` or `valid.jsonl`
    were accidentally reintroduced.
    """
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.forbidden_strings = [
            'split="valid"',
            "split='valid'",
            'valid.jsonl',
            'valid_data'
        ]
        
    def test_no_stale_references(self):
        directories_to_check = ['src', 'scripts', 'tests']
        
        found_stale = []
        
        for directory in directories_to_check:
            dir_path = os.path.join(self.project_root, directory)
            for root, dirs, files in os.walk(dir_path):
                if '__pycache__' in dirs:
                    dirs.remove('__pycache__')
                for file in files:
                    if not file.endswith('.py'):
                        continue
                        
                    # Skip this test file itself, since it contains the forbidden strings!
                    if file == "test_stale_references.py":
                        continue
                        
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        try:
                            lines = f.readlines()
                        except UnicodeDecodeError:
                            continue
                        
                    for i, line in enumerate(lines):
                        for forbidden in self.forbidden_strings:
                            if forbidden in line:
                                found_stale.append(f"{filepath}:{i+1} -> {forbidden}")
                                
        if found_stale:
            self.fail("Found stale references in codebase:\n" + "\n".join(found_stale))

if __name__ == "__main__":
    unittest.main()
