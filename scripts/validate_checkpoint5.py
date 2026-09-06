import sys
import os
import unittest

def run_tests_and_report():
    print("CHECKPOINT 5 VALIDATION")
    print("-----------------------")
    
    loader = unittest.TestLoader()
    
    try:
        import tests.test_validation_checkpoint5 as tc5
    except ImportError as e:
        print(f"Failed to import validation tests: {e}")
        print("\nFINAL RESULT: FAIL")
        sys.exit(1)
        
    test_cases = [
        ("Independent reference checks", tc5.TestIndependentReference),
        ("Randomized property checks", tc5.TestRandomizedProperties),
        ("Boundary checks", tc5.TestBoundaryConditions),
        ("Train/validation separation", tc5.TestTrainValidationSeparation),
        ("Device checks", tc5.TestDeviceBehavior)
    ]
    
    all_passed = True
    with open(os.devnull, 'w') as devnull:
        for name, test_cls in test_cases:
            suite = loader.loadTestsFromTestCase(test_cls)
            runner = unittest.TextTestRunner(stream=devnull, verbosity=0)
            result = runner.run(suite)
            
            if result.wasSuccessful():
                print(f"{name}: PASS")
            else:
                print(f"{name}: FAIL")
                all_passed = False
            
    if all_passed:
        print("\nFINAL RESULT: PASS")
        sys.exit(0)
    else:
        print("\nFINAL RESULT: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure current directory is in path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    run_tests_and_report()
