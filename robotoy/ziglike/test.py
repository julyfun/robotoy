import argparse
from functools import wraps
import sys

class TestRunner:
    _instance = None
    _tests = []
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.parser = argparse.ArgumentParser(add_help=False)
            cls._instance.parser.add_argument("--test", action="store_true", help="Run tests")
            cls._instance.parser.add_argument("--test-filter", type=str, help="Run only tests matching filter")
        return cls._instance
    
    def register_test(self, func):
        self._tests.append(func)
        return func
    
    def run_tests(self):
        args, _ = self.parser.parse_known_args()
        if not args.test:
            return
        
        print("\n=== Running Tests ===")
        tests_to_run = self._tests
        if args.test_filter:
            tests_to_run = [t for t in tests_to_run if args.test_filter.lower() in t.__name__.lower()]
        
        for test_func in tests_to_run:
            print(f"\nRunning {test_func.__name__}...")
            try:
                test_func()
                print(f"✓ {test_func.__name__} passed")
            except AssertionError as e:
                print(f"✗ {test_func.__name__} failed: {str(e)}")
        
        print("\n=== Tests Complete ===")
        sys.exit(0)

_runner = TestRunner()

def test(func):
    """Decorator to register a test function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return _runner.register_test(wrapper)

def enable_testing():
    """Call this in your main script to enable test execution"""
    _runner.run_tests()

# [run]

def run(func):
    func()

