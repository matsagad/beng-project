from utils.data import ClassWithData
import os


class TestWithData(ClassWithData):
    def __init__(self):
        super().__init__()

    def run_all_tests(self):
        terminal_size = os.get_terminal_size()
        border = "\u2500" * terminal_size.columns
        print(
            f"{border}\nRunning all tests from \033[1m{self.__class__.__name__}\033[0m."
        )
        passed, total = 0, 0
        for fn_name in dir(self):
            test_fn = getattr(self, fn_name)
            if fn_name.startswith("test") and hasattr(test_fn, "__call__"):
                try:
                    total += 1
                    print(f"\n\033[1m{fn_name}:\033[0m")
                    test_fn()
                    print("\033[92mPASSED\033[0m")
                    passed += 1
                except AssertionError as e:
                    print(f"Error: {e if e else '(no message set)'}")
                    print("\033[91mFAILED\033[0m")
        print(f"\nCompleted: {passed}/{total} tests passed.\n{border}")
