from test import test, enable_testing

@test
def func():
    assert 1 + 2 == 3

@test
def func2():
    assert 1 + 2 == 4

if __name__ == "__main__":
    enable_testing()
