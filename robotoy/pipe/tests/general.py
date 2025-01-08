import unittest
from typing import List
from robotoy.pipe import Pipe, PipeStart, Here, kw, Unpipe, UnpipeAs, pipe


class TestPipe(unittest.TestCase):
    def test_add_multiply(self):
        def add(x: int) -> int:
            return x + 10

        def multiply(x: int, y: int) -> int:
            return x * y

        result = Pipe(5) | add | (multiply, 2)
        self.assertEqual(result.value, 30)

    def test_keywords_and_sorted(self):
        numbers = [3, 1, 4, 1, 5, 9]
        result = Pipe(numbers) | (sorted, kw(reverse=True))
        self.assertEqual(result.value, [9, 5, 4, 3, 1, 1])

    def test_sum_three(self):
        def sum_three(x: int, y: int, z: int) -> int:
            return x + y + z

        result = Pipe(1) | (sum_three, Here(), Here(), Here())
        self.assertEqual(result.value, 3)

    def test_unpipeas(self):
        nums = [1, 2, 3, 4, 5, 10, 20, 23, 25, 27]
        filtered = (
            pipe
            | nums
            | (filter, lambda x: x % 2 == 0, Here())
            | (filter, lambda x: x <= 10, Here())
            | list
            | UnpipeAs(List)
        )
        self.assertEqual(filtered, [2, 4, 10])


if __name__ == "__main__":
    unittest.main()
