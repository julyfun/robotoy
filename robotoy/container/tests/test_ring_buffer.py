import unittest
import numpy as np

import sys
import os

from robotoy.container.ring_buffer import RingBuffer


class TestRingBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = RingBuffer(5)

    def test_push_and_peek(self):
        data = np.array([1.0, 2.0, 3.0])
        self.buffer.push(data)
        result = self.buffer.peek(0)
        np.testing.assert_array_equal(result, data)

    def test_pull(self):
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([4.0, 5.0, 6.0])
        self.buffer.push(data1)
        self.buffer.push(data2)
        result = self.buffer.pull()
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], data1)
        np.testing.assert_array_equal(result[1], data2)

    def test_out_of_cap_with_lots_of_data(self):
        data = np.array([1.0, 2.0, 3.0])
        for i in range(10):
            self.buffer.push(np.copy(data * i))
        result = self.buffer.pull()
        self.assertEqual(len(result), 5)
        np.testing.assert_array_equal(result[4], data * 9)
        self.buffer.push(np.copy(data * 2))
        np.testing.assert_array_equal(self.buffer.pull()[0], data * 2)

    def test_pop_front(self):
        data = np.array([1.0, 2.0, 3.0])
        self.buffer.push(data * 1)
        self.buffer.push(data * 2)
        self.buffer.push(data * 3)
        self.buffer.push(data * 4)
        self.buffer.push(data * 5)
        arr = self.buffer.pop_front()
        np.testing.assert_array_equal(arr, data * 1)
        np.testing.assert_array_equal(self.buffer.pop_front(), data * 2)
        np.testing.assert_array_equal(self.buffer.pull()[0], data * 3)
        np.testing.assert_array_equal(self.buffer.pop_front(), [])

    def test_reset(self):
        data = np.array([1.0, 2.0, 3.0])
        self.buffer.push(data)
        self.buffer.reset()
        result = self.buffer.pull()
        self.assertEqual(len(result), 0)

    def test_get_valid_len(self):
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([4.0, 5.0, 6.0])
        self.buffer.push(data1)
        self.buffer.push(data2)
        valid_len = self.buffer.get_valid_len()
        self.assertEqual(valid_len, 2)

    def test_get_cap_and_end(self):
        self.assertEqual(self.buffer.get_cap(), 5)
        self.assertEqual(self.buffer.get_end(), 0)
        data = np.array([1.0, 2.0, 3.0])
        self.buffer.push(data)
        self.assertEqual(self.buffer.get_end(), 1)


if __name__ == "__main__":
    unittest.main()
