import threading
from typing import Optional, Any, List, Dict, Tuple


class RingBuffer:
    def __init__(self, cap: int = 1024):
        self.cap = cap
        self.end = 0
        self.mutex = threading.Lock()
        self.buf = [None] * cap

        self.id_start = {'default': 0}

    def reset(self) -> Optional[Exception]:
        with self.mutex:
            self.end = 0
            self.id_start = {'default': 0}
            return None

    def push(self, item: Any) -> Optional[Exception]:
        """
        Push an item into the buffer.
        """
        with self.mutex:
            self.buf[self.end % self.cap] = item
            self.end += 1
            return None

    def peek(self, index: int = -1) -> Tuple[Optional[Any], int, Optional[Exception]]:
        """
        Peek at the item at the given index.
        """
        if index >= self.end:
            return None, -1, Exception("index out of range")
        if index < self.end - self.cap:
            # won't happen if called from pull()
            return None, -1, Exception("index out of range")
        place = index % self.cap
        return self.buf[place], index, None

    def pull(self, client_id: str = 'default') -> Tuple[List[Any], Optional[Exception]]:
        """
        Return all unread items in the buffer.
        """
        if client_id not in self.id_start.keys():
            self.id_start[client_id] = 0

        with self.mutex:
            if self.id_start[client_id] >= self.end:
                return [], None
            if self.id_start[client_id] < self.end - self.cap:
                self.id_start[client_id] = self.end - self.cap
            ret = []
            for idx in range(self.id_start[client_id], self.end):
                item, _, err = self.peek(idx)
                if err is not None:
                    return [], err
                ret.append(item)
            self.id_start[client_id] = self.end
            return ret, None

    def get_valid_len(self, client_id: str = 'default') -> int:
        if client_id not in self.id_start.keys():
            self.id_start[client_id] = 0
        with self.mutex:
            return self.end - max(self.end - self.cap, self.id_start[client_id])

    def get_cap(self) -> int:
        return self.cap

    def get_end(self) -> int:
        return self.end
