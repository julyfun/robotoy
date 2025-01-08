from typing import (
    Callable,
    TypeVar,
    Generic,
    overload,
)
from typing_extensions import ParamSpec

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")


class Unpipe:
    pass


unpipe = Unpipe()


class Pipe(Generic[T]):
    def __init__(self, value: T):
        self.value = value

    @overload
    def __or__(self, func: Callable[P, R]) -> "Pipe[R]": ...

    @overload
    def __or__(self, func: Unpipe) -> T: ...

    def __or__(self, nxt):
        if isinstance(nxt, Unpipe):
            return self.value
        return Pipe(nxt(self.value))


if __name__ == "__main__":

    def add_one(value: int) -> int:
        return value + 1

    def multiply(value: int, y: int) -> int:
        return value * y

    x1 = (
        Pipe(input().split())
        | (lambda x: list(map(int, x)))
        | (lambda x: sorted(x, reverse=True))
        | (lambda x: x[0])
        | add_one
        | unpipe
    )
    print(x1)
