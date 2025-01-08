from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
    Union,
    TypeVar,
    Type,
    cast,
    overload,
    Generic,
)
from dataclasses import dataclass

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class Keywords:
    """Container for keyword arguments."""

    kwargs: Dict[str, Any]


class Here:
    """Placeholder token for pipe argument substitution."""

    def __init__(self, func: Callable[[Any], Any] = lambda x: x):
        self.func = func


class Unpipe:
    """Special type to signal value extraction at the end of pipe chain."""

    pass


def typed_unpipe_v0_1(target_type: Type[T]) -> Callable[[Any], T]:
    """Return a function that casts input to the specified type."""

    def _unpipe(value: Any) -> T:
        return cast(T, value)

    return _unpipe


class UnpipeAs(Generic[T]):
    """Special type to signal value extraction at the end of pipe chain with specific type."""

    def __init__(self, type_: Type[T]):
        self.type = type_


class Pipe:
    """
    A class implementing the pipe operator (|) for function composition.

    Allows for elegant function chaining with support for positional and keyword arguments,
    and special placeholder substitution using the `here` token.
    """

    def __init__(self, value: Any):
        """Initialize pipe with a value."""
        self._value = value

    @property
    def value(self) -> Any:
        """Get the current value in the pipe."""
        return self._value

    @staticmethod
    def _ensure_callable(func: Any) -> None:
        """Validate that the provided object is callable."""
        if not callable(func):
            raise TypeError(f"Expected a callable, got {type(func).__name__}")

    def _substitute_here(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Substitute 'here' placeholders with the current pipe value.

        Returns:
            Tuple of (modified args, modified kwargs)
        """
        modified_args = list(args)
        modified_kwargs = kwargs.copy()

        # Check for 'here' in positional arguments
        here_in_args = False
        for i, arg in enumerate(modified_args):
            if isinstance(arg, Here):
                modified_args[i] = arg.func(self._value)
                here_in_args = True

        # Check for 'here' in keyword arguments
        for k, v in modified_kwargs.items():
            if isinstance(v, Here):
                modified_kwargs[k] = v.func(self._value)
                here_in_args = True

        # If 'here' wasn't found anywhere, prepend value to args
        if not here_in_args:
            modified_args.insert(0, self._value)

        return tuple(modified_args), modified_kwargs

    @overload
    def __or__(self, other: UnpipeAs[T]) -> T: ...

    @overload
    def __or__(self, other: Union[Callable, Tuple, Unpipe]) -> Union["Pipe", Any]: ...

    def __or__(
        self, other: Union[Callable, Tuple, Unpipe, UnpipeAs[T]]
    ) -> Union["Pipe", Any, T]:
        """
        Implementation of the pipe operator (|).

        Args:
            other: Either a callable, a tuple of (callable, *args, Keywords?), Unpipe, or UnpipeAs

        Returns:
            A new Pipe instance with the result of the operation or the unwrapped value
        """
        if isinstance(other, Unpipe):
            return self._value

        if isinstance(other, UnpipeAs):
            if not isinstance(self._value, other.type):
                raise TypeError(
                    f"Expected pipe value to be {other.type}, got {type(self._value)}"
                )
            return cast(other.type, self._value)

        if isinstance(other, tuple):
            func, *rest = other
            self._ensure_callable(func)

            # Handle keyword arguments if present
            if rest and isinstance(rest[-1], Keywords):
                *args, kw = rest
                args, kwargs = self._substitute_here(tuple(args), kw.kwargs)
            else:
                args, kwargs = self._substitute_here(tuple(rest), {})

            return Pipe(func(*args, **kwargs))

        self._ensure_callable(other)
        return Pipe(other(self._value))

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"Pipe({repr(self._value)})"


def kw(**kwargs) -> Keywords:
    """Convenience function to create Keywords instances."""
    return Keywords(kwargs)


class PipeStart:
    """
    Special class to handle the initial value of a pipe chain
    when using the empty pipe() function.
    """

    def __or__(self, other: Any) -> Pipe:
        return Pipe(other)


pipe = PipeStart()

here = Here()

unpipe = Unpipe()

# Example usage:
if __name__ == "__main__":
    import sys
    from io import StringIO

    # Basic function composition
    def add(x: int) -> int:
        return x + 10

    def multiply(x: int, y: int) -> int:
        return x * y

    # Example pipelines
    result = Pipe(5) | add | (multiply, 2) | str | print

    # Using keywords and here placeholder
    numbers = [3, 1, 4, 1, 5, 9]
    (Pipe(numbers) | (sorted, kw(reverse=True)) | print)

    # Multiple here placeholders
    def sum_three(x: int, y: int, z: int) -> int:
        return x + y + z

    (Pipe(1) | (sum_three, here, here, here) | print)  # Outputs: 3

    from typing import List

    nums = [1, 2, 3, 4, 5, 10, 20, 23, 25, 27]
    filtered = (
        pipe
        | nums
        | (filter, lambda x: x % 2 == 0, here)
        | (filter, lambda x: x <= 10, here)
        | list
        | UnpipeAs(List)  # or just unpipe
    )
    print(filtered)  # Outputs: [2, 4, 10]

    sys.stdin = StringIO("1 2 3 5")
    (
        pipe
        | input().split()
        | (map, int, here)
        | list
        | (sorted, kw(reverse=True))
        | print
    )

    sys.stdin = StringIO("1 2 3 5")
    (
        pipe
        | input().split()
        | (map, int, Here(lambda x: [y + "1" for y in x]))
        | list
        | (sorted, kw(reverse=True))
        | print
    )

    # same as:
    # print(sorted(list(map(int, input().split()))), reverse=True))
