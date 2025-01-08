class Pipe:
    def __init__(self, value):
        self.value = value

    def __or__(self, func):
        self.value = func(self.value)
        return self


if __name__ == "__main__":

    def add_one(value):
        return value + 1

    def multiply(value, y):
        return value * y

    res = (
        Pipe(input().split())
        | (lambda x: map(int, x))
        | list
        | (lambda x: sorted(x, reverse=True))
    )

    print(res.value)
