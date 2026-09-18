from datetime import datetime


def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


def get_current_time():
    now = datetime.now()
    return str(now)


if __name__ == "__main__":
    print(add(5, 3))
    print(multiply(5, 3))
    print(get_current_time())
