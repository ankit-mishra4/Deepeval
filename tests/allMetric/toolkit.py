
from datetime import datetime


def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


def get_current_time():
    now = datetime.now()
    return str(now)