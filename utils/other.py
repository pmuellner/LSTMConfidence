import random

def uniform(low, high, n):
    return [random.uniform(low, high) for _ in range(n)]