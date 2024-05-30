### Content of `multiplication_a_b.py`:


def multiply(a, b):
    try:
        return float(a * b)
    except Exception:
        return "error"


if __name__ == "__main__":
    # Example usage
    print(multiply(4, 5))
    print(multiply(4, 5))
    print(multiply("a", 3))
