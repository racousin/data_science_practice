"""
Generate Math Problem Dataset for Module 8 Exercise 3

This script generates 1000 math problems across multiple categories:
- Arithmetic (addition, subtraction, multiplication, division)
- Algebraic equations
- Geometry (area, perimeter, volume)
- Percentage problems
- Fraction operations
- Word problems

All solutions are numeric (integers or floats with 2 decimal precision).
"""

import random
import pandas as pd
import numpy as np
from typing import Tuple, List
import math

random.seed(42)
np.random.seed(42)


class MathProblemGenerator:
    """Generate various types of math problems with numeric solutions."""

    def __init__(self):
        self.categories = {
            'arithmetic': self.generate_arithmetic,
            'algebra': self.generate_algebra,
            'geometry': self.generate_geometry,
            'percentage': self.generate_percentage,
            'fractions': self.generate_fractions,
            'word_problems': self.generate_word_problem,
        }

    def generate_arithmetic(self) -> Tuple[str, float]:
        """Generate basic arithmetic problems."""
        ops = [
            ('addition', lambda a, b: (f"What is {a} + {b}?", a + b)),
            ('subtraction', lambda a, b: (f"Calculate {a} - {b}", a - b)),
            ('multiplication', lambda a, b: (f"What is {a} × {b}?", a * b)),
            ('division', lambda a, b: (f"Divide {a} by {b}", round(a / b, 2))),
            ('mixed', lambda a, b, c: (f"Calculate {a} + {b} - {c}", a + b - c)),
            ('order', lambda a, b, c: (f"What is ({a} + {b}) × {c}?", (a + b) * c)),
        ]

        op_type, op_func = random.choice(ops)

        if op_type == 'division':
            b = random.randint(2, 20)
            a = b * random.randint(1, 50)  # Ensure clean division
            return op_func(a, b)
        elif op_type in ['mixed', 'order']:
            a, b, c = random.randint(1, 50), random.randint(1, 50), random.randint(2, 10)
            return op_func(a, b, c)
        else:
            a, b = random.randint(1, 100), random.randint(1, 100)
            return op_func(a, b)

    def generate_algebra(self) -> Tuple[str, float]:
        """Generate simple algebraic equation problems."""
        templates = [
            # Linear equations: ax + b = c
            lambda: self._linear_equation(),
            # Simple quadratic
            lambda: self._simple_quadratic(),
            # Two-variable system (but ask for one variable)
            lambda: self._two_variable_system(),
        ]

        return random.choice(templates)()

    def _linear_equation(self) -> Tuple[str, float]:
        """Generate linear equation: ax + b = c, solve for x."""
        a = random.randint(2, 10)
        b = random.randint(-20, 20)
        x = random.randint(-10, 10)
        c = a * x + b

        problem = f"Solve for x: {a}x + {b} = {c}"
        return problem, float(x)

    def _simple_quadratic(self) -> Tuple[str, float]:
        """Generate x^2 = a, solve for positive x."""
        x = random.randint(1, 15)
        a = x * x

        problem = f"Find the positive value of x where x² = {a}"
        return problem, float(x)

    def _two_variable_system(self) -> Tuple[str, float]:
        """Generate system of equations, ask for one variable."""
        x = random.randint(1, 10)
        y = random.randint(1, 10)

        # x + y = sum
        # x - y = diff
        sum_xy = x + y
        diff_xy = x - y

        problem = f"If x + y = {sum_xy} and x - y = {diff_xy}, what is x?"
        return problem, float(x)

    def generate_geometry(self) -> Tuple[str, float]:
        """Generate geometry problems (area, perimeter, volume)."""
        shapes = [
            ('rectangle_area', lambda w, h: (
                f"What is the area of a rectangle with width {w} and height {h}?",
                w * h
            )),
            ('rectangle_perimeter', lambda w, h: (
                f"What is the perimeter of a rectangle with width {w} and height {h}?",
                2 * (w + h)
            )),
            ('triangle_area', lambda b, h: (
                f"What is the area of a triangle with base {b} and height {h}?",
                round(0.5 * b * h, 2)
            )),
            ('circle_area', lambda r: (
                f"What is the area of a circle with radius {r}? (use π ≈ 3.14)",
                round(3.14 * r * r, 2)
            )),
            ('circle_circumference', lambda r: (
                f"What is the circumference of a circle with radius {r}? (use π ≈ 3.14)",
                round(2 * 3.14 * r, 2)
            )),
            ('cube_volume', lambda s: (
                f"What is the volume of a cube with side length {s}?",
                s ** 3
            )),
            ('rectangular_prism', lambda l, w, h: (
                f"What is the volume of a rectangular prism with length {l}, width {w}, and height {h}?",
                l * w * h
            )),
        ]

        shape_type, shape_func = random.choice(shapes)

        if shape_type in ['rectangle_area', 'rectangle_perimeter']:
            w, h = random.randint(1, 30), random.randint(1, 30)
            return shape_func(w, h)
        elif shape_type == 'triangle_area':
            b, h = random.randint(2, 40), random.randint(2, 40)
            return shape_func(b, h)
        elif shape_type in ['circle_area', 'circle_circumference']:
            r = random.randint(1, 20)
            return shape_func(r)
        elif shape_type == 'cube_volume':
            s = random.randint(1, 10)
            return shape_func(s)
        else:  # rectangular_prism
            l, w, h = random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)
            return shape_func(l, w, h)

    def generate_percentage(self) -> Tuple[str, float]:
        """Generate percentage problems."""
        templates = [
            lambda: self._percent_of_number(),
            lambda: self._percentage_increase(),
            lambda: self._percentage_decrease(),
            lambda: self._find_percentage(),
        ]

        return random.choice(templates)()

    def _percent_of_number(self) -> Tuple[str, float]:
        """What is X% of Y?"""
        percent = random.randint(1, 100)
        number = random.randint(10, 500)
        result = round(percent * number / 100, 2)

        problem = f"What is {percent}% of {number}?"
        return problem, result

    def _percentage_increase(self) -> Tuple[str, float]:
        """Increase X by Y%"""
        number = random.randint(20, 200)
        percent = random.choice([10, 20, 25, 50, 75])
        result = round(number * (1 + percent / 100), 2)

        problem = f"Increase {number} by {percent}%"
        return problem, result

    def _percentage_decrease(self) -> Tuple[str, float]:
        """Decrease X by Y%"""
        number = random.randint(20, 200)
        percent = random.choice([10, 20, 25, 50])
        result = round(number * (1 - percent / 100), 2)

        problem = f"Decrease {number} by {percent}%"
        return problem, result

    def _find_percentage(self) -> Tuple[str, float]:
        """X is what percent of Y?"""
        percent = random.randint(10, 90)
        whole = random.randint(50, 500)
        part = round(whole * percent / 100)

        problem = f"{part} is what percent of {whole}?"
        return problem, round(percent, 2)

    def generate_fractions(self) -> Tuple[str, float]:
        """Generate fraction operation problems."""
        templates = [
            lambda: self._add_fractions(),
            lambda: self._subtract_fractions(),
            lambda: self._multiply_fractions(),
            lambda: self._fraction_of_number(),
        ]

        return random.choice(templates)()

    def _add_fractions(self) -> Tuple[str, float]:
        """Add two fractions with same denominator."""
        denom = random.choice([2, 3, 4, 5, 10])
        num1 = random.randint(1, denom - 1)
        num2 = random.randint(1, denom - 1)
        result = round((num1 + num2) / denom, 2)

        problem = f"What is {num1}/{denom} + {num2}/{denom}? (decimal form)"
        return problem, result

    def _subtract_fractions(self) -> Tuple[str, float]:
        """Subtract fractions with same denominator."""
        denom = random.choice([2, 4, 5, 10])
        num1 = random.randint(denom, denom * 3)
        num2 = random.randint(1, num1 - 1)
        result = round((num1 - num2) / denom, 2)

        problem = f"What is {num1}/{denom} - {num2}/{denom}? (decimal form)"
        return problem, result

    def _multiply_fractions(self) -> Tuple[str, float]:
        """Multiply a fraction by a whole number."""
        denom = random.choice([2, 4, 5, 10])
        num = random.randint(1, denom - 1)
        whole = random.randint(2, 10)
        result = round((num * whole) / denom, 2)

        problem = f"What is {num}/{denom} × {whole}? (decimal form)"
        return problem, result

    def _fraction_of_number(self) -> Tuple[str, float]:
        """Find fraction of a number."""
        denom = random.choice([2, 4, 5, 10])
        num = random.randint(1, denom - 1)
        number = random.randint(10, 200)
        result = round((num * number) / denom, 2)

        problem = f"What is {num}/{denom} of {number}?"
        return problem, result

    def generate_word_problem(self) -> Tuple[str, float]:
        """Generate word problems."""
        templates = [
            lambda: self._shopping_problem(),
            lambda: self._distance_problem(),
            lambda: self._age_problem(),
            lambda: self._money_problem(),
            lambda: self._time_problem(),
        ]

        return random.choice(templates)()

    def _shopping_problem(self) -> Tuple[str, float]:
        """Shopping/purchase problems."""
        items = ['apples', 'books', 'pens', 'notebooks', 'candies']
        item = random.choice(items)
        quantity = random.randint(3, 20)
        price = round(random.uniform(0.5, 10), 2)
        total = round(quantity * price, 2)

        problem = f"If one {item[:-1]} costs ${price}, how much do {quantity} {item} cost?"
        return problem, total

    def _distance_problem(self) -> Tuple[str, float]:
        """Distance/speed problems."""
        speed = random.randint(30, 80)
        time = random.randint(2, 8)
        distance = speed * time

        problem = f"A car travels at {speed} km/h for {time} hours. What distance does it cover?"
        return problem, float(distance)

    def _age_problem(self) -> Tuple[str, float]:
        """Age-related problems."""
        current_age = random.randint(10, 40)
        years_ago = random.randint(5, 15)
        past_age = current_age - years_ago

        problem = f"John is {current_age} years old now. How old was he {years_ago} years ago?"
        return problem, float(past_age)

    def _money_problem(self) -> Tuple[str, float]:
        """Money calculation problems."""
        initial = random.randint(50, 500)
        spent = random.randint(10, initial - 10)
        remaining = initial - spent

        problem = f"Sarah has ${initial}. She spends ${spent}. How much money does she have left?"
        return problem, float(remaining)

    def _time_problem(self) -> Tuple[str, float]:
        """Time calculation problems."""
        start_hour = random.randint(8, 14)
        duration = random.randint(2, 8)
        end_hour = start_hour + duration

        problem = f"A meeting starts at {start_hour}:00 and lasts {duration} hours. At what hour does it end? (24-hour format)"
        return problem, float(end_hour)

    def generate_problem(self, category: str = None) -> Tuple[str, str, float]:
        """
        Generate a single math problem.

        Returns:
            tuple: (category, problem_text, solution)
        """
        if category is None:
            category = random.choice(list(self.categories.keys()))

        problem_text, solution = self.categories[category]()

        # Round solution to 2 decimal places
        solution = round(solution, 2)

        return category, problem_text, solution


def generate_dataset(n_problems: int = 1000, train_split: float = 0.9) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate a dataset of math problems.

    Args:
        n_problems: Total number of problems to generate
        train_split: Proportion of data for training (default 0.9)

    Returns:
        tuple: (train_df, test_df, test_target_df)
    """
    generator = MathProblemGenerator()

    # Generate problems with balanced categories
    categories = list(generator.categories.keys())
    n_per_category = n_problems // len(categories)
    remainder = n_problems % len(categories)

    problems = []

    # Generate balanced problems
    for category in categories:
        n_to_generate = n_per_category + (1 if remainder > 0 else 0)
        remainder -= 1

        for _ in range(n_to_generate):
            cat, problem, solution = generator.generate_problem(category)
            problems.append({
                'category': cat,
                'problem': problem,
                'solution': solution
            })

    # Shuffle the problems
    random.shuffle(problems)

    # Create DataFrame
    df = pd.DataFrame(problems)

    # Split into train and test
    n_train = int(n_problems * train_split)

    train_df = df[:n_train].copy()
    test_df = df[n_train:].copy()

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Create test target file (for evaluation)
    test_target_df = test_df[['solution']].copy()
    test_target_df.insert(0, 'id', range(len(test_target_df)))

    # Add id column to test_df and remove solution
    test_df.insert(0, 'id', range(len(test_df)))
    test_display_df = test_df[['id', 'category', 'problem']].copy()

    # Add id column to train_df
    train_df.insert(0, 'id', range(len(train_df)))

    return train_df, test_display_df, test_target_df


def main():
    """Generate and save the math dataset."""
    print("Generating math problem dataset...")

    # Generate 1000 problems (900 train, 100 test)
    train_df, test_df, test_target_df = generate_dataset(n_problems=1000, train_split=0.9)

    print(f"Generated {len(train_df)} training problems")
    print(f"Generated {len(test_df)} test problems")

    # Print category distribution
    print("\nTraining set category distribution:")
    print(train_df['category'].value_counts().sort_index())

    print("\nTest set category distribution:")
    print(test_df['category'].value_counts().sort_index())

    # Save datasets
    output_dir = "website/public/modules/data-science-practice/module8/exercise"

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    test_target_df.to_csv(f"{output_dir}/test_target.csv", index=False)

    print(f"\nDatasets saved to {output_dir}/")
    print("- train.csv (with solutions)")
    print("- test.csv (without solutions)")
    print("- test_target.csv (ground truth for evaluation)")

    # Display samples
    print("\nSample training problems:")
    print(train_df.head(10).to_string(index=False))

    print("\nSample test problems:")
    print(test_df.head(5).to_string(index=False))

    print("\nSample test targets:")
    print(test_target_df.head(5).to_string(index=False))

    # Statistics
    print("\nDataset statistics:")
    print(f"Solution range (train): [{train_df['solution'].min():.2f}, {train_df['solution'].max():.2f}]")
    print(f"Solution mean (train): {train_df['solution'].mean():.2f}")
    print(f"Solution median (train): {train_df['solution'].median():.2f}")


if __name__ == "__main__":
    main()
