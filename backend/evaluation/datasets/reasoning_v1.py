"""Reasoning suite v1: deterministic integer-answer problems in four categories.

Every answer is computed by this generator (verified by construction and by tests), never
typed by hand. Splits: 10 screening + 40 held-out items per category. The held-out split must
not be regenerated or edited after results exist; a change requires a new version (v2).
"""
from __future__ import annotations

import math
import random

SEED = 20261109
PER_CATEGORY = {"screening": 10, "heldout": 40}
CATEGORIES = ("arithmetic", "multi_step", "programming_concepts", "algorithmic")
NAMES = ["Ava", "Ben", "Chen", "Dara", "Eli", "Fatima", "Gus", "Hana", "Ivan", "Jia", "Kofi", "Lena", "Mateo", "Nia", "Omar", "Priya"]
ITEMS_ = [("apples", "basket"), ("stickers", "sheet"), ("pencils", "box"), ("cookies", "tray"), ("marbles", "bag"), ("books", "shelf")]
PROMPT = ("{question}\nThink step by step, then give the final answer on the last line in the form "
          "\"Answer: <integer>\".")


def arithmetic(r: random.Random):
    kind = r.choice(["add", "sub", "mul", "mixed", "div"])
    if kind == "add":
        a, b = r.randint(100, 999), r.randint(100, 999)
        return f"What is {a} + {b}?", a + b
    if kind == "sub":
        a, b = r.randint(300, 999), r.randint(100, 299)
        return f"What is {a} - {b}?", a - b
    if kind == "mul":
        a, b = r.randint(12, 49), r.randint(3, 19)
        return f"What is {a} * {b}?", a * b
    if kind == "div":
        b, q = r.randint(3, 15), r.randint(6, 40)
        return f"What is {b * q} divided by {b}?", q
    a, b, c = r.randint(10, 40), r.randint(2, 9), r.randint(5, 60)
    return f"What is {a} * {b} - {c}?", a * b - c


def multi_step(r: random.Random):
    name = r.choice(NAMES)
    kind = r.choice(["change", "groups", "remaining", "rate", "share"])
    if kind == "change":
        price_a, price_b, n_a, n_b = r.randint(2, 9), r.randint(3, 12), r.randint(2, 6), r.randint(1, 4)
        total = price_a * n_a + price_b * n_b
        paid = (total // 10 + 1) * 10 + r.choice([0, 10])
        return (f"Pencils cost ${price_a} each and notebooks cost ${price_b} each. {name} buys {n_a} pencils and "
                f"{n_b} notebooks and pays with ${paid}. How many dollars of change does {name} receive?", paid - total)
    if kind == "groups":
        thing, holder = r.choice(ITEMS_)
        holders, per, extra = r.randint(3, 9), r.randint(4, 12), r.randint(1, 15)
        return (f"{name} has {holders} {holder}s with {per} {thing} in each {holder}, plus {extra} loose {thing}. "
                f"How many {thing} does {name} have in total?", holders * per + extra)
    if kind == "remaining":
        start, gave, bought = r.randint(30, 90), r.randint(5, 25), r.randint(3, 20)
        friend = r.choice([n for n in NAMES if n != name])
        return (f"{name} had {start} stickers, gave {gave} to {friend}, and then bought {bought} more. "
                f"After that, {name} gave half of the stickers to a sibling. How many stickers does {name} keep?"
                if (start - gave + bought) % 2 == 0 else
                f"{name} had {start} stickers, gave {gave} to {friend}, and then bought {bought} more. "
                f"How many stickers does {name} have now?",
                (start - gave + bought) // 2 if (start - gave + bought) % 2 == 0 else start - gave + bought)
    if kind == "rate":
        per_hour, hours, days = r.randint(3, 12), r.randint(2, 8), r.randint(2, 6)
        return (f"A machine makes {per_hour} parts per hour. It runs {hours} hours a day for {days} days. "
                f"How many parts does it make in total?", per_hour * hours * days)
    people, each, leftover = r.randint(3, 8), r.randint(4, 15), r.randint(0, 2)
    total = people * each + leftover
    return (f"{total} cookies are shared equally among {people} children, and any cookies that cannot be shared "
            f"equally are kept aside. How many cookies does each child get?", total // people)


def programming_concepts(r: random.Random):
    kind = r.choice(["floordiv", "mod", "len_str", "aug", "slice_len", "count", "neg_floordiv", "list_index", "set_len", "range_sum", "nested_len", "power"])
    if kind == "floordiv":
        a, b = r.randint(20, 99), r.randint(3, 9)
        return f"In Python, what is the value of {a} // {b}?", a // b
    if kind == "mod":
        a, b = r.randint(20, 99), r.randint(3, 9)
        return f"In Python, what is the value of {a} % {b}?", a % b
    if kind == "neg_floordiv":
        a, b = r.randint(7, 40), r.randint(2, 6)
        while a % b == 0:
            a += 1
        return f"In Python, what is the value of -{a} // {b}?", -a // b
    if kind == "len_str":
        word = r.choice(["feather", "quantize", "benchmark", "python", "inference", "token", "laptop", "gradient"])
        return f"In Python, what does len(\"{word}\") return?", len(word)
    if kind == "aug":
        x, a, m, s = r.randint(1, 9), r.randint(1, 9), r.randint(2, 5), r.randint(1, 9)
        return f"In Python, what is the value of x after running: x = {x}; x += {a}; x *= {m}; x -= {s}", (x + a) * m - s
    if kind == "slice_len":
        n, i, j = r.randint(8, 15), r.randint(1, 4), r.randint(5, 8)
        return f"In Python, what does len(list(range({n}))[{i}:{j}]) return?", len(list(range(n))[i:j])
    if kind == "count":
        word, letter = r.choice([("banana", "a"), ("mississippi", "s"), ("bookkeeper", "e"), ("committee", "t"), ("parallel", "l")])
        return f"In Python, what does \"{word}\".count(\"{letter}\") return?", word.count(letter)
    if kind == "list_index":
        values = r.sample(range(10, 99), 5)
        k = r.choice([-1, -2, 1, 3])
        return f"In Python, what is the value of {values}[{k}]?", values[k]
    if kind == "set_len":
        values = [r.randint(1, 6) for _ in range(9)]
        return f"In Python, what does len(set({values})) return?", len(set(values))
    if kind == "range_sum":
        a, b, step = r.randint(0, 5), r.randint(12, 30), r.choice([1, 2, 3])
        return f"In Python, what does sum(range({a}, {b}, {step})) return?", sum(range(a, b, step))
    if kind == "nested_len":
        values = [r.randint(1, 9), [r.randint(1, 9), r.randint(1, 9)], [r.randint(1, 9)], r.randint(1, 9)]
        return f"In Python, what does len({values}) return?", len(values)
    a, b = r.randint(2, 5), r.randint(2, 6)
    return f"In Python, what is the value of {a} ** {b}?", a ** b


def algorithmic(r: random.Random):
    kind = r.choice(["gcd", "lcm", "primes_below", "fib", "digit_sum", "popcount", "sorted_pos", "max_diff", "count_div", "binary_steps"])
    if kind == "gcd":
        g, a, b = r.randint(2, 12), r.randint(2, 15), r.randint(2, 15)
        return f"What is the greatest common divisor of {g * a} and {g * b}?", math.gcd(g * a, g * b)
    if kind == "lcm":
        a, b = r.randint(2, 15), r.randint(2, 15)
        return f"What is the least common multiple of {a} and {b}?", a * b // math.gcd(a, b)
    if kind == "primes_below":
        n = r.randint(10, 60)
        return f"How many prime numbers are less than {n}?", sum(all(k % d for d in range(2, math.isqrt(k) + 1)) for k in range(2, n))
    if kind == "fib":
        n = r.randint(6, 20)
        a, b = 1, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return f"In the Fibonacci sequence where F(1) = 1 and F(2) = 1, what is F({n})?", a
    if kind == "digit_sum":
        n = r.randint(1000, 999999)
        return f"What is the sum of the digits of {n}?", sum(map(int, str(n)))
    if kind == "popcount":
        n = r.randint(5, 255)
        return f"How many 1 bits are in the binary representation of {n}?", bin(n).count("1")
    if kind == "sorted_pos":
        values = r.sample(range(1, 99), 7)
        k = r.randint(2, 6)
        return f"If the list {values} is sorted in ascending order, what number is at position {k} (counting from 1)?", sorted(values)[k - 1]
    if kind == "max_diff":
        values = r.sample(range(1, 99), 6)
        return f"What is the difference between the largest and smallest numbers in {values}?", max(values) - min(values)
    if kind == "count_div":
        n, d = r.randint(30, 200), r.randint(3, 9)
        return f"How many integers from 1 to {n} inclusive are divisible by {d}?", n // d
    n = r.randint(10, 1000)
    return (f"A binary search on a sorted list of {n} distinct elements halves the remaining range each step. "
            f"In the worst case, how many elements must it compare (that is, what is floor(log2({n})) + 1)?",
            n.bit_length())


GENERATORS = {"arithmetic": arithmetic, "multi_step": multi_step, "programming_concepts": programming_concepts, "algorithmic": algorithmic}


def build() -> dict:
    items = []
    for category in CATEGORIES:
        rng = random.Random(f"{SEED}-{category}")
        seen = set()
        for split, count in PER_CATEGORY.items():
            made = 0
            while made < count:
                question, answer = GENERATORS[category](rng)
                if question in seen:
                    continue
                seen.add(question)
                made += 1
                items.append({"id": f"{category}-{split}-{made:03d}", "category": category, "split": split,
                              "question": question, "prompt": PROMPT.format(question=question), "answer": int(answer)})
    return {"name": "reasoning", "version": "v1", "seed": SEED, "answer_format": "Answer: <integer>", "items": items,
            "notes": "Answers computed by the generator; questions unique within the suite."}
