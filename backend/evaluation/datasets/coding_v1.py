"""Coding suite v1: 40 Python problems with hidden unit tests.

Expected outputs are computed from the trusted reference implementations below (written for
this suite, never model output) and frozen into coding-v1.json. Prompts contain only the
specification: hidden test inputs and expected outputs are never shown to the model.
The split is fixed by a seeded shuffle; the held-out split must not be edited after results exist.
"""
from __future__ import annotations

import math
import random
from collections import Counter

SPLIT_SEED = 20261109
SCREENING_SIZE = 10


def add_digits(n):
    return sum(int(c) for c in str(n))


def is_prime(n):
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def reverse_words(s):
    return " ".join(reversed(s.split(" "))) if s else ""


def count_vowels(s):
    return sum(c in "aeiou" for c in s.lower())


def fizzbuzz(n):
    return ["FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 else "Buzz" if i % 5 == 0 else str(i) for i in range(1, n + 1)]


def factorial(n):
    return math.factorial(n)


def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def gcd(a, b):
    return math.gcd(a, b)


def flatten(nested):
    out = []
    for item in nested:
        out.extend(flatten(item) if isinstance(item, list) else [item])
    return out


def second_largest(items):
    distinct = sorted(set(items))
    return distinct[-2] if len(distinct) >= 2 else None


def is_anagram(a, b):
    return sorted(a) == sorted(b)


def run_length_encode(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}{j - i}")
        i = j
    return "".join(out)


def chunk(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def rotate_left(items, k):
    if not items:
        return []
    k %= len(items)
    return items[k:] + items[:k]


def transpose(matrix):
    return [list(row) for row in zip(*matrix)] if matrix else []


def merge_sorted(a, b):
    return sorted(a + b)


def word_counts(text):
    return dict(Counter(text.lower().split()))


def title_case(s):
    return " ".join(w[:1].upper() + w[1:].lower() for w in s.split(" "))


def balanced_brackets(s):
    pairs, stack = {")": "(", "]": "[", "}": "{"}, []
    for c in s:
        if c in "([{":
            stack.append(c)
        elif c in pairs:
            if not stack or stack.pop() != pairs[c]:
                return False
    return not stack


def binary_to_int(s):
    return int(s, 2)


def int_to_roman(n):
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
             (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, symbol in table:
        while n >= value:
            out, n = out + symbol, n - value
    return out


def roman_to_int(s):
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, c in enumerate(s):
        v = values[c]
        total += -v if i + 1 < len(s) and values[s[i + 1]] > v else v
    return total


def longest_common_prefix(words):
    if not words:
        return ""
    prefix = words[0]
    for w in words[1:]:
        while not w.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


def max_subarray_sum(nums):
    best = current = nums[0]
    for x in nums[1:]:
        current = max(x, current + x)
        best = max(best, current)
    return best


def unique_paths(m, n):
    return math.comb(m + n - 2, m - 1)


def binary_search(items, target):
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        lo, hi = (mid + 1, hi) if items[mid] < target else (lo, mid - 1)
    return -1


def is_palindrome(text):
    return text == text[::-1]


def deduplicate(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def caesar_shift(text, k):
    return "".join(chr((ord(c) - 97 + k) % 26 + 97) if "a" <= c <= "z" else c for c in text)


def pascal_row(n):
    return [math.comb(n, k) for k in range(n + 1)]


def sum_of_multiples(limit):
    return sum(x for x in range(limit) if x % 3 == 0 or x % 5 == 0)


def mode(items):
    counts = Counter(items)
    top = max(counts.values())
    return min(x for x, c in counts.items() if c == top)


def median(items):
    s = sorted(items)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def compress_spaces(s):
    return " ".join(part for part in s.split(" ") if part)


def nth_prime(n):
    count, candidate = 0, 1
    while count < n:
        candidate += 1
        if is_prime(candidate):
            count += 1
    return candidate


def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def is_power_of_two(n):
    return n > 0 and n & (n - 1) == 0


def collatz_steps(n):
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def invert_dict(d):
    return {v: k for k, v in d.items()}


PROBLEMS = [
    ("add_digits", "add_digits(n)", "returns the sum of the decimal digits of the non-negative integer n",
     [[0], [7], [1234], [9999], [10203]]),
    ("is_prime", "is_prime(n)", "returns True if the integer n is a prime number and False otherwise (numbers below 2 are not prime)",
     [[0], [1], [2], [3], [4], [17], [21], [97], [100]]),
    ("reverse_words", "reverse_words(s)", "returns the words of s (separated by single spaces) in reverse order, joined by single spaces",
     [["hello world"], ["a"], [""], ["one two three"], ["keep  it"]]),
    ("count_vowels", "count_vowels(s)", "returns how many characters of s are vowels (a, e, i, o, u), case-insensitive",
     [[""], ["xyz"], ["Education"], ["AEIOU aeiou"], ["rhythm"]]),
    ("fizzbuzz", "fizzbuzz(n)", "returns a list of strings for 1..n where multiples of 3 are 'Fizz', multiples of 5 are 'Buzz', "
     "multiples of both are 'FizzBuzz' and other numbers are their decimal string", [[0], [1], [5], [15]]),
    ("factorial", "factorial(n)", "returns n! for a non-negative integer n (0! is 1)", [[0], [1], [5], [10], [20]]),
    ("fibonacci", "fibonacci(n)", "returns the n-th Fibonacci number where fibonacci(0) == 0 and fibonacci(1) == 1",
     [[0], [1], [2], [10], [30], [50]]),
    ("gcd", "gcd(a, b)", "returns the greatest common divisor of non-negative integers a and b, with gcd(0, 0) == 0",
     [[0, 0], [0, 5], [12, 18], [17, 5], [100, 75], [270, 192]]),
    ("flatten", "flatten(nested)", "returns a flat list of all integers in an arbitrarily nested list, in left-to-right order",
     [[[]], [[1, 2, 3]], [[1, [2, [3, [4]]], 5]], [[[[]], [6]]]]),
    ("second_largest", "second_largest(items)", "returns the second largest distinct value in a list of integers, or None if there are fewer than two distinct values",
     [[[1]], [[3, 3]], [[1, 2]], [[5, 1, 5, 3]], [[-2, -7, -3]]]),
    ("is_anagram", "is_anagram(a, b)", "returns True if string b is a rearrangement of exactly the characters of string a (case-sensitive)",
     [["listen", "silent"], ["abc", "abd"], ["", ""], ["aab", "abb"], ["Tea", "eat"]]),
    ("run_length_encode", "run_length_encode(s)", "returns a run-length encoding where each run of a repeated character becomes the character followed by the run length, e.g. 'aab' becomes 'a2b1'",
     [[""], ["a"], ["aaabcc"], ["abcd"], ["zzzzzzzzzzzz"]]),
    ("chunk", "chunk(items, size)", "splits the list items into consecutive sublists of length size (the last one may be shorter)",
     [[[], 3], [[1, 2, 3, 4, 5], 2], [[1, 2, 3], 3], [[1, 2], 5]]),
    ("rotate_left", "rotate_left(items, k)", "returns a new list rotated left by k positions (k may exceed the length; an empty list stays empty)",
     [[[], 4], [[1, 2, 3, 4, 5], 2], [[1, 2, 3], 3], [[1, 2, 3], 7]]),
    ("transpose", "transpose(matrix)", "returns the transpose of a rectangular matrix given as a list of row lists (an empty matrix gives [])",
     [[[]], [[[1, 2, 3]]], [[[1, 2], [3, 4], [5, 6]]]]),
    ("merge_sorted", "merge_sorted(a, b)", "merges two ascending lists of integers into one ascending list",
     [[[], []], [[1, 3, 5], [2, 4, 6]], [[1, 1], [1]], [[], [0, 2]], [[-5, 10], [-6, 11]]]),
    ("word_counts", "word_counts(text)", "returns a dictionary mapping each lowercase word to how many times it appears; words are separated by whitespace and compared case-insensitively",
     [[""], ["a b a"], ["The the THE cat"], ["one\ttwo\nthree two"]]),
    ("title_case", "title_case(s)", "returns s where every space-separated word has its first character uppercased and the rest lowercased",
     [["hello world"], ["PYTHON is FUN"], [""], ["a"]]),
    ("balanced_brackets", "balanced_brackets(s)", "returns True if the brackets (), [] and {} in s are balanced and properly nested; other characters are ignored",
     [[""], ["()[]{}"], ["([)]"], ["{[()()]}"], ["(("], ["a(b)c]"]]),
    ("binary_to_int", "binary_to_int(s)", "converts a non-empty string of '0' and '1' characters to the integer it represents in base 2",
     [["0"], ["1"], ["101"], ["11111111"], ["100000000000"]]),
    ("int_to_roman", "int_to_roman(n)", "converts an integer from 1 to 3999 to an uppercase Roman numeral using standard subtractive notation",
     [[1], [4], [9], [14], [40], [1994], [3999]]),
    ("roman_to_int", "roman_to_int(s)", "converts a valid uppercase Roman numeral (standard subtractive notation) to an integer",
     [["I"], ["IV"], ["IX"], ["LVIII"], ["MCMXCIV"], ["MMMCMXCIX"]]),
    ("longest_common_prefix", "longest_common_prefix(words)", "returns the longest string that is a prefix of every string in the list words ('' for an empty list)",
     [[[]], [["flower", "flow", "flight"]], [["dog", "racecar"]], [["same", "same"]], [["a"]]]),
    ("two_sum", "two_sum(nums, target)", "returns [i, j] with i < j for the first pair (smallest i, then smallest j) whose values sum to target, or [] if none exists",
     [[[2, 7, 11, 15], 9], [[3, 2, 4], 6], [[1, 2], 7], [[0, 0, 0], 0], [[], 1]]),
    ("max_subarray_sum", "max_subarray_sum(nums)", "returns the largest sum of a non-empty contiguous sublist of the non-empty integer list nums",
     [[[1]], [[-3, -1, -2]], [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], [[5, 4, -1, 7, 8]]]),
    ("unique_paths", "unique_paths(m, n)", "returns the number of paths from the top-left to the bottom-right cell of a grid "
     "with m rows and n columns when each step moves one cell right or one cell down", [[1, 1], [2, 2], [3, 7], [3, 2], [10, 10]]),
    ("binary_search", "binary_search(items, target)", "returns the index of target in a sorted list of distinct integers, or -1 if absent",
     [[[], 1], [[1, 3, 5], 3], [[1, 3, 5], 2], [[4], 4], [[1, 2, 3, 4, 5, 6, 7], 7]]),
    ("is_palindrome", "is_palindrome(text)", "returns whether text equals its reverse (case-sensitive)",
     [[""], ["racecar"], ["abc"], ["Aa"], ["abba"]]),
    ("deduplicate", "deduplicate(items)", "returns a list with duplicates removed, preserving the order of first occurrence",
     [[[]], [[2, 1, 2, 3, 1]], [[0, 0]], [[5, 4, 3]]]),
    ("caesar_shift", "caesar_shift(text, k)", "shifts each lowercase letter a-z forward by k positions in the alphabet, wrapping around; all other characters are unchanged",
     [["abc", 1], ["xyz", 3], ["Hello, world!", 13], ["", 5], ["abc", 26]]),
    ("pascal_row", "pascal_row(n)", "returns row n (0-indexed) of Pascal's triangle as a list of integers",
     [[0], [1], [4], [10]]),
    ("sum_of_multiples", "sum_of_multiples(limit)", "returns the sum of all non-negative integers below limit that are divisible by 3 or 5",
     [[0], [10], [16], [1000]]),
    ("mode", "mode(items)", "returns the most frequent value in a non-empty list of integers; if several values tie, return the smallest of them",
     [[[1]], [[1, 2, 2, 3]], [[4, 4, 1, 1]], [[-1, -1, 5, 5, 5, -1]]]),
    ("median", "median(items)", "returns the median of a non-empty list of integers as a float (the mean of the two middle values for even lengths)",
     [[[3]], [[3, 1, 2]], [[4, 1, 3, 2]], [[-5, 5]]]),
    ("compress_spaces", "compress_spaces(s)", "replaces every run of space characters in s with a single space and removes leading and trailing spaces",
     [[""], ["a  b"], ["  lead and trail  "], ["   "], ["no_spaces"]]),
    ("nth_prime", "nth_prime(n)", "returns the n-th prime number, counting from nth_prime(1) == 2",
     [[1], [2], [6], [25], [100]]),
    ("hamming_distance", "hamming_distance(a, b)", "returns the number of positions at which two equal-length strings differ",
     [["", ""], ["karolin", "kathrin"], ["1011101", "1001001"], ["same", "same"]]),
    ("is_power_of_two", "is_power_of_two(n)", "returns True if the integer n is a positive power of two (1, 2, 4, ...) and False otherwise",
     [[0], [1], [2], [6], [64], [-8], [1024]]),
    ("collatz_steps", "collatz_steps(n)", "returns how many steps the Collatz process takes to reach 1 from the positive integer n (halve even numbers, map odd n to 3n+1)",
     [[1], [2], [6], [7], [27]]),
    ("invert_dict", "invert_dict(d)", "returns a new dictionary with keys and values swapped (all values are distinct strings)",
     [[{}], [{"a": "x"}], [{"one": "1", "two": "2"}]]),
]

PROMPT = ("Write a Python function `{signature}` that {description}. Use only the Python standard library. "
          "Respond with the complete function in a single Python code block and no explanation.")


def build() -> dict:
    references = globals()
    items = []
    for name, signature, description, inputs in PROBLEMS:
        reference = references[name]
        cases = [[list(args), reference(*[_copy(a) for a in args])] for args in inputs]
        items.append({"id": name, "function": name, "signature": signature,
                      "prompt": PROMPT.format(signature=signature, description=description), "cases": cases})
    order = list(range(len(items)))
    random.Random(SPLIT_SEED).shuffle(order)
    screening = set(order[:SCREENING_SIZE])
    for index, item in enumerate(items):
        item["split"] = "screening" if index in screening else "heldout"
    return {"name": "coding", "version": "v1", "split_seed": SPLIT_SEED, "items": items,
            "notes": "Hidden unit tests; expected values computed from trusted reference implementations. Prompts never include test cases."}


def _copy(value):
    import copy
    return copy.deepcopy(value)
