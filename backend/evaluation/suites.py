"""Public screening tasks, deliberately separate from any future held-out acceptance dataset."""
REASONING = [
    {'id': 'arithmetic', 'prompt': 'A box has 7 red and 5 blue marbles. How many marbles are there? Reply with only the integer.', 'answer': '12'},
    {'id': 'multiplication', 'prompt': 'There are 6 shelves with 8 books on each shelf. How many books are there? Reply with only the integer.', 'answer': '48'},
    {'id': 'subtraction', 'prompt': 'Lina has 23 coins and gives away 9. How many remain? Reply with only the integer.', 'answer': '14'},
]
CODING = [
    {'id': 'binary-search', 'function': 'binary_search', 'prompt': 'Write Python function binary_search(items, target). For sorted distinct integers return the index of target, or -1 if absent. Output only the function, without imports or explanations.',
     'cases': [([[], 1], -1), ([[1, 3, 5], 3], 1), ([[1, 3, 5], 2], -1), ([[4], 4], 0)]},
    {'id': 'palindrome', 'function': 'is_palindrome', 'prompt': 'Write Python function is_palindrome(text) returning whether text equals its reverse, case-sensitive. Output only the function, without imports or explanations.',
     'cases': [([''], True), (['racecar'], True), (['abc'], False), (['Aa'], False)]},
    {'id': 'deduplicate', 'function': 'deduplicate', 'prompt': 'Write Python function deduplicate(items) returning a list of integers with duplicates removed, preserving first occurrence order. Output only the function, without imports or explanations.',
     'cases': [([[]], []), ([[2, 1, 2, 3, 1]], [2, 1, 3]), ([[0, 0]], [0])]},
]
