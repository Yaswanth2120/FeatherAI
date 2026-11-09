import json
import math
from collections import Counter
import pytest
from backend.evaluation import answers
from backend.evaluation.datasets import load, SHA256
from backend.evaluation.datasets import coding_v1, reasoning_v1
from backend.evaluation.runner import compare_dataset_evaluations
from backend.evaluation.scoring import paired_comparison


@pytest.mark.parametrize('text,strict,lenient', [
    ('7 + 5 = 12\nAnswer: 12', 12, 12),
    ('Answer: 5+3=8', 5, 8),            # first number after the last tag
    ('Answer: **42**', 42, 42),
    ('answer: $1,250', 1250, 1250),
    ('Answer: -7', -7, -7),
    ('Answer: 18.0', 18, 18),
    ('Answer: 4.5', None, 5),            # non-integer is a format failure; lenient takes the last integer token "5"
    ('Answer: about 18', None, 18),
    ('Answer: 3\nWait, Answer: 4', 4, 4),
    ('The total is 12.', None, 12),
    ('', None, None),
])
def test_extraction(text, strict, lenient):
    found = answers.extract(text)
    assert found['strict'] == strict
    if lenient is not None and text != 'Answer: 4.5':
        assert found['lenient'] == lenient


def test_score_categories():
    assert answers.score('Answer: 12', 12, False)['category'] == 'correct'
    assert answers.score('Answer: 13', 12, False)['category'] == 'incorrect'
    assert answers.score('so 12', 12, False)['category'] == 'invalid_format'
    assert answers.score('so 12', 12, False)['lenient_passed'] is True
    assert answers.score('step one...', 12, True)['category'] == 'truncated'
    # A truncated output that still reached a tagged answer is scored normally.
    assert answers.score('Answer: 12 and then', 12, True)['passed'] is True


def test_frozen_datasets_match_generators_and_hashes():
    for name, module in [('reasoning-v1', reasoning_v1), ('coding-v1', coding_v1)]:
        data = load(name)
        regenerated = json.loads(json.dumps(module.build(), sort_keys=True))
        stored = {k: v for k, v in data.items() if k != 'sha256'}
        assert stored == regenerated
        assert len(SHA256[name]) == 64


def test_reasoning_split_sizes_and_unique_questions():
    data = load('reasoning-v1')
    counts = Counter((i['category'], i['split']) for i in data['items'])
    assert all(counts[(c, 'screening')] == 10 and counts[(c, 'heldout')] == 40 for c in reasoning_v1.CATEGORIES)
    assert len({i['question'] for i in data['items']}) == len(data['items'])
    assert all(i['prompt'].endswith('"Answer: <integer>".') for i in data['items'])


def test_coding_references_and_no_test_leakage():
    data = load('coding-v1')
    assert Counter(i['split'] for i in data['items']) == {'screening': 10, 'heldout': 30}
    for item in data['items']:
        reference = getattr(coding_v1, item['function'])
        for args, expected in item['cases']:
            assert json.loads(json.dumps(reference(*json.loads(json.dumps(args))))) == expected
        # Hidden expected outputs never appear in prompts (short values such as 0 or True are
        # unavoidable words and are not checked).
        for _, expected in item['cases']:
            rendered = json.dumps(expected)
            if len(rendered) > 4:
                assert rendered not in item['prompt'] and rendered.strip('"') not in item['prompt'], (item['id'], rendered)


def test_tampered_dataset_is_rejected(monkeypatch):
    monkeypatch.setitem(SHA256, 'reasoning-v1', '0' * 64)
    with pytest.raises(ValueError, match='hash mismatch'):
        load('reasoning-v1')


def test_paired_comparison_statistics():
    base = [True] * 30 + [False] * 10
    same = paired_comparison(base, base)
    assert same['difference'] == 0 and same['difference_ci95'] == [0, 0] and same['mcnemar_exact_p'] == 1
    worse = paired_comparison(base, [False] * 10 + [True] * 20 + [False] * 10)
    assert worse['difference'] == pytest.approx(-0.25)
    assert worse['difference_ci95'][1] < 0
    assert worse['baseline_only_correct'] == 10 and worse['candidate_only_correct'] == 0
    assert worse['mcnemar_exact_p'] == pytest.approx(2 / 2 ** 10)
    with pytest.raises(ValueError):
        paired_comparison([True], [])


def evaluation(passes, source='s', category='arithmetic'):
    return {'dataset_sha256': 'd', 'suite_sha256': 'x', 'settings_sha256': 'o', 'split': 'heldout', 'prompt_template': 't',
            'source_sha256': source, 'status': 'completed',
            'results': [{'task_id': str(i), 'item_category': category, 'passed': p} for i, p in enumerate(passes)]}


def test_baseline_validity_labels_uninformative_categories():
    weak = compare_dataset_evaluations(evaluation([True] * 3 + [False] * 37), evaluation([False] * 40))
    assert weak['categories']['arithmetic']['informative'] is False
    assert weak['categories']['arithmetic']['evidence'] == 'insufficient'
    strong = compare_dataset_evaluations(evaluation([True] * 30 + [False] * 10), evaluation([True] * 30 + [False] * 10))
    assert strong['categories']['overall']['informative'] and strong['categories']['overall']['evidence'] == 'no detectable change'
    assert compare_dataset_evaluations(evaluation([True]), evaluation([True], source='other'))['status'] == 'incomparable'
