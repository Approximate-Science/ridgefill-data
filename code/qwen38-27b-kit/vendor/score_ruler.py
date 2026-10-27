# Metrics follow NVIDIA/RULER scripts/eval/synthetic/constants.py (Apache-2.0).
"""RULER case-insensitive substring metrics; scores are percentages."""
import argparse
import json
import re

from score_common import categories, load_pairs, summarize

TASKS = ['niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
         'niah_multivalue', 'niah_multiquery', 'vt', 'cwe', 'fwe', 'qa_1', 'qa_2']


def score(items_path, predictions_path):
    rows = []
    for item, text in load_pairs(items_path, predictions_path):
        task = item['task']
        if task not in TASKS:
            raise ValueError(f"Unknown task for {item['id']}: {task}")
        prediction = re.sub(r'[\x00-\x1f]', '\n', text.strip()).strip() if isinstance(text, str) else ''
        unparseable = not prediction or not any(c.isalnum() for c in prediction)
        matches = [answer.lower() in prediction.lower() for answer in item['answers']]
        value = (float(any(matches)) if task.startswith('qa_') else sum(matches) / len(matches)) if not unparseable else 0.0
        rows.append(dict(task=task, length_bucket=item['length_bucket'], score=value, unparseable=unparseable))
    by_task = categories(rows, 'task')
    overall = summarize(rows)
    overall['macro_task_score'] = round(sum(r['score'] for r in by_task.values()) / len(by_task), 6)
    return dict(overall=overall, by_task=by_task, by_length_bucket=categories(rows, 'length_bucket'),
                unparseable=overall['unparseable'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('items')
    parser.add_argument('predictions')
    args = parser.parse_args()
    try:
        result = score(args.items, args.predictions)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
