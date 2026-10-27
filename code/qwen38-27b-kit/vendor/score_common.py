import json
from pathlib import Path


def read_rows(path, allow_empty=False):
    rows = {}
    with Path(path).open() as source:
        lines = list(source)
    for line_number, line in enumerate(lines, 1):
        row = json.loads(line)
        identifier = row.get('id')
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f'{path}:{line_number}: invalid id')
        if identifier in rows:
            raise ValueError(f'Duplicate id: {identifier}')
        rows[identifier] = row
    if not rows and not allow_empty:
        raise ValueError(f'Empty file: {path}')
    return rows


def load_pairs(items_path, predictions_path):
    items = read_rows(items_path)
    predictions = read_rows(predictions_path, allow_empty=True)
    missing = items.keys() - predictions.keys()
    extra = predictions.keys() - items.keys()
    if missing:
        raise ValueError('Missing prediction(s): ' + ', '.join(sorted(missing)))
    if extra:
        raise ValueError('Unexpected prediction(s): ' + ', '.join(sorted(extra)))
    for identifier, item in items.items():
        answers = item.get('answers')
        if not isinstance(answers, list) or not answers or not all(isinstance(a, str) and a for a in answers):
            raise ValueError(f'Invalid gold answers: {identifier}')
    return [(item, predictions[identifier].get('text')) for identifier, item in items.items()]


def summarize(rows):
    return dict(count=len(rows), score=round(100 * sum(r['score'] for r in rows) / len(rows), 6),
                unparseable=sum(r['unparseable'] for r in rows))


def categories(rows, field):
    return {key: summarize([r for r in rows if r[field] == key]) for key in sorted({r[field] for r in rows})}
