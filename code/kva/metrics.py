"""The locked set's official metrics, ported line for line in plain Python (no new dependencies).

RULER (datasets/test-set-locked/RULER, commit in ruler_commit.txt):
  postprocess_pred                 scripts/eval/evaluate.py postprocess_pred
  string_match_all / _part         scripts/eval/synthetic/constants.py (task scores, 0-100, 2 dp)
  ruler_row                        one row's term of those sums (0-1), so tables can resample prompts
LongBench v1 (github.com/THUDM/LongBench, LongBench/metrics.py and eval.py, main at 2e00731f, read 2026-10-02):
  normalize_answer, f1_score, qa_f1_score, classification_score, code_sim_score    metrics.py, verbatim logic
  rouge_score       metrics.py: Rouge().get_scores([pred], [gt], avg=True)["rouge-l"]["f"] of the `rouge`
                    package 1.0.1 (rouge_l_summary_level, exclusive=True), re-implemented in rouge_l
  fuzz_ratio        fuzzywuzzy 0.18 fuzz.ratio; LongBench's requirements install fuzzywuzzy without
                    python-Levenshtein, so it runs on difflib.SequenceMatcher, as here
  longbench_row     eval.py scorer's per-row rule (trec: first line only; max over the answers)
  longbench_score   eval.py scorer's task score (0-100, 2 dp)
Known difference: rouge 1.0.1 rebuilds the LCS recursively, so a sentence pair needing about 1000 steps
raises RecursionError, which LongBench's bare except scores 0.0; rouge_l is iterative and scores it.
"""
import difflib
import re
import string
from collections import Counter

# ---- RULER


def postprocess_pred(predict_str):
    predict_str = predict_str.strip()
    np_pattern = re.compile(r'[\x00-\x1f]')          # non-printable characters become newlines
    return np_pattern.sub('\n', predict_str).strip()


def string_match_part(preds, refs):
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)


def string_match_all(preds, refs):
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)


def ruler_row(metric, pred, ref):
    """One row's term of string_match_{all,part} (0-1): the task score is round(sum / n * 100, 2)."""
    hits = [1.0 if r.lower() in pred.lower() else 0.0 for r in ref]
    if metric == "string_match_all":
        return sum(hits) / len(ref)
    if metric == "string_match_part":
        return max(hits)
    raise ValueError(f"unknown RULER metric {metric!r}")


def ruler_score(rows):
    """Task score from ruler_row terms, in the official order of operations."""
    return round(sum(rows) / len(rows) * 100, 2)


# ---- LongBench


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:          # removes while iterating, as the official code (skips the next term)
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = (1.0 / len(em_match_list))
    else:
        score = 0.0
    return score


def fuzz_ratio(s1, s2):
    """fuzzywuzzy fuzz.ratio: its decorators in order (None, equal, empty), then difflib, int(round())."""
    if s1 is None or s2 is None:
        return 0
    if s1 == s2:
        return 100
    if len(s1) == 0 or len(s2) == 0:
        return 0
    return int(round(100 * difflib.SequenceMatcher(None, s1, s2).ratio()))


def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz_ratio(prediction, ground_truth) / 100)


def _sentences(text):
    """rouge.Rouge's split: on '.', dropping empty pieces before collapsing whitespace."""
    return [" ".join(_.split()) for _ in text.split(".") if len(_) > 0]


def _words(sentences):
    return [w for s in sentences for w in s.split(" ")]


def _lcs_words(x, y):
    """rouge_score._recon_lcs: the words of one LCS of x and y, with its DP table and tie-break
    (step up only when table[i-1, j] > table[i, j-1]); iterative."""
    n, m = len(x), len(y)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            table[i][j] = table[i - 1][j - 1] + 1 if x[i - 1] == y[j - 1] else max(table[i - 1][j], table[i][j - 1])
    out, i, j = [], n, m
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            out.append(x[i - 1])
            i, j = i - 1, j - 1
        elif table[i - 1][j] > table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return out


def rouge_l(hyp, ref):
    """rouge 1.0.1 rouge_l_summary_level (exclusive=True) on Rouge's sentence split; raises ValueError on an
    empty side, as the package does. Word counts and the LCS union are sets of words."""
    hyp, ref = _sentences(hyp), _sentences(ref)
    if len(hyp) <= 0 or len(ref) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")
    m = len(set(_words(ref)))
    n = len(set(_words(hyp)))
    union = set()
    for ref_s in ref:
        ref_words = ref_s.split(" ")
        for eval_s in hyp:
            union |= set(_lcs_words(ref_words, eval_s.split(" ")))
    llcs = len(union)
    r_lcs = llcs / m
    p_lcs = llcs / n
    return 2.0 * ((p_lcs * r_lcs) / (p_lcs + r_lcs + 1e-8))


def rouge_score(prediction, ground_truth, **kwargs):
    try:
        return rouge_l(prediction, ground_truth)
    except (ValueError, ZeroDivisionError):
        return 0.0


LONGBENCH_METRIC = {"narrativeqa": qa_f1_score, "qasper": qa_f1_score, "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score,
                    "gov_report": rouge_score, "trec": classification_score, "lcc": code_sim_score,
                    "repobench-p": code_sim_score}
FIRST_LINE = ("trec", "triviaqa", "samsum", "lsht")


def longbench_row(dataset, prediction, answers, all_classes):
    """eval.py scorer for one row (0-1): first line only on the few-shot tasks, best over the answers."""
    score = 0.
    if dataset in FIRST_LINE:
        prediction = prediction.lstrip('\n').split('\n')[0]
    for ground_truth in answers:
        score = max(score, LONGBENCH_METRIC[dataset](prediction, ground_truth, all_classes=all_classes))
    return score


def longbench_score(rows):
    return round(100 * sum(rows) / len(rows), 2)
