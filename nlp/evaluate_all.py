"""
Evaluation script — runs all fine-tuned checkpoints on test sets and
computes real metrics. Populates evaluation_results/metrics.json.

Usage:
    py -m nlp.evaluate_all                 # evaluate all models × tasks
    py -m nlp.evaluate_all --models mt5    # one model
    py -m nlp.evaluate_all --n 100         # 100 test samples per task

Required packages:
    pip install sacrebleu rouge-score scikit-learn seqeval
"""

import os, sys, json, argparse, traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation_results")
os.makedirs(EVAL_DIR, exist_ok=True)
METRICS_PATH = os.path.join(EVAL_DIR, "metrics.json")

sys.path.insert(0, BASE_DIR)

from nlp.pipeline import (
    task_classify_disaster, task_extract_location,
    task_translate, task_sentiment, task_summarize,
    detect_language, MODEL_REGISTRY, score_candidate_label,
)
from nlp.train_all import (
    load_classification_data, load_ner_data,
    load_translation_data, load_sentiment_data, load_summarization_data,
)

DISASTER_CLASSES = ["Flood", "Landslide", "Cyclone", "Earthquake", "Other"]
_MODEL_DISPLAY_SHORT = {"mt5": "mT5", "indicbart": "IndicBART", "mbart": "mBART50"}

def _compute_confusion_and_curves(model_key: str, test_examples: list, n: int = 100) -> dict:
    """
    Real confusion matrix + ROC/PR curves for the disaster_classification task,
    computed from actual model predictions and actual teacher-forced label
    scores (see score_candidate_label in pipeline.py) - not simulated.
    Returns {} on any failure rather than fabricating a plausible-looking result.
    """
    import numpy as np
    from sklearn.metrics import roc_curve, precision_recall_curve

    y_true_idx, y_scores = [], {c: [] for c in DISASTER_CLASSES}
    conf = {a: {p: 0 for p in DISASTER_CLASSES} for a in DISASTER_CLASSES}

    for ex in test_examples[:n]:
        raw_input = ex.get("input", "")
        text = raw_input.split(": ", 1)[1].strip() if ": " in raw_input else raw_input.strip()
        true_label = str(ex.get("target", "")).strip()
        if true_label not in DISASTER_CLASSES:
            continue  # only score examples whose gold label is one of the 5 known classes

        try:
            pred = task_classify_disaster(text, model_key)
        except Exception:
            pred = "Other"
        pred = pred if pred in DISASTER_CLASSES else "Other"
        conf[true_label][pred] += 1

        y_true_idx.append(DISASTER_CLASSES.index(true_label))
        for c in DISASTER_CLASSES:
            try:
                score = score_candidate_label(model_key, "disaster_classification", text, c)
            except Exception:
                score = 0.0
            y_scores[c].append(score)

    if not y_true_idx:
        return {}

    conf_rows = [
        {"actual": a, **{f"predicted{p}": conf[a][p] for p in DISASTER_CLASSES}}
        for a in DISASTER_CLASSES
    ]

    # One-vs-rest ROC/PR per class, averaged, interpolated onto a shared grid
    # so all 3 models can be plotted on the same x-axis in the dashboard.
    grid = np.linspace(0, 1, 21)
    y_true_arr = np.array(y_true_idx)
    tpr_interp_sum = np.zeros_like(grid)
    prec_interp_sum = np.zeros_like(grid)
    n_classes_scored = 0

    for ci, c in enumerate(DISASTER_CLASSES):
        binary_true = (y_true_arr == ci).astype(int)
        scores = np.array(y_scores[c])
        if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
            continue  # ROC undefined if class never/always appears in this sample
        try:
            fpr, tpr, _ = roc_curve(binary_true, scores)
            tpr_interp_sum += np.interp(grid, fpr, tpr)
            prec, rec, _ = precision_recall_curve(binary_true, scores)
            # precision_recall_curve returns recall descending; sort for interp
            order = np.argsort(rec)
            prec_interp_sum += np.interp(grid, rec[order], prec[order])
            n_classes_scored += 1
        except Exception:
            continue

    if n_classes_scored == 0:
        return {"confusion_matrix": conf_rows}

    return {
        "confusion_matrix": conf_rows,
        "roc_points": list(zip(grid.tolist(), (tpr_interp_sum / n_classes_scored).tolist())),
        "pr_points":  list(zip(grid.tolist(), (prec_interp_sum / n_classes_scored).tolist())),
    }

# ─── Metric computations ──────────────────────────────────────────────────────

def _cls_metrics(preds, labels):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc      = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy":  round(float(acc), 4),
        "precision": round(float(p),   4),
        "recall":    round(float(r),   4),
        "f1":        round(float(f1),  4),
    }

def _ner_metrics(preds, labels):
    """Exact-match precision/recall/F1 for location extraction."""
    tp = sum(1 for p, l in zip(preds, labels)
             if p.strip().lower() == l.strip().lower() and l.strip().lower() not in ("none",""))
    pred_pos  = sum(1 for p in preds  if p.strip().lower() not in ("none","","unknown"))
    label_pos = sum(1 for l in labels if l.strip().lower() not in ("none",""))
    precision = tp / max(pred_pos,  1)
    recall    = tp / max(label_pos, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }

def _translation_metrics(preds, labels):
    from sacrebleu import corpus_bleu, corpus_chrf
    refs  = [[l] for l in labels]
    bleu  = corpus_bleu(preds, refs).score
    chrf  = corpus_chrf(preds, refs).score
    return {"bleu": round(bleu, 2), "chrf": round(chrf, 2)}

def _rouge_metrics(preds, labels):
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1 = r2 = rl = 0.0
    for p, l in zip(preds, labels):
        s   = scorer.score(l, p)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
    n = max(len(preds), 1)
    return {
        "rouge1": round(r1 / n, 4),
        "rouge2": round(r2 / n, 4),
        "rougeL": round(rl / n, 4),
    }

# ─── Per-task evaluation runners ─────────────────────────────────────────────

def _run_eval_task(model_key: str, task: str,
                   test_examples: list, n: int = 200) -> dict:
    """Run model_key on up to n test examples for the given task. Returns metric dict."""
    preds, labels = [], []

    for ex in test_examples[:n]:
        # Strip task prefix from input to get raw text
        raw_input = ex.get("input", "")
        if ": " in raw_input:
            text = raw_input.split(": ", 1)[1].strip()
        else:
            text = raw_input.strip()
        label = str(ex.get("target", "")).strip()
        labels.append(label)

        try:
            if task == "disaster_classification":
                pred = task_classify_disaster(text, model_key)
            elif task == "location_extraction":
                loc  = task_extract_location(text, model_key)
                pred = loc.get("district") or ""
            elif task == "translation":
                lang = detect_language(text)
                pred = task_translate(text, model_key, lang)
            elif task == "sentiment":
                pred = task_sentiment(text, model_key)
            elif task == "summarization":
                pred = task_summarize(text, model_key)
            else:
                pred = ""
        except Exception as e:
            pred = f"ERROR: {str(e)[:40]}"

        preds.append(pred)

    # Compute metrics
    if task in ("disaster_classification", "sentiment"):
        return _cls_metrics(preds, labels)
    elif task == "location_extraction":
        return _ner_metrics(preds, labels)
    elif task == "translation":
        return _translation_metrics(preds, labels)
    elif task == "summarization":
        return _rouge_metrics(preds, labels)
    return {}

# ─── Main evaluation loop ────────────────────────────────────────────────────

TASK_LOADERS = {
    "disaster_classification": load_classification_data,
    "location_extraction":     load_ner_data,
    "translation":             load_translation_data,
    "sentiment":               load_sentiment_data,
    "summarization":           load_summarization_data,
}

def run_all(models=None, tasks=None, n: int = 200):
    """Evaluate all model × task combinations. Writes metrics.json."""
    all_models = models or list(MODEL_REGISTRY.keys())
    all_tasks  = tasks  or list(TASK_LOADERS.keys())

    # Load existing metrics (to allow partial updates)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
        except Exception:
            pass

    for task in all_tasks:
        print(f"\n[Eval] ─── Task: {task} ───")
        loader = TASK_LOADERS[task]
        try:
            ds = loader()
        except Exception as e:
            print(f"  SKIP — dataset load failed: {e}", file=sys.stderr)
            continue

        val_key    = "validation" if "validation" in ds else ("test" if "test" in ds else None)
        if not val_key:
            print(f"  SKIP — no validation/test split for {task}", file=sys.stderr)
            continue

        test_data = list(ds[val_key])
        if task not in metrics:
            metrics[task] = {}

        for mk in all_models:
            print(f"  [{task}] evaluating {mk} ({n} samples)...")
            try:
                result = _run_eval_task(mk, task, test_data, n)
                metrics[task][mk] = result
                print(f"    {mk}: {result}")
            except Exception as e:
                traceback.print_exc()
                metrics[task][mk] = {"error": str(e)[:120]}
                print(f"    {mk}: ERROR — {e}", file=sys.stderr)

        # Save after each task (incremental)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  [Eval] Saved to {METRICS_PATH}")

    # ── Real confusion matrix + ROC/PR curves (disaster_classification only -
    # the one task with a small, fixed label set the dashboard's confusion
    # matrix panel is built around). Computed from real predictions and real
    # teacher-forced label scores - see _compute_confusion_and_curves above.
    if "disaster_classification" in all_tasks:
        print("\n[Eval] ─── Computing confusion matrices + ROC/PR curves ───")
        try:
            cls_ds = load_classification_data()
            cls_val_key = "validation" if "validation" in cls_ds else ("test" if "test" in cls_ds else None)
            if cls_val_key:
                cls_test = list(cls_ds[cls_val_key])
                confusion_matrices = {}
                roc_by_grid, pr_by_grid = {}, {}
                for mk in all_models:
                    print(f"  [ConfMatrix/ROC/PR] {mk}...")
                    result = _compute_confusion_and_curves(mk, cls_test, n=100)
                    if not result:
                        print(f"    {mk}: SKIPPED - no scoreable examples", file=sys.stderr)
                        continue
                    confusion_matrices[_MODEL_DISPLAY_SHORT.get(mk, mk)] = result.get("confusion_matrix", [])
                    if "roc_points" in result:
                        roc_by_grid[mk] = result["roc_points"]
                        pr_by_grid[mk] = result["pr_points"]

                if confusion_matrices:
                    metrics["confusion_matrices"] = confusion_matrices
                if roc_by_grid:
                    # Merge per-model curves onto the shared grid into the
                    # single array-of-points shape the frontend expects:
                    # [{fpr, mT5, IndicBART, mBART50}, ...]
                    any_model = next(iter(roc_by_grid))
                    n_points = len(roc_by_grid[any_model])
                    roc_curves, pr_curves = [], []
                    for i in range(n_points):
                        fpr_val = roc_by_grid[any_model][i][0]
                        rec_val = pr_by_grid[any_model][i][0]
                        roc_point = {"fpr": fpr_val}
                        pr_point  = {"recall": rec_val}
                        for mk, short in _MODEL_DISPLAY_SHORT.items():
                            if mk in roc_by_grid:
                                roc_point[short] = roc_by_grid[mk][i][1]
                                pr_point[short]  = pr_by_grid[mk][i][1]
                        roc_curves.append(roc_point)
                        pr_curves.append(pr_point)
                    metrics["roc_curves"] = roc_curves
                    metrics["pr_curves"] = pr_curves

                with open(METRICS_PATH, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                print(f"  [Eval] Confusion matrices + curves saved to {METRICS_PATH}")
            else:
                print("  SKIP - no validation/test split for disaster_classification", file=sys.stderr)
        except Exception as e:
            traceback.print_exc()
            print(f"  [Eval] Confusion matrix/curve computation failed: {e}", file=sys.stderr)

    print(f"\n[Eval] Complete. Metrics written to {METRICS_PATH}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned NLP checkpoints")
    parser.add_argument("--models", default="all",
        help="Comma-separated model keys or 'all'")
    parser.add_argument("--tasks",  default="all",
        help="Comma-separated task names or 'all'")
    parser.add_argument("--n",      type=int, default=200,
        help="Number of test samples per task (default 200)")
    a = parser.parse_args()

    models_to_eval = (None if a.models == "all"
                      else [m.strip() for m in a.models.split(",") if m.strip()])
    tasks_to_eval  = (None if a.tasks  == "all"
                      else [t.strip() for t in a.tasks.split(",")  if t.strip()])

    run_all(models=models_to_eval, tasks=tasks_to_eval, n=a.n)
