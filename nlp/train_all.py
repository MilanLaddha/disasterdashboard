"""
Fine-tuning script — ALL 3 models × ALL 5 tasks.

Usage:
    py -m nlp.train_all                          # train everything
    py -m nlp.train_all --models mt5             # train only mT5
    py -m nlp.train_all --tasks sentiment        # train only sentiment
    py -m nlp.train_all --models mt5 --tasks disaster_classification,sentiment

Datasets:
    disaster_classification -> QCRI/HumAID-all (verified real HF dataset, 76.5k tweets)
    location_extraction     -> ai4bharat/naamapadam (Hindi + Marathi combined)
    translation              -> ai4bharat/samanantar (Hindi + Marathi -> English combined)
    sentiment                -> ai4bharat/IndicSentiment
    summarization             -> csebuetnlp/xlsum (Hindi + Marathi + English combined)

Note: disaster_classification is English-only at the source (no ready-made
Hindi/Marathi multiclass disaster-type dataset exists publicly). Hindi/
Marathi disaster-type performance therefore reflects zero-shot cross-lingual
transfer from the multilingual base model, not direct fine-tuning on labeled
Hindi/Marathi data. Document this as a known limitation, not a bug.

Checkpoints saved to: checkpoints/{model}/{task}/
"""

import os, sys, argparse, json, traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
sys.path.insert(0, BASE_DIR)

try:
    import torch
except ImportError:
    torch = None


MAX_INPUT  = 512
MAX_TARGET = 128
MAX_TRAIN  = 10_000   # cap training samples for feasibility
MAX_VAL    = 1_000

# ─── Dataset loaders ──────────────────────────────────────────────────────────

def load_classification_data():
    """HumAID disaster tweet classification (QCRI/HumAID-all — verified real
    dataset, 76.5k tweets, columns: tweet_text, class_label)."""
    from datasets import load_dataset
    try:
        ds = load_dataset("QCRI/HumAID-all", trust_remote_code=True)
        def fmt(ex):
            return {
                "input":  f"classify_disaster: {ex.get('tweet_text', ex.get('text', ''))}",
                "target": str(ex.get("class_label", ex.get("label", "Unknown"))),
            }
        return ds.map(fmt, remove_columns=list(ds["train"].features.keys()))
    except Exception as e:
        print(f"[Dataset] QCRI/HumAID-all failed ({e}), trying crisis_nlp fallback...", file=sys.stderr)
    # Fallback: use CrisisNLP tweet classification
    try:
        ds = load_dataset("cardiffnlp/tweet_topic_single", trust_remote_code=True)
        def fmt2(ex):
            return {"input": f"classify_disaster: {ex['text']}", "target": str(ex.get("label_name", "Other"))}
        return ds.map(fmt2, remove_columns=list(ds["train"].features.keys()))
    except Exception as e2:
        raise ValueError(
            f"Could not load classification dataset. Install with: "
            f"pip install datasets and ensure internet access.\nOriginal: {e2}"
        )

def load_ner_data():
    """ai4bharat/naamapadam NER -> location extraction.
    Trains on Hindi + Marathi combined (previously Hindi-only, leaving
    Marathi location extraction relying solely on zero-shot transfer)."""
    from datasets import load_dataset, concatenate_datasets, DatasetDict
    # NER tag IDs: 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC
    def fmt(ex):
        tokens  = ex.get("tokens", [])
        tags    = ex.get("ner_tags", [])
        loc_toks = [t for t, g in zip(tokens, tags) if g in (5, 6)]
        target  = " ".join(loc_toks) if loc_toks else "None"
        return {
            "input":  f"extract_location: {' '.join(tokens)}",
            "target": target,
        }

    splits_out = {}
    for split in ("train", "validation"):
        parts = []
        for lang in ("hi", "mr"):
            try:
                ds_lang = load_dataset("ai4bharat/naamapadam", lang, trust_remote_code=True)
                if split in ds_lang:
                    parts.append(ds_lang[split].map(fmt, remove_columns=list(ds_lang[split].features.keys())))
            except Exception as e:
                print(f"[Dataset] naamapadam/{lang}/{split} failed: {e}", file=sys.stderr)
        if parts:
            splits_out[split] = concatenate_datasets(parts).shuffle(seed=42)
    if not splits_out:
        raise ValueError("Could not load any naamapadam splits (hi or mr).")
    return DatasetDict(splits_out)

def load_translation_data():
    """ai4bharat/samanantar parallel corpus, Hindi + Marathi -> English combined
    (previously Hindi-only)."""
    from datasets import load_dataset, concatenate_datasets, DatasetDict
    def fmt(ex):
        return {
            "input":  f"translate to English: {ex['src']}",
            "target": ex["tgt"],
        }

    splits_out = {}
    for lang in ("hi", "mr"):
        try:
            ds_lang = load_dataset("ai4bharat/samanantar", lang, trust_remote_code=True)
            for split in ds_lang:
                mapped = ds_lang[split].map(fmt, remove_columns=list(ds_lang[split].features.keys()))
                splits_out.setdefault(split, []).append(mapped)
        except Exception as e:
            print(f"[Dataset] samanantar/{lang} failed: {e}", file=sys.stderr)
    if not splits_out:
        raise ValueError("Could not load samanantar for hi or mr.")
    return DatasetDict({k: concatenate_datasets(v).shuffle(seed=42) for k, v in splits_out.items()})

def load_sentiment_data():
    """ai4bharat/IndicSentiment."""
    from datasets import load_dataset
    ds = load_dataset("ai4bharat/IndicSentiment", trust_remote_code=True)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    def fmt(ex):
        lbl = ex.get("LABEL", ex.get("label", 0))
        if isinstance(lbl, str) and lbl not in label_map.values():
            lbl_str = lbl
        else:
            lbl_str = label_map.get(int(lbl), "Unknown")
        return {
            "input":  f"sentiment: {ex.get('INDIC_SENT', ex.get('text', ''))}",
            "target": lbl_str,
        }
    return ds.map(fmt, remove_columns=list(ds["train"].features.keys()))

def load_summarization_data():
    """csebuetnlp/xlsum cross-lingual summarization, Hindi + Marathi + English
    combined (previously Hindi-only)."""
    from datasets import load_dataset, concatenate_datasets, DatasetDict
    def fmt(ex):
        return {
            "input":  f"summarize: {ex['text'][:500]}",
            "target": ex["summary"],
        }

    splits_out = {}
    for lang in ("hindi", "marathi", "english"):
        try:
            ds_lang = load_dataset("csebuetnlp/xlsum", lang, trust_remote_code=True)
            for split in ds_lang:
                mapped = ds_lang[split].map(fmt, remove_columns=list(ds_lang[split].features.keys()))
                splits_out.setdefault(split, []).append(mapped)
        except Exception as e:
            print(f"[Dataset] xlsum/{lang} failed: {e}", file=sys.stderr)
    if not splits_out:
        raise ValueError("Could not load xlsum for hindi, marathi, or english.")
    return DatasetDict({k: concatenate_datasets(v).shuffle(seed=42) for k, v in splits_out.items()})

DATASET_LOADERS = {
    "disaster_classification": load_classification_data,
    "location_extraction":     load_ner_data,
    "translation":             load_translation_data,
    "sentiment":               load_sentiment_data,
    "summarization":           load_summarization_data,
}

# ─── Tokenisation helpers ─────────────────────────────────────────────────────

def _tok_seq2seq_mt5(batch, tokenizer):
    """Standard seq2seq tokenisation for mT5."""
    enc = tokenizer(batch["input"],
                    max_length=MAX_INPUT, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        dec = tokenizer(batch["target"],
                        max_length=MAX_TARGET, truncation=True, padding="max_length")
    dec["input_ids"] = [
        [(i if i != tokenizer.pad_token_id else -100) for i in ids]
        for ids in dec["input_ids"]
    ]
    enc["labels"] = dec["input_ids"]
    return enc

def _make_tok_mbart(tokenizer, arch, src_code="en_XX", tgt_code="en_XX"):
    """Returns a tokenise function for IndicBART / mBART-50."""
    def tok_fn(batch):
        if arch == "indicbart":
            inputs = [f"{src_code} {t}" for t in batch["input"]]
        else:
            tokenizer.src_lang = src_code
            inputs = batch["input"]

        enc = tokenizer(inputs, max_length=MAX_INPUT, truncation=True, padding="max_length")

        if arch == "mbart50":
            tokenizer.src_lang = tgt_code   # temporarily set to tgt for label tok
        with tokenizer.as_target_tokenizer():
            dec = tokenizer(batch["target"], max_length=MAX_TARGET, truncation=True, padding="max_length")

        dec["input_ids"] = [
            [(i if i != tokenizer.pad_token_id else -100) for i in ids]
            for ids in dec["input_ids"]
        ]
        enc["labels"] = dec["input_ids"]
        # Restore src_lang
        if arch == "mbart50":
            tokenizer.src_lang = src_code
        return enc
    return tok_fn

# ─── Core training function ───────────────────────────────────────────────────

def train_one(model_key: str, task: str, epochs: int = 3,
              batch_size: int = 4, lr: float = 5e-5) -> bool:
    """Fine-tune model_key on task. Saves to checkpoints/{model_key}/{task}/"""
    from transformers import (
        AutoTokenizer, MT5ForConditionalGeneration,
        MBartForConditionalGeneration, MBart50TokenizerFast,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq, EarlyStoppingCallback,
    )
    from nlp.pipeline import MODEL_REGISTRY

    defn    = MODEL_REGISTRY[model_key]
    hf_id   = defn["hf_id"]
    arch    = defn["arch"]
    out_dir = os.path.join(CKPT_DIR, model_key, task)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}\nTRAINING  {model_key} / {task}\nOutput:   {out_dir}\n{'='*60}")

    # Load dataset
    try:
        loader = DATASET_LOADERS[task]
        ds     = loader()
    except Exception as e:
        print(f"[Train] FAILED to load dataset for {task}: {e}", file=sys.stderr)
        traceback.print_exc()
        return False

    val_key    = "validation" if "validation" in ds else ("test" if "test" in ds else None)
    train_split = ds["train"].select(range(min(MAX_TRAIN, len(ds["train"]))))
    val_split   = (ds[val_key].select(range(min(MAX_VAL, len(ds[val_key]))))
                   if val_key else None)

    # Load model + tokeniser (always from HF base for fine-tuning)
    print("[Train] Loading model from HuggingFace...")
    tok_kw = defn.get("tok_kwargs", {})
    if arch == "mt5":
        tokenizer = AutoTokenizer.from_pretrained(hf_id, **tok_kw)
        model     = MT5ForConditionalGeneration.from_pretrained(hf_id)
    elif arch == "mbart50":
        tokenizer = MBart50TokenizerFast.from_pretrained(hf_id)
        model     = MBartForConditionalGeneration.from_pretrained(hf_id)
    else:  # indicbart
        tokenizer = AutoTokenizer.from_pretrained(hf_id, **tok_kw)
        model     = MBartForConditionalGeneration.from_pretrained(hf_id)

    # Tokenise
    print("[Train] Tokenising dataset...")
    if arch == "mt5":
        tok_fn = lambda b: _tok_seq2seq_mt5(b, tokenizer)
    else:
        src = "<2en>" if arch == "indicbart" else "en_XX"
        tgt = "<2en>" if arch == "indicbart" else "en_XX"
        tok_fn = _make_tok_mbart(tokenizer, arch, src, tgt)

    tok_train = train_split.map(tok_fn, batched=True, remove_columns=["input", "target"])
    tok_val   = val_split.map(tok_fn, batched=True, remove_columns=["input", "target"]) if val_split else None

    # Training args
    args = Seq2SeqTrainingArguments(
        output_dir               = out_dir,
        num_train_epochs         = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        learning_rate            = lr,
        weight_decay             = 0.01,
        save_strategy            = "epoch",
        evaluation_strategy      = "epoch" if tok_val else "no",
        load_best_model_at_end   = bool(tok_val),
        predict_with_generate    = True,
        generation_max_length    = MAX_TARGET,
        fp16                     = torch.cuda.is_available() if torch else False,
        report_to                = "none",
        logging_steps            = 100,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if tok_val else []

    trainer = Seq2SeqTrainer(
        model          = model,
        args           = args,
        train_dataset  = tok_train,
        eval_dataset   = tok_val,
        tokenizer      = tokenizer,
        data_collator  = collator,
        callbacks      = callbacks or None,
    )

    print("[Train] Starting fine-tuning...")
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[Train] Checkpoint saved to {out_dir}")
    return True

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune all 3 models × all 5 tasks")
    parser.add_argument("--models",     default="all",
        help="Comma-separated: mt5,indicbart,mbart  or  all")
    parser.add_argument("--tasks",      default="all",
        help="Comma-separated task names or all")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=5e-5)
    args = parser.parse_args()

    ALL_MODELS = ["mt5", "indicbart", "mbart"]
    ALL_TASKS  = list(DATASET_LOADERS.keys())

    models_to_run = ALL_MODELS if args.models == "all" \
                    else [m.strip() for m in args.models.split(",") if m.strip()]
    tasks_to_run  = ALL_TASKS  if args.tasks  == "all" \
                    else [t.strip() for t in args.tasks.split(",")  if t.strip()]

    summary = {}
    for mk in models_to_run:
        summary[mk] = {}
        for task in tasks_to_run:
            ok = train_one(mk, task, args.epochs, args.batch_size, args.lr)
            summary[mk][task] = "SUCCESS" if ok else "FAILED"

    print("\n\n─── Training Summary ───")
    for mk, tasks in summary.items():
        for task, status in tasks.items():
            print(f"  {mk:12s} / {task:30s}: {status}")
