from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .data import (
    NUMERIC_FEATURE_NAMES,
    Example,
    Vocabulary,
    expand_records_to_sentence_claims,
    load_examples,
    split_examples,
)
from .model import (
    LABEL_ORDER,
    Batch,
    ClaimCaptionTableVerifier,
    TensorizedDataset,
    collate_batch,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or run a SciTab-style claim+caption+table verifier baseline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a verifier baseline.")
    train_parser.add_argument("--train-file", required=True, help="SciTab-style training JSON or JSONL file.")
    train_parser.add_argument("--dev-file", help="Optional development file.")
    train_parser.add_argument("--test-file", help="Optional test file evaluated after training.")
    train_parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics.")
    train_parser.add_argument("--dev-ratio", type=float, default=0.2, help="Used when --dev-file is omitted.")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=3e-3)
    train_parser.add_argument("--embedding-dim", type=int, default=128)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--max-claim-length", type=int, default=64)
    train_parser.add_argument("--max-text-length", type=int, default=96)
    train_parser.add_argument("--max-table-length", type=int, default=256)
    train_parser.add_argument("--max-vocab-size", type=int, default=20000)
    train_parser.add_argument("--min-token-freq", type=int, default=1)
    train_parser.add_argument("--seed", type=int, default=13)
    train_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved verifier checkpoint.")
    eval_parser.add_argument("--model-dir", required=True)
    eval_parser.add_argument("--data-file", required=True)
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    predict_parser = subparsers.add_parser("predict", help="Predict labels for unlabeled or labeled records.")
    predict_parser.add_argument("--model-dir", required=True)
    predict_parser.add_argument("--data-file", required=True)
    predict_parser.add_argument("--output-file", required=True)
    predict_parser.add_argument("--batch-size", type=int, default=32)
    predict_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit existing text/description sentences against caption + table.",
    )
    audit_parser.add_argument("--model-dir", required=True)
    audit_parser.add_argument("--data-file", required=True, help="JSON/JSONL with text or sentences plus table fields.")
    audit_parser.add_argument("--output-file", required=True, help="JSON output file for grouped audit results.")
    audit_parser.add_argument("--batch-size", type=int, default=32)
    audit_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "predict":
        predict_command(args)
    else:
        audit_command(args)


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = load_examples(args.train_file)
    if args.dev_file:
        dev_examples = load_examples(args.dev_file)
    else:
        train_examples, dev_examples = split_examples(train_examples, args.dev_ratio, args.seed)
    test_examples = load_examples(args.test_file) if args.test_file else []

    vocabulary = Vocabulary.build(
        (
            example.claim_tokens + example.text_tokens + example.table_tokens
            for example in train_examples
        ),
        min_freq=args.min_token_freq,
        max_size=args.max_vocab_size,
    )
    label_to_id = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    config = {
        "max_claim_length": args.max_claim_length,
        "max_text_length": args.max_text_length,
        "max_table_length": args.max_table_length,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "num_numeric_features": len(NUMERIC_FEATURE_NAMES),
        "label_to_id": label_to_id,
    }

    model = ClaimCaptionTableVerifier(
        vocab_size=len(vocabulary.token_to_id),
        num_numeric_features=len(NUMERIC_FEATURE_NAMES),
        num_labels=len(LABEL_ORDER),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = build_loader(
        train_examples,
        vocabulary,
        label_to_id,
        args,
        batch_size=args.batch_size,
        shuffle=True,
    )
    dev_loader = build_loader(
        dev_examples,
        vocabulary,
        label_to_id,
        args,
        batch_size=args.batch_size,
        shuffle=False,
    )

    history: List[Dict[str, float]] = []
    best_state: Dict[str, object] | None = None
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        dev_metrics = evaluate_model(model, dev_loader, device)
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_accuracy": dev_metrics["accuracy"],
        }
        history.append(epoch_record)
        if dev_metrics["macro_f1"] > best_score:
            best_score = dev_metrics["macro_f1"]
            best_state = {
                "model_state": model.state_dict(),
                "vocabulary": vocabulary.to_dict(),
                "config": config,
                "metrics": dev_metrics,
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    checkpoint_path = output_dir / "model.pt"
    torch.save(best_state, checkpoint_path)
    metrics_payload: Dict[str, object] = {
        "train_size": len(train_examples),
        "dev_size": len(dev_examples),
        "history": history,
        "best_dev": best_state["metrics"],
        "feature_names": NUMERIC_FEATURE_NAMES,
        "checkpoint": str(checkpoint_path),
    }

    if test_examples:
        best_model, vocabulary, config = load_model(output_dir, device)
        test_loader = build_loader(
            test_examples,
            vocabulary,
            config["label_to_id"],
            argparse.Namespace(
                max_claim_length=config["max_claim_length"],
                max_text_length=config["max_text_length"],
                max_table_length=config["max_table_length"],
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
        metrics_payload["test"] = evaluate_model(best_model, test_loader, device)

    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics_payload, indent=2))


def evaluate_command(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, vocabulary, config = load_model(args.model_dir, device)
    examples = load_examples(args.data_file)
    loader = build_loader(
        examples,
        vocabulary,
        config["label_to_id"],
        argparse.Namespace(
            max_claim_length=config["max_claim_length"],
            max_text_length=config["max_text_length"],
            max_table_length=config["max_table_length"],
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(json.dumps(evaluate_model(model, loader, device), indent=2))


def predict_command(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, vocabulary, config = load_model(args.model_dir, device)
    examples = load_examples(args.data_file)
    loader = build_loader(
        examples,
        vocabulary,
        config["label_to_id"],
        argparse.Namespace(
            max_claim_length=config["max_claim_length"],
            max_text_length=config["max_text_length"],
            max_table_length=config["max_table_length"],
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    records = predict_examples(model, loader, device)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps({"predictions": len(records), "output_file": str(output_path)}, indent=2))


def audit_command(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, vocabulary, config = load_model(args.model_dir, device)
    claim_records = expand_records_to_sentence_claims(args.data_file)
    examples = examples_from_records(claim_records)
    loader = build_loader(
        examples,
        vocabulary,
        config["label_to_id"],
        argparse.Namespace(
            max_claim_length=config["max_claim_length"],
            max_text_length=config["max_text_length"],
            max_table_length=config["max_table_length"],
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    sentence_predictions = predict_examples(model, loader, device)
    sentence_map = {record["example_id"]: record for record in sentence_predictions}
    grouped_results = build_audit_report(claim_records, sentence_map)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(grouped_results, indent=2), encoding="utf-8")
    summary = grouped_results["summary"]
    print(json.dumps({"output_file": str(output_path), **summary}, indent=2))


def build_loader(
    examples: Sequence[Example],
    vocabulary: Vocabulary,
    label_to_id: Dict[str, int],
    args: argparse.Namespace,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    dataset = TensorizedDataset(
        examples=examples,
        vocabulary=vocabulary,
        label_to_id=label_to_id,
        max_claim_length=args.max_claim_length,
        max_text_length=args.max_text_length,
        max_table_length=args.max_table_length,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
    )


def train_one_epoch(
    model: ClaimCaptionTableVerifier,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses: List[float] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if batch.labels is None:
            raise ValueError("Training batch is missing labels.")
        optimizer.zero_grad()
        logits = model(batch.claim_ids, batch.text_ids, batch.table_ids, batch.numeric_features)
        loss = loss_fn(logits, batch.labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


def evaluate_model(
    model: ClaimCaptionTableVerifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    gold_labels: List[int] = []
    pred_labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = model(batch.claim_ids, batch.text_ids, batch.table_ids, batch.numeric_features)
            predictions = torch.argmax(logits, dim=-1).cpu().tolist()
            pred_labels.extend(int(item) for item in predictions)
            if batch.labels is not None:
                gold_labels.extend(int(item) for item in batch.labels.cpu().tolist())
    if not gold_labels:
        raise ValueError("Evaluation requires labeled data.")
    macro_f1 = f1_score(gold_labels, pred_labels, average="macro")
    accuracy = accuracy_score(gold_labels, pred_labels)
    report = classification_report(
        gold_labels,
        pred_labels,
        labels=list(range(len(LABEL_ORDER))),
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "classification_report": report,
    }


def predict_examples(
    model: ClaimCaptionTableVerifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> List[Dict[str, object]]:
    model.eval()
    predictions: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = model(batch.claim_ids, batch.text_ids, batch.table_ids, batch.numeric_features)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_ids = np.argmax(probs, axis=-1)
            for example_id, pred_id, score_vector in zip(batch.example_ids, pred_ids, probs):
                score_map = {label: float(score_vector[idx]) for idx, label in enumerate(LABEL_ORDER)}
                predictions.append(
                    {
                        "example_id": example_id,
                        "predicted_label": LABEL_ORDER[int(pred_id)],
                        "scores": score_map,
                    }
                )
    return predictions


def examples_from_records(records: Sequence[Dict[str, object]]) -> List[Example]:
    examples = []
    from .data import normalize_record

    for index, record in enumerate(records):
        example = normalize_record(record, index=index)
        if example is not None:
            examples.append(example)
    return examples


def build_audit_report(
    claim_records: Sequence[Dict[str, object]],
    sentence_map: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    grouped: Dict[str, Dict[str, object]] = {}
    for claim_record in claim_records:
        example_id = str(claim_record["id"])
        prediction = sentence_map.get(example_id)
        if prediction is None:
            continue
        parent_id = str(claim_record.get("parent_id") or "unknown_parent")
        group = grouped.setdefault(
            parent_id,
            {
                "parent_id": parent_id,
                "paper": claim_record.get("paper") or "",
                "paper_id": claim_record.get("paper_id") or "",
                "table_id": claim_record.get("table_id") or "",
                "caption": claim_record.get("table_caption") or "",
                "table_preview": build_table_preview(claim_record),
                "supported_sentences": [],
                "refuted_sentences": [],
                "nei_sentences": [],
            },
        )
        scores = prediction["scores"]
        support_score = float(scores["supports"])
        refute_score = float(scores["refutes"])
        nei_score = float(scores["nei"])
        contradiction_score = refute_score - support_score
        uncertainty_score = nei_score
        audit_item = {
            "example_id": example_id,
            "sentence_index": int(claim_record.get("sentence_index") or 0),
            "claim": claim_record.get("claim") or "",
            "predicted_label": prediction["predicted_label"],
            "scores": scores,
            "support_score": support_score,
            "contradiction_score": contradiction_score,
            "uncertainty_score": uncertainty_score,
        }
        predicted_label = prediction["predicted_label"]
        if predicted_label == "supports":
            group["supported_sentences"].append(audit_item)
        elif predicted_label == "refutes":
            group["refuted_sentences"].append(audit_item)
        else:
            group["nei_sentences"].append(audit_item)

    results = list(grouped.values())
    for record in results:
        record["supported_sentences"].sort(key=lambda item: (-float(item["support_score"]), int(item["sentence_index"])))
        record["refuted_sentences"].sort(
            key=lambda item: (-float(item["contradiction_score"]), -float(item["scores"]["refutes"]), int(item["sentence_index"]))
        )
        record["nei_sentences"].sort(
            key=lambda item: (-float(item["uncertainty_score"]), int(item["sentence_index"]))
        )
        record["summary"] = {
            "supports": len(record["supported_sentences"]),
            "refutes": len(record["refuted_sentences"]),
            "nei": len(record["nei_sentences"]),
        }
    summary = {
        "records": len(results),
        "supported_sentences": sum(len(record["supported_sentences"]) for record in results),
        "refuted_sentences": sum(len(record["refuted_sentences"]) for record in results),
        "nei_sentences": sum(len(record["nei_sentences"]) for record in results),
        "audited_sentences": len(claim_records),
    }
    return {"summary": summary, "results": results}


def build_table_preview(record: Dict[str, object], max_rows: int = 4, max_cols: int = 6) -> Dict[str, object]:
    columns = record.get("table_column_names") or []
    rows = record.get("table_content_values") or []
    preview_columns: List[str] = []
    if isinstance(columns, list):
        preview_columns = [str(value) for value in columns[:max_cols]]
    preview_rows: List[List[str]] = []
    if isinstance(rows, list):
        for row in rows[:max_rows]:
            if isinstance(row, list):
                preview_rows.append([str(value) for value in row[:max_cols]])
    return {
        "columns": preview_columns,
        "rows": preview_rows,
        "truncated_rows": max(0, len(rows) - len(preview_rows)) if isinstance(rows, list) else 0,
        "truncated_cols": max(0, len(columns) - len(preview_columns)) if isinstance(columns, list) else 0,
    }


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    labels = batch.labels.to(device) if batch.labels is not None else None
    return Batch(
        claim_ids=batch.claim_ids.to(device),
        text_ids=batch.text_ids.to(device),
        table_ids=batch.table_ids.to(device),
        numeric_features=batch.numeric_features.to(device),
        labels=labels,
        example_ids=batch.example_ids,
    )


def load_model(model_dir: str | Path, device: torch.device) -> Tuple[ClaimCaptionTableVerifier, Vocabulary, Dict[str, object]]:
    checkpoint = torch.load(Path(model_dir) / "model.pt", map_location=device)
    vocabulary = Vocabulary.from_dict(checkpoint["vocabulary"])
    config = checkpoint["config"]
    model = ClaimCaptionTableVerifier(
        vocab_size=len(vocabulary.token_to_id),
        num_numeric_features=config["num_numeric_features"],
        num_labels=len(config["label_to_id"]),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, vocabulary, config


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
