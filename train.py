"""Fine-tune DistilBERT on BANKING77 and export the best checkpoint."""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/banking77-distilbert"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Optional small subset for a quick smoke run.")
    parser.add_argument("--max-test-samples", type=int, default=None,
                        help="Optional evaluation subset for a quick smoke run.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0

    with torch.inference_mode():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
            loss_total += outputs.loss.item() * batch["labels"].size(0)

    return {"accuracy": correct / total, "loss": loss_total / total}


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}; loading BANKING77...")

    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    label_names = dataset["train"].features["label"].names
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    if args.max_train_samples:
        tokenized["train"] = tokenized["train"].shuffle(seed=args.seed).select(
            range(min(args.max_train_samples, len(tokenized["train"])))
        )
    if args.max_test_samples:
        tokenized["test"] = tokenized["test"].select(
            range(min(args.max_test_samples, len(tokenized["test"])))
        )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        tokenized["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    test_loader = DataLoader(
        tokenized["test"], batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    id2label = {index: label for index, label in enumerate(label_names)}
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    best_accuracy = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += outputs.loss.item()
            progress.set_postfix(loss=f"{outputs.loss.item():.4f}")

        metrics = evaluate(model, test_loader, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = running_loss / len(train_loader)
        history.append(metrics)
        print(
            f"Epoch {epoch + 1}: train_loss={metrics['train_loss']:.4f}, "
            f"test_loss={metrics['loss']:.4f}, test_accuracy={metrics['accuracy']:.4f}"
        )

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)

    summary = {
        "dataset": "PolyAI/banking77",
        "model": args.model_name,
        "train_examples": len(tokenized["train"]),
        "test_examples": len(tokenized["test"]),
        "num_labels": len(label_names),
        "best_test_accuracy": best_accuracy,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Exported model and tokenizer to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()