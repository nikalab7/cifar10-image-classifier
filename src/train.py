"""Train and evaluate a CIFAR-10 classifier.

Example:
    python -m src.train --model small_cnn --no-augment --epochs 30 --out-dir results/small_cnn_noaug
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data import get_loaders
from src.evaluate import evaluate_test, plot_curves
from src.models import build_model, count_parameters


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["small_cnn", "resnet18"], default="small_cnn")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="random crop + horizontal flip on training images")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                   help="resnet18 only: start from ImageNet weights")
    p.add_argument("--image-size", type=int, default=32,
                   help="images are resized to this size (use 64 for resnet18)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--train-subset", type=int, default=0,
                   help="train on only this many images (0 = all 45,000); useful on a CPU")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results/run")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--fake", action="store_true",
                   help="tiny random dataset, only to check that the code runs")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise only evaluates.

    Note: training loss / accuracy are measured while the model is in train mode and
    on augmented images, so they are usually a bit pessimistic compared with eval mode.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += x.size(0)
    return total_loss / seen, correct / seen


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders(
        args.data_dir, args.augment, args.image_size, args.batch_size,
        seed=args.seed, num_workers=args.num_workers, fake=args.fake,
        train_limit=args.train_subset)

    model = build_model(args.model, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"device={device}  model={args.model}  parameters={count_parameters(model):,}")

    history, best_val, best_epoch, best_state = [], -1.0, 0, None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best_val:  # keep the best epoch, judged on validation data only
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.3f} / {train_acc:.3%}"
              f"  val {val_loss:.3f} / {val_acc:.3%}")
    training_seconds = time.time() - start

    # The test set is used exactly once, with the best-validation checkpoint.
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pt")
    metrics = evaluate_test(model, test_loader, device, out_dir)

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_curves(history, out_dir / "curves.png")

    best = history[best_epoch - 1]
    summary = {
        "config": vars(args),
        "device": str(device),
        "parameters": count_parameters(model),
        "train_images": len(train_loader.dataset),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "train_val_gap_at_best": best["train_acc"] - best["val_acc"],
        "training_seconds": round(training_seconds, 1),
        **metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    low, high = metrics["test_accuracy_ci95"]
    print(f"\nbest epoch {best_epoch}  val {best_val:.3%}")
    print(f"TEST accuracy {metrics['test_accuracy']:.3%}  (95% CI {low:.3%} - {high:.3%})")
    print("most frequent mistakes:", metrics["top_confusions"][:3])


if __name__ == "__main__":
    main()
