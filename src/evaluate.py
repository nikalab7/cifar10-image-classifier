"""Test-set evaluation and error analysis.

Produces: accuracy with a bootstrap confidence interval, per-class accuracy, macro-F1,
a confusion matrix, the most frequent confusions and the most confident mistakes.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.data import CLASSES, MEAN, STD

NUM_CLASSES = len(CLASSES)


@torch.no_grad()
def predict(model, loader, device, keep_wrong=16):
    """Predict every image in `loader`.

    Also returns the `keep_wrong` most confident mistakes as
    (confidence, image, true_label, predicted_label) tuples.
    """
    model.eval()
    all_pred, all_true, wrong = [], [], []
    for x, y in loader:
        probs = model(x.to(device)).softmax(dim=1).cpu()
        confidence, pred = probs.max(dim=1)
        all_pred.append(pred)
        all_true.append(y)
        for i in (pred != y).nonzero(as_tuple=True)[0].tolist():
            wrong.append((confidence[i].item(), x[i].clone(), y[i].item(), pred[i].item()))
        wrong = sorted(wrong, key=lambda item: -item[0])[:keep_wrong]
    return torch.cat(all_pred).numpy(), torch.cat(all_true).numpy(), wrong


def bootstrap_ci(correct, n_boot=2000, seed=0):
    """95% bootstrap confidence interval for accuracy. `correct` is a boolean array."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    accuracies = [correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.percentile(accuracies, 2.5)), float(np.percentile(accuracies, 97.5))


def _to_image(tensor):
    """Undo normalisation so a tensor can be shown as a picture."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    threshold = cm.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(wrong, path):
    if not wrong:
        return
    cols = 4
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (confidence, image, true, pred) in zip(axes, wrong):
        ax.imshow(_to_image(image))
        ax.set_title(f"true: {CLASSES[true]}\npred: {CLASSES[pred]} ({confidence:.0%})", fontsize=8)
    fig.suptitle("Most confident mistakes on the test set", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax_loss.set_title("Loss")
    ax_acc.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax_acc.plot(epochs, [h["val_acc"] for h in history], label="validation")
    ax_acc.set_title("Accuracy")
    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_test(model, loader, device, out_dir):
    """Evaluate on the test set, save the plots / report and return a metrics dict."""
    out_dir = Path(out_dir)
    pred, true, wrong = predict(model, loader, device)
    labels = list(range(NUM_CLASSES))

    cm = confusion_matrix(true, pred, labels=labels)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    # The five most frequent (true class -> predicted class) mistakes.
    pairs = [(int(cm[i, j]), i, j) for i in labels for j in labels if i != j]
    top_confusions = [
        {"true": CLASSES[i], "predicted": CLASSES[j], "count": count}
        for count, i, j in sorted(pairs, reverse=True)[:5]
    ]

    report = classification_report(true, pred, labels=labels, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    correct = pred == true
    ci_low, ci_high = bootstrap_ci(correct)

    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")
    plot_misclassified(wrong, out_dir / "misclassified.png")
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2))

    return {
        "test_accuracy": float(correct.mean()),
        "test_accuracy_ci95": [ci_low, ci_high],
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class_accuracy": {c: float(a) for c, a in zip(CLASSES, per_class_accuracy)},
        "top_confusions": top_confusions,
    }
