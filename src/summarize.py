"""Collect results/*/metrics.json into a markdown table.

    python -m src.summarize                  # print the table
    python -m src.summarize --update-readme  # also write it into README.md
"""
import argparse
import json
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def build_table(results_dir):
    rows = []
    for path in sorted(Path(results_dir).glob("*/metrics.json")):
        if path.parent.name.startswith("_"):  # skip smoke-test runs such as results/_smoke
            continue
        m = json.loads(path.read_text())
        cfg = m["config"]
        model = cfg["model"]
        if model == "resnet18":
            init = "ImageNet init" if cfg["pretrained"] else "random init"
            model += f" ({init}, {cfg['image_size']}px)"
        low, high = m["test_accuracy_ci95"]
        rows.append(
            f"| `{path.parent.name}` | {model} | {'yes' if cfg['augment'] else 'no'} "
            f"| {m.get('train_images', 'n/a')} | {cfg['epochs']} "
            f"| {m['parameters'] / 1e6:.2f}M | {m['best_val_accuracy'] * 100:.2f} "
            f"| {m['test_accuracy'] * 100:.2f} ({low * 100:.1f}–{high * 100:.1f}) "
            f"| {m['train_val_gap_at_best'] * 100:+.1f} |"
        )
    if not rows:
        raise SystemExit(f"No metrics.json files found under {results_dir}/")
    header = ("| Run | Model | Augmentation | Train images | Epochs | Parameters | Val accuracy (%) "
              "| Test accuracy (%) [95% CI] | Train − val gap (pp) |\n"
              "|---|---|---|---|---|---|---|---|---|\n")
    return header + "\n".join(rows)


def update_readme(readme, table):
    text = readme.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"Markers {START} / {END} not found in {readme}")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(f"{before}{START}\n{table}\n{END}{after}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()

    table = build_table(args.results_dir)
    print(table)
    if args.update_readme:
        update_readme(Path(args.readme), table)
        print(f"\nUpdated {args.readme}")


if __name__ == "__main__":
    main()
