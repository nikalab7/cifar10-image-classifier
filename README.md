# CIFAR-10 Image Classifier

A small, reproducible PyTorch project that trains image classifiers on CIFAR-10 and tries to explain *why* they work or fail, not just report an accuracy number.

It answers two questions, plus an optional third:

1. **Augmentation ablation:** how much does simple augmentation (random crop + horizontal flip) help a small CNN trained from scratch, and is the gain larger than the measurement noise?
2. **Error analysis:** which classes does the model confuse, and which mistakes does it make with the highest confidence?
3. *(optional, needs a GPU)* **Transfer learning:** does an ImageNet-pretrained ResNet-18, fine-tuned for a few epochs, beat the small CNN?

> Code written with AI assistance (Claude). All numbers in this README are produced by the scripts in this repository and stored in `results/*/metrics.json`.

## Evaluation protocol

- Official CIFAR-10 split: 50,000 training and 10,000 test images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- 5,000 training images (fixed seed) are held out as a **validation set**. They are always read *without* augmentation. The validation set is used only to choose the best epoch.
- The **test set is evaluated once**, with the best-validation checkpoint.
- Reported: test accuracy with a 95% bootstrap confidence interval, per-class accuracy, macro-F1, confusion matrix and the most confident mistakes.
- Each configuration is trained once (single seed). A difference smaller than the confidence interval should not be read as a real effect.
- With `--train-subset N`, only the training set is subsampled (a random N of the 45,000 training images). The validation and test sets are never reduced.

## Models

| Name | Description |
|---|---|
| `small_cnn` | Three blocks of two 3×3 conv layers (batch norm + ReLU) with max-pooling, then dropout and one linear layer. Trained from scratch. |
| `resnet18` | torchvision ResNet-18 with a new 10-class head, initialised from ImageNet weights; all layers are fine-tuned. |

Training: cross-entropy loss, AdamW, cosine learning-rate schedule; the best epoch is chosen on validation accuracy.

## Run it

```bash
pip install -r requirements.txt

# 1. Smoke test on random data (seconds, nothing is downloaded)
python -m src.train --fake --epochs 1 --num-workers 0 --no-pretrained --out-dir results/_smoke

# 2. The augmentation ablation (CIFAR-10 is downloaded automatically).
#    10,000 training images and 15 epochs keep this practical on a laptop CPU.
python -m src.train --model small_cnn --no-augment --train-subset 10000 --epochs 15 --out-dir results/small_cnn_noaug
python -m src.train --model small_cnn --augment --train-subset 10000 --epochs 15 --out-dir results/small_cnn_aug

# 2b. Optional, needs a GPU: fine-tune a pretrained ResNet-18
python -m src.train --model resnet18 --augment --image-size 64 --lr 3e-4 --epochs 10 --out-dir results/resnet18_finetune

# 3. Write the results table into this README
python -m src.summarize --update-readme
```

A GPU is strongly recommended. On Google Colab choose *Runtime → Change runtime type → GPU*, then:

```python
!git clone https://github.com/<your-username>/cifar10-image-classifier.git
%cd cifar10-image-classifier
!pip install -q -r requirements.txt
!python -m src.train --model small_cnn --augment --epochs 30 --out-dir results/small_cnn_aug
```

## Results

<!-- RESULTS:START -->
_Run the experiments above, then `python -m src.summarize --update-readme` fills this table._
<!-- RESULTS:END -->

Every run writes its artefacts to `results/<run>/`: `metrics.json`, `history.csv`, `curves.png`, `confusion_matrix.png`, `misclassified.png` and `classification_report.json`.

## Repository layout

```
src/data.py        CIFAR-10 loading, augmentation, train/validation split
src/models.py      SmallCNN and ResNet-18 builder
src/train.py       training loop, checkpoint selection, command-line interface
src/evaluate.py    test metrics, bootstrap CI, confusion matrix, worst mistakes
src/summarize.py   collects results/*/metrics.json into a markdown table
```

## Limitations

- One seed per configuration, so small differences between runs are not conclusive.
- No hyper-parameter search; learning rates and epochs are reasonable defaults, not tuned.
- CIFAR-10 images are only 32×32, which limits what any model can learn.
- Runs that use `--train-subset` see fewer training images, so their absolute accuracy is lower than a full-data run.

## Next steps

- Export the best model to ONNX and serve it with a FastAPI endpoint in Docker.
- Track experiments with Weights & Biases or MLflow.
- Repeat each run with several seeds and report mean ± standard deviation.
- Try stronger augmentation (CutMix, AutoAugment) and compare with the simple baseline.
