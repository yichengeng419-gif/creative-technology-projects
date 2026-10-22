"""
train_merged.py
================
MyFridge food-detection model — training script (merged dataset version).

This script trains a YOLOv8 object detector on a MERGED dataset that combines:
  1. A public Roboflow "Fridge Objects" dataset (packaged supermarket products,
     top-down view, CC BY 4.0)
  2. Our own self-captured & self-labelled images (same top-down capture protocol)

The two datasets are merged with a unified class table (duplicate class names
are mapped to a single id) before training. See `merge_datasets.py` for the
merging step that produces /content/merged/data.yaml.

Environment: trained on Google Colab with an NVIDIA Tesla T4 GPU.
Local training on Apple MPS was attempted first but produced NaN losses, so
training was moved to a cloud GPU (training/inference separation).

Usage (in Colab):
    pip install ultralytics
    python train_merged.py
"""

from ultralytics import YOLO


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Path to the merged dataset config (produced by merge_datasets.py).
DATA_YAML = "/content/merged/data.yaml"

# Base model. We use yolov8m for higher accuracy than yolov8s.
# Swap to "yolov8s.pt" for a lighter, faster model better suited to edge devices.
BASE_MODEL = "yolov8m.pt"

# Training hyperparameters
EPOCHS = 120          # upper bound; early stopping (patience) may stop sooner
IMG_SIZE = 640
BATCH = 16
PATIENCE = 40         # stop if val metrics don't improve for 40 epochs
RUN_NAME = "fridge_merged_m"


def main() -> None:
    # Load the pretrained base model (weights downloaded automatically).
    model = YOLO(BASE_MODEL)

    # Fine-tune on the merged fridge dataset.
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=0,              # 0 = first CUDA GPU; use "cpu" if no GPU
        patience=PATIENCE,
        name=RUN_NAME,
        # --- regularisation / augmentation ---
        label_smoothing=0.1,   # soften hard labels -> less overconfidence on
                               # visually similar classes (e.g. cabbage variants)
        degrees=10,            # random rotation +/-10 deg (food orientation varies)
        fliplr=0.5,            # horizontal flip
        hsv_h=0.015,           # colour jitter to be robust to lighting
        hsv_s=0.7,
        hsv_v=0.4,
    )

    save_dir = str(model.trainer.save_dir)
    print("Training finished. Results saved to:", save_dir)

    # Evaluate the best checkpoint on the held-out test split.
    best = YOLO(f"{save_dir}/weights/best.pt")
    metrics = best.val(data=DATA_YAML, split="test")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP@0.50  : {metrics.box.map50:.4f}")
    print(f"mAP@.5:.95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()