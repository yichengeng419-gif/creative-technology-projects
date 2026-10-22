"""
merge_datasets.py
=================
Merge two YOLOv8-format datasets into one, with a unified class table.

MyFridge combines:
  - ds_ready : a public Roboflow "Fridge Objects" dataset (10 classes)
  - ds_mine  : our own self-captured & labelled dataset (50 classes)

The two datasets number their classes differently (e.g. "milk" may be id 6 in
one and id 20 in the other). Naively concatenating them would corrupt the
labels. This script builds a single unified class list, then REMAPS every
label file's class id to the unified id, and copies all images + remapped
labels into /content/merged.

Output:
  /content/merged/
    train/images, train/labels
    valid/images, valid/labels
    test/images,  test/labels
    data.yaml   (unified class list, ready for train_merged.py)

Usage (in Colab, after unzipping both datasets into ds_ready/ and ds_mine/):
    pip install pyyaml
    python merge_datasets.py
"""

import os
import glob
import shutil
import yaml


# ----------------------------------------------------------------------------
# Configuration — point these at the two unzipped dataset folders
# ----------------------------------------------------------------------------
YAML_READY = "ds_ready/data.yaml"   # public Roboflow dataset
YAML_MINE = "ds_mine/data.yaml"     # our own dataset
OUT = "merged"                      # output folder
SPLITS = ["train", "valid", "test"]


def load_names(yaml_path: str) -> list:
    """Read the class-name list from a YOLO data.yaml (handles list or dict)."""
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    names = d["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return names


def process(ds_dir: str, mapping: dict, tag: str) -> None:
    """Copy images and remap each label's class id into the merged folder.

    `tag` prefixes filenames (e.g. "r_", "m_") so files from the two datasets
    never collide.
    """
    for split in SPLITS:
        img_dir = f"{ds_dir}/{split}/images"
        lbl_dir = f"{ds_dir}/{split}/labels"
        if not os.path.isdir(img_dir):
            continue
        for img in glob.glob(f"{img_dir}/*"):
            base = os.path.basename(img)
            stem = os.path.splitext(base)[0]
            shutil.copy(img, f"{OUT}/{split}/images/{tag}_{base}")

            lbl = f"{lbl_dir}/{stem}.txt"
            if os.path.exists(lbl):
                out_lines = []
                for line in open(lbl):
                    parts = line.split()
                    if not parts:
                        continue
                    parts[0] = str(mapping[int(parts[0])])  # remap class id
                    out_lines.append(" ".join(parts))
                with open(f"{OUT}/{split}/labels/{tag}_{stem}.txt", "w") as f:
                    f.write("\n".join(out_lines))


def main() -> None:
    names_ready = load_names(YAML_READY)
    names_mine = load_names(YAML_MINE)
    print(f"ready dataset ({len(names_ready)} classes):", names_ready)
    print(f"mine dataset  ({len(names_mine)} classes):", names_mine)

    # Build a unified class list, de-duplicating by name (order preserved).
    unified = []
    for n in names_ready + names_mine:
        if n not in unified:
            unified.append(n)
    print(f"\nunified ({len(unified)} classes):", unified)

    # old id -> new id maps for each source dataset
    map_ready = {i: unified.index(n) for i, n in enumerate(names_ready)}
    map_mine = {i: unified.index(n) for i, n in enumerate(names_mine)}

    # Prepare output dirs
    for split in SPLITS:
        os.makedirs(f"{OUT}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUT}/{split}/labels", exist_ok=True)

    process("ds_ready", map_ready, "r")
    process("ds_mine", map_mine, "m")

    # Write the merged data.yaml
    data = {
        "train": f"/content/{OUT}/train/images",
        "val": f"/content/{OUT}/valid/images",
        "test": f"/content/{OUT}/test/images",
        "nc": len(unified),
        "names": unified,
    }
    with open(f"{OUT}/data.yaml", "w") as f:
        yaml.dump(data, f, allow_unicode=True)

    print("\nMerge complete.")
    print("train:", len(glob.glob(f"{OUT}/train/images/*")), "images")
    print("valid:", len(glob.glob(f"{OUT}/valid/images/*")), "images")


if __name__ == "__main__":
    main()