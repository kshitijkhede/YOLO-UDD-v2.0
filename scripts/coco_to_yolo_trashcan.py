"""
Convert TrashCan 1.0 (COCO instance format) to YOLO detection format for
Ultralytics, collapsing the ~22 fine-grained categories into the 3-class
config {trash:0, animal:1, rov:2} by name prefix.

Output layout (under --out-dir, default /kaggle/working/trashcan_yolo):

    trashcan_yolo/
      images/train -> symlink to the source train images
      images/val   -> symlink to the source val images
      labels/train/<stem>.txt   (one YOLO line per box: cls xc yc w h, normalized)
      labels/val/<stem>.txt
      data.yaml

Usage:
    python scripts/coco_to_yolo_trashcan.py \
        --dataset-dir /kaggle/input/datasets/mexwell/trashcan-1-0/dataset/instance_version \
        --out-dir /kaggle/working/trashcan_yolo
"""

import os
import json
import shutil
import argparse


CLASS_NAMES = ["trash", "animal", "rov"]


def name_to_class(name):
    """Map a (possibly fine-grained) COCO category name to 0/1/2, or None to skip."""
    name = str(name).lower().strip()
    if name in ("trash", "animal", "rov"):
        return {"trash": 0, "animal": 1, "rov": 2}[name]
    if name.startswith("rov"):
        return 2
    if name.startswith("animal"):
        return 1
    if name.startswith("trash"):
        return 0
    return None  # 'plant' and anything else is not in the 3-class config


def find_image_dir(dataset_dir, split):
    """Return the source image directory for a split across common layouts."""
    candidates = [
        os.path.join(dataset_dir, "images", split),
        os.path.join(dataset_dir, split),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"Could not find images for split '{split}'. Tried: {candidates}"
    )


def find_ann_file(dataset_dir, split):
    candidates = [
        os.path.join(dataset_dir, f"instances_{split}_trashcan.json"),
        os.path.join(dataset_dir, "annotations", f"{split}.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Could not find annotations for split '{split}'. Tried: {candidates}"
    )


def convert_split(dataset_dir, out_dir, split):
    ann_file = find_ann_file(dataset_dir, split)
    img_dir = find_image_dir(dataset_dir, split)
    print(f"[{split}] annotations: {ann_file}")
    print(f"[{split}] images:      {img_dir}")

    with open(ann_file, "r") as f:
        coco = json.load(f)

    # cat_id -> 3-class label
    cat_map = {}
    for c in coco.get("categories", []):
        cat_map[c["id"]] = name_to_class(c.get("name", ""))

    images = {im["id"]: im for im in coco["images"]}

    # group annotations by image
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    # label output dir
    lbl_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(lbl_dir, exist_ok=True)

    # COPY images into out_dir/images/<split>. We copy real files (not a symlink)
    # because Ultralytics resolves symlinks to their real path and then derives
    # label paths by swapping '/images/' -> '/labels/'; the source images live
    # under '.../<split>/' (no 'images' component), so a symlink breaks label
    # discovery. Copying keeps the images/ <-> labels/ parallel layout intact.
    img_out = os.path.join(out_dir, "images", split)
    os.makedirs(img_out, exist_ok=True)
    shutil.copytree(img_dir, img_out, dirs_exist_ok=True)

    n_boxes, n_skipped, n_imgs = 0, 0, 0
    for img_id, im in images.items():
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        W, H = im.get("width"), im.get("height")
        lines = []
        for a in anns_by_img.get(img_id, []):
            cls = cat_map.get(a["category_id"])
            if cls is None:
                n_skipped += 1
                continue
            x, y, w, h = a["bbox"]  # COCO: top-left x,y + w,h (absolute px)
            if w <= 0 or h <= 0:
                continue
            xc = (x + w / 2) / W
            yc = (y + h / 2) / H
            nw = w / W
            nh = h / H
            # clip to [0,1]
            xc, yc = min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)
            nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
            if nw < 1e-4 or nh < 1e-4:
                continue
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            n_boxes += 1
        # write label file (empty file for images with no in-config boxes)
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        n_imgs += 1

    print(f"[{split}] wrote {n_imgs} label files, {n_boxes} boxes "
          f"({n_skipped} out-of-config boxes skipped)")
    return img_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True,
                    help="TrashCan instance_version dir (has instances_*_trashcan.json)")
    ap.add_argument("--out-dir", default="/kaggle/working/trashcan_yolo")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_imgs = convert_split(args.dataset_dir, args.out_dir, "train")
    val_imgs = convert_split(args.dataset_dir, args.out_dir, "val")

    # data.yaml: paths relative to out_dir so the images/ <-> labels/ layout
    # resolves cleanly for Ultralytics.
    data_yaml = os.path.join(args.out_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {args.out_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")
    print(f"\nWrote {data_yaml}")
    with open(data_yaml) as f:
        print(f.read())


if __name__ == "__main__":
    main()
