import json
import os

MIN_SIZE = 1.0  # minimum width/height in pixels to keep

def fix_file(path, out_path, report_limit=10):
    with open(path, 'r') as f:
        data = json.load(f)

    images = {img['id']:{'file_name':img.get('file_name'), 'width':img.get('width'), 'height':img.get('height')} for img in data.get('images', [])}

    fixed_annotations = []
    removed = 0
    adjusted = 0
    problems = []
    for ann in data.get('annotations', []):
        bbox = ann.get('bbox')
        img_id = ann.get('image_id')
        if bbox is None or len(bbox)!=4:
            removed += 1
            continue
        x,y,w,h = bbox
        if not all(isinstance(v, (int,float)) for v in (x,y,w,h)):
            removed += 1
            continue
        if w <= 0 or h <= 0:
            removed += 1
            continue
        img = images.get(img_id)
        if img and img.get('width') and img.get('height'):
            iw = img['width']; ih = img['height']
            # Clip
            new_x = max(0.0, x)
            new_y = max(0.0, y)
            new_w = w
            new_h = h
            if new_x + new_w > iw:
                new_w = max(0.0, iw - new_x)
            if new_y + new_h > ih:
                new_h = max(0.0, ih - new_y)
            # Small epsilon fix for floating rounding > image size
            if new_w != w or new_h != h or new_x != x or new_y != y:
                adjusted += 1
                ann['bbox'] = [new_x, new_y, new_w, new_h]
            # discard degenerate
            if new_w < MIN_SIZE or new_h < MIN_SIZE:
                removed += 1
                if len(problems) < report_limit:
                    problems.append((img_id, 'degenerate_after_clipping', ann['bbox']))
                continue
            fixed_annotations.append(ann)
        else:
            # no image size info — keep but hope for the best
            fixed_annotations.append(ann)

    data['annotations'] = fixed_annotations
    with open(out_path, 'w') as f:
        json.dump(data, f)

    print(f"Processed {os.path.basename(path)}: original anns={len(data.get('annotations', [])) + removed}, kept={len(fixed_annotations)}, removed={removed}, adjusted={adjusted}")
    if problems:
        print('Sample problems:')
        for p in problems:
            print(p)

if __name__ == '__main__':
    base = os.path.join('data','trashcan','annotations')
    train = os.path.join(base,'train.json')
    val = os.path.join(base,'val.json')
    out_train = os.path.join(base,'train_fixed.json')
    out_val = os.path.join(base,'val_fixed.json')
    print('Fixing train...')
    fix_file(train, out_train)
    print('Fixing val...')
    fix_file(val, out_val)
    print('Done. Backup originals if you want and replace train.json/val.json with fixed files after review.')
