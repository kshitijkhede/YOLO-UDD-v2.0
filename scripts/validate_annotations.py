import json
import os

def check_annotations(json_path, max_report=20):
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']:{'file_name':img.get('file_name'), 'width':img.get('width'), 'height':img.get('height')} for img in data.get('images', [])}

    problems = []
    total = 0
    for ann in data.get('annotations', []):
        total += 1
        bbox = ann.get('bbox')
        img_id = ann.get('image_id')
        if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            problems.append((img_id, 'invalid_bbox_format', ann))
            continue
        x, y, w, h = bbox
        # check numeric
        if any(not isinstance(v, (int, float)) for v in (x,y,w,h)):
            problems.append((img_id, 'non_numeric', bbox))
            continue
        if w <= 0 or h <= 0:
            problems.append((img_id, 'non_positive_wh', bbox))
            continue
        if x < 0 or y < 0:
            problems.append((img_id, 'negative_xy', bbox))
        img_info = images.get(img_id)
        if img_info:
            iw = img_info.get('width')
            ih = img_info.get('height')
            if iw is None or ih is None:
                # can't check bounds
                pass
            else:
                if x + w <= 0 or y + h <= 0:
                    problems.append((img_id, 'bbox_outside_nonpositive', bbox))
                if x + w > iw or y + h > ih:
                    problems.append((img_id, 'bbox_exceeds_image', {'bbox':bbox, 'img_w':iw, 'img_h':ih}))
    
    print(f"Checked {total} annotations in {os.path.basename(json_path)}")
    print(f"Problems found: {len(problems)}")
    if problems:
        print('\nFirst problems:')
        for i, p in enumerate(problems[:max_report]):
            print(i+1, p)
    return len(problems)

if __name__ == '__main__':
    base = os.path.join('data','trashcan','annotations')
    train = os.path.join(base,'train.json')
    val = os.path.join(base,'val.json')
    for path in [train, val]:
        if os.path.exists(path):
            check_annotations(path)
        else:
            print('File not found:', path)
