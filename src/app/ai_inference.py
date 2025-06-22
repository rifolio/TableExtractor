import argparse
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from huggingface_hub import snapshot_download
from torchvision import transforms


def ensure_local_model(repo_id: str, folder_name: str) -> Path:
    """
    Ensure that `models/<folder_name>/` exists, downloading it if necessary.
    Returns the Path to the local model directory.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    local_dir = project_root / 'models' / folder_name

    if not local_dir.exists():
        print(f"📥 Downloading {repo_id} to {local_dir} …")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )
        print("✅ Download complete\n")
    else:
        print(f"✅ Found local model at {local_dir}\n")

    return local_dir


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    soft = outputs.logits.softmax(-1)
    scores, indices = soft.max(-1)
    pred_labels = indices[0].cpu().numpy()
    pred_scores = scores[0].cpu().numpy()
    raw_bboxes = outputs.pred_boxes[0].cpu()
    bboxes = [list(map(float, box)) for box in rescale_bboxes(raw_bboxes, img_size)]

    objects = []
    for lbl, scr, bbox in zip(pred_labels, pred_scores, bboxes):
        label = id2label.get(int(lbl), 'unknown')
        if label != 'no object':
            objects.append({'label': label, 'score': float(scr), 'bbox': bbox})
    return objects


def objects_to_crops(img, objects, class_thresholds, padding=10):
    crops = []
    for obj in objects:
        if obj['score'] < class_thresholds.get(obj['label'], 0.5):
            continue
        bbox = obj['bbox']
        padded = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]
        crops.append(img.crop(padded))
    return crops


def extract_table_structure_data(cells):
    rows = [c for c in cells if c['label'] == 'table row']
    cols = [c for c in cells if c['label'] == 'table column']
    return {'table_rows': rows, 'table_columns': cols}


def main(image_paths):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Ensure models are present locally
    det_dir = ensure_local_model(
        repo_id="microsoft/table-transformer-detection",
        folder_name="table-transformer-detection"
    )
    struct_dir = ensure_local_model(
        repo_id="microsoft/table-structure-recognition-v1.1-all",
        folder_name="table-structure-recognition"
    )

    # 2) Load table detection model from local path
    print("Loading table detection model from disk…")
    det_model = AutoModelForObjectDetection.from_pretrained(
        str(det_dir), local_files_only=True, revision="no_timm"
    )
    det_model.to(device)
    det_id2label = det_model.config.id2label.copy()
    det_id2label[len(det_id2label)] = 'no object'

    # Detection transform
    class MaxResize:
        def __init__(self, max_size=800): self.max_size = max_size
        def __call__(self, image):
            w, h = image.size
            scale = self.max_size / max(w, h)
            return image.resize((int(w*scale), int(h*scale)))
    det_transform = transforms.Compose([
        MaxResize(800), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # 3) Load table structure model from local path
    print("Loading table structure model from disk…")
    struct_model = TableTransformerForObjectDetection.from_pretrained(
        str(struct_dir), local_files_only=True
    )
    struct_model.to(device)
    struct_id2label = struct_model.config.id2label.copy()
    struct_id2label[len(struct_id2label)] = 'no object'

    struct_transform = transforms.Compose([
        MaxResize(1000), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    det_thresholds = {'table': 0.5, 'table rotated': 0.5}

    # 4) Inference loop
    for path in image_paths:
        print(f"\nProcessing {path}")
        img = Image.open(path).convert("RGB")

        # Table detection
        pixels = det_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            det_out = det_model(pixels)
        tables = outputs_to_objects(det_out, img.size, det_id2label)
        print(f"Detected {len(tables)} table region(s):", tables)

        # Crop and structure recognition
        crops = objects_to_crops(img, tables, det_thresholds)
        for i, crop in enumerate(crops):
            print(f"\nAnalyzing structure for table {i}")
            px = struct_transform(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                struct_out = struct_model(px)
            cells = outputs_to_objects(struct_out, crop.size, struct_id2label)
            struct = extract_table_structure_data(cells)
            print(f"Rows: {len(struct['table_rows'])}, Columns: {len(struct['table_columns'])}")

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Table detection and structure inference'
    )
    parser.add_argument('images', nargs='+', help='Paths to PNG images to process')
    args = parser.parse_args()
    main(args.images)
