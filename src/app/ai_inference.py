import argparse
import os
from PIL import Image
import torch
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from huggingface_hub import hf_hub_download
from torchvision import transforms


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
    # Get class predictions and boxes
    soft = outputs.logits.softmax(-1)
    scores, indices = soft.max(-1)
    pred_labels = indices[0].cpu().numpy()
    pred_scores = scores[0].cpu().numpy()
    raw_bboxes = outputs.pred_boxes[0].cpu()
    bboxes = [list(map(float, box)) for box in rescale_bboxes(raw_bboxes, img_size)]

    # Map to objects
    objects = []
    for lbl, scr, bbox in zip(pred_labels, pred_scores, bboxes):
        label = id2label.get(int(lbl), 'unknown')
        if label != 'no object':
            objects.append({'label': label, 'score': float(scr), 'bbox': bbox})
    return objects


def objects_to_crops(img, objects, class_thresholds, padding=10):
    """
    Crop out table regions based on detection outputs.
    """
    crops = []
    for obj in objects:
        if obj['score'] < class_thresholds.get(obj['label'], 0.5):
            continue
        bbox = obj['bbox']
        padded = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]
        cropped_img = img.crop(padded)
        crops.append(cropped_img)
    return crops


def extract_table_structure_data(cells):
    """
    Given a list of cell/row/column detections,
    return dict with keys 'table_rows' and 'table_columns'.
    """
    rows = [c for c in cells if c['label'] == 'table row']
    cols = [c for c in cells if c['label'] == 'table column']
    return {'table_rows': rows, 'table_columns': cols}


def main(image_paths):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load table detection model
    print("Loading table detection model...")
    det_model = AutoModelForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection", revision="no_timm"
    )
    det_model.to(device)
    det_id2label = det_model.config.id2label.copy()
    det_id2label[len(det_id2label)] = 'no object'

    # Transform for detection
    class MaxResize:
        def __init__(self, max_size=800): self.max_size = max_size
        def __call__(self, image):
            w, h = image.size
            max_dim = max(w, h)
            scale = self.max_size / max_dim
            return image.resize((int(w*scale), int(h*scale)))
    det_transform = transforms.Compose([
        MaxResize(800), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Load table structure model
    print("Loading table structure model...")
    struct_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-structure-recognition-v1.1-all"
    )
    struct_model.to(device)
    struct_id2label = struct_model.config.id2label.copy()
    struct_id2label[len(struct_id2label)] = 'no object'

    # Transform for structure
    struct_transform = transforms.Compose([
        MaxResize(1000), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Thresholds
    det_thresholds = {'table': 0.5, 'table rotated': 0.5}

    for path in image_paths:
        print(f"\nProcessing {path}")
        img = Image.open(path).convert("RGB")
        # Detect tables
        pixels = det_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            det_out = det_model(pixels)
        tables = outputs_to_objects(det_out, img.size, det_id2label)
        print(f"Detected {len(tables)} table region(s):")
        for t in tables:
            print(t)

        # Crop and analyze each table region
        crops = objects_to_crops(img, tables, det_thresholds, padding=10)
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
    parser = argparse.ArgumentParser(description='Table detection and structure inference')
    parser.add_argument('images', nargs='+', help='Paths to PNG images to process')
    args = parser.parse_args()
    main(args.images)
