#!/usr/bin/env python3
import sys
from pathlib import Path
from pdf_reader import PDFReader
import ai_inference

def main():
    # Project root is three levels up from scr/app/main.py
    base_dir = Path(__file__).resolve().parent.parent.parent

    # Locate the test PDF in <project-root>/pdfs/test.pdf
    pdf_file = base_dir / 'pdfs' / 'test.pdf'
    if not pdf_file.is_file():
        print(f"Error: test PDF not found at {pdf_file}", file=sys.stderr)
        sys.exit(1)

    try:
        # Convert PDF to images
        reader = PDFReader(str(pdf_file))
        images = reader.convert_to_images()
        print(f"Converted '{pdf_file.name}' into {len(images)} images:")
        for img_path in images:
            print(f"  • {img_path}")

        # Run AI inference on generated images
        print("\nRunning AI inference on generated images...")
        ai_inference.main([str(p) for p in images])

    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
