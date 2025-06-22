import sys
from pathlib import Path
from pdf_reader import PDFReader

def main():
    # project root is three levels up from scr/app/main.py
    base_dir = Path(__file__).resolve().parent.parent.parent

    # point at your test PDF in <project-root>/pdfs/test.pdf
    pdf_file = base_dir / 'pdfs' / 'test.pdf'
    if not pdf_file.is_file():
        print(f"Error: test PDF not found at {pdf_file}", file=sys.stderr)
        sys.exit(1)

    try:
        reader = PDFReader(str(pdf_file))
        # by default this writes into <project-root>/images/<pdf_name>/
        images = reader.convert_to_images()
        print(f"Converted '{pdf_file.name}' into {len(images)} images:")
        for img_path in images:
            print(f"  • {img_path}")
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
