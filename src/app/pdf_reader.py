import os
from pathlib import Path
import fitz          # PyMuPDF
from PIL import Image
from io import BytesIO

class PDFReader:
    """
    Convert pages of a PDF file into individual image files using PyMuPDF.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        self.pdf_name = self.pdf_path.stem

    def convert_to_images(
        self,
        output_root: Path = None,
        dpi: int = 200,
        fmt: str = 'png'
    ) -> list[Path]:
        """
        Render the PDF pages to images and save them.

        Args:
            output_root (Path, optional): Base folder for `images/`. Defaults to project-root/images/.
            dpi (int): Rendering resolution. Defaults to 200.
            fmt (str): Output format (`png`, `jpeg`, etc.).

        Returns:
            List[Path]: Saved image file paths.
        """
        # efault output directory
        if output_root is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            output_root = base_dir / 'images'

        output_dir = output_root / self.pdf_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # oepn PDF
        doc = fitz.open(str(self.pdf_path))
        saved_paths = []

        # calc transformation matrix for DPI
        zoom = dpi / 72  # as MuPDF uses 72 dpi as base
        mat = fitz.Matrix(zoom, zoom)

        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # to bytes then open with PIL to ensure consistent API
            img_data = pix.tobytes(output=fmt)
            img = Image.open(BytesIO(img_data))

            filename = f"{self.pdf_name}_page_{page_number+1}.{fmt}"
            file_path = output_dir / filename
            img.save(file_path, fmt.upper())

            saved_paths.append(file_path)

        return saved_paths

    @classmethod
    def convert_all_in_directory(
        cls,
        pdfs_dir: Path = None,
        images_root: Path = None,
        dpi: int = 200
    ) -> dict[str, list[Path]]:
        """
        Batch-convert all PDFs in a directory.
        """
        if pdfs_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            pdfs_dir = base_dir / 'pdfs'
        if images_root is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            images_root = base_dir / 'images'

        if not pdfs_dir.is_dir():
            raise NotADirectoryError(f"PDFs directory not found: {pdfs_dir}")

        results = {}
        for pdf_file in pdfs_dir.glob('*.pdf'):
            reader = cls(str(pdf_file))
            images = reader.convert_to_images(output_root=images_root, dpi=dpi)
            results[pdf_file.name] = images

        return results


if __name__ == '__main__':
    # Example usage:
    try:
        res = PDFReader.convert_all_in_directory()
        for name, imgs in res.items():
            print(f"→ {name}: {len(imgs)} pages → images/{Path(name).stem}/")
    except Exception as e:
        print(f"Error: {e}")
