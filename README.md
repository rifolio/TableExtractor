# PDF Table Detection and Structure Recognition

This project uses AI models to detect and analyze table structures in PDF documents. It converts PDF pages to images and then uses Microsoft's Table Transformer models to identify tables and their structural elements (rows, columns, cells).

## Features

- **PDF to Image Conversion**: Converts PDF pages to high-quality images using PyMuPDF
- **Table Detection**: Identifies table regions in images using Microsoft's table-transformer-detection model
- **Structure Recognition**: Analyzes table structure (rows, columns, cells) using Microsoft's table-structure-recognition model
- **Local Model Management**: Automatically downloads and caches Hugging Face models locally
- **Docker Support**: Containerized application for easy deployment

## Prerequisites

- Python 3.10+
- Docker (optional, for containerized execution)
- At least 4GB RAM (for model loading)

## Project Structure

```
├── src/
│   └── app/
│       ├── main.py              # Main application entry point
│       ├── pdf_reader.py        # PDF to image conversion
│       ├── ai_inference.py      # AI model inference
│       └── __init__.py
├── pdfs/                        # Input PDF files
├── images/                      # Generated images from PDFs
├── models/                      # Downloaded AI models
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
└── test.ipynb                  # Model download script
```

## Installation

### Option 1: Local Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/rifolio/TableExtractor/
   cd TableExtractor
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download AI models** (first time only):

   Run the model download script:

   ```bash
   python -c "
   from huggingface_hub import snapshot_download
   import os

   repos = {
       'table-transformer-detection': 'microsoft/table-transformer-detection',
       'table-structure-recognition': 'microsoft/table-structure-recognition-v1.1-all'
   }

   for folder, repo_id in repos.items():
       local_path = f'models/{folder}'
       print(f'Downloading {repo_id} to {local_path}...')
       snapshot_download(repo_id=repo_id, local_dir=local_path)
   "
   ```

### Option 2: Docker Installation

1. **Build the Docker image**:

   ```bash
   docker build -t pdf-table-detector .
   ```

2. **Run with Docker Compose** (recommended):
   ```bash
   docker-compose up --build
   ```

## Usage

### Running the Application

#### Local Execution

1. **Place your PDF file** in the `pdfs/` directory:

   ```bash
   cp your_document.pdf pdfs/test.pdf
   ```

2. **Run the main application**:
   ```bash
   python src/app/main.py
   ```

#### Docker Execution

1. **Place your PDF file** in the `pdfs/` directory:

   ```bash
   cp your_document.pdf pdfs/test.pdf
   ```

2. **Run with Docker**:

   ```bash
   # Using Docker Compose (recommended)
   docker-compose up

   # Or using Docker directly
   docker run -v $(pwd):/app pdf-table-detector
   ```

3. **Run main from terminal** (after Docker image is ready):

   ```bash
   # Interactive shell in container
   docker run -it -v $(pwd):/app pdf-table-detector /bin/bash

   # Then inside the container:
   python src/app/main.py

   # Or run directly:
   docker run -v $(pwd):/app pdf-table-detector python src/app/main.py
   ```

### Expected Output

The application will:

1. Convert the PDF to images (saved in `images/test/`)
2. Detect table regions in each image
3. Analyze table structure (rows, columns)
4. Print detection results to console

Example output:

```
Converted 'test.pdf' into 2 images:
  • /app/images/test/test_page_1.png
  • /app/images/test/test_page_2.png

Running AI inference on generated images...
✅ Found local model at /app/models/table-transformer-detection
✅ Found local model at /app/models/table-structure-recognition

Processing /app/images/test/test_page_1.png
Detected 1 table region(s): [{'label': 'table', 'score': 0.95, 'bbox': [100, 200, 500, 400]}]

Analyzing structure for table 0
Rows: 5, Columns: 3
```

## Configuration

### Environment Variables

- `OMP_NUM_THREADS`: Number of OpenMP threads (default: 4)
- `MKL_NUM_THREADS`: Number of MKL threads (default: 4)
- `PYTHONUNBUFFERED`: Force Python to flush output immediately (set to 1)

### Model Settings

- **Detection Threshold**: 0.5 (configurable in `ai_inference.py`)
- **Image Resolution**: 200 DPI for PDF conversion
- **Max Image Size**: 800px for detection, 1000px for structure recognition

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `OMP_NUM_THREADS` and `MKL_NUM_THREADS` in docker-compose.yml
2. **Model Download Fails**: Check internet connection and try running the download script manually
3. **PDF Not Found**: Ensure your PDF is named `test.pdf` and placed in the `pdfs/` directory

### Performance Tips

- Use GPU if available (modify Dockerfile to use CUDA base image)
- Adjust thread counts based on your system capabilities
- For large PDFs, consider processing pages individually

## Dependencies

- **PyTorch**: Deep learning framework (CPU-only version)
- **Transformers**: Hugging Face transformers library
- **PyMuPDF**: PDF processing
- **Pillow**: Image processing
- **EasyOCR**: OCR capabilities (optional)
- **Matplotlib**: Visualization
- **Pandas**: Data manipulation

## License

MIT
