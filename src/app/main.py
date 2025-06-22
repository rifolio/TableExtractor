import numpy as np
import pandas as pd
import matplotlib
from PIL import Image
import torch, torchaudio, torchvision
import transformers
import huggingface_hub
import easyocr
from tqdm import tqdm

def main():
    print("=== Library Versions ===")
    print(f"numpy:           {np.__version__}")
    print(f"pandas:          {pd.__version__}")
    print(f"matplotlib:      {matplotlib.__version__}")
    print(f"Pillow:          {Image.__version__}")
    print(f"torch:           {torch.__version__}")
    print(f"torchaudio:      {torchaudio.__version__}")
    print(f"torchvision:     {torchvision.__version__}")
    print(f"transformers:    {transformers.__version__}")
    print(f"huggingface_hub: {huggingface_hub.__version__}")
    print(f"easyocr:         {easyocr.__version__}")
    print("\n=== Quick CPU Tests ===")
    # Numpy array
    arr = np.linspace(0, 1, 5)
    print("numpy linspace:", arr)

    # Torch tensor
    t = torch.arange(5)
    print("torch tensor:", t)

    # TQDM progress bar
    print("tqdm demo:")
    for i in tqdm(range(3), desc="Loop"):
        pass

    print("\nAll checks passed!")

if __name__ == "__main__":
    main()
