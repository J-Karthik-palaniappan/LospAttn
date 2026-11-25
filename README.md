# LoSpAttn: Fast Training-Free Hybrid Attention for Vision and Diffusion Transformers

Code for generating images with pretrained Diffusion Transformers using LoSpAttn. No training is needed; Calibration csv files provided.

| `--model` | Model | Script |
|---|---|---|
| `dit256` | DiT-XL/2, 256×256, ImageNet class-conditional | `sample_dit.py` |
| `dit512` | DiT-XL/2, 512×512, ImageNet class-conditional | `sample_dit.py` |
| `pixart1k` | PixArt-Σ XL/2, 1024×1024, text-to-image | `sample_pixart.py` |
| `pixart2k` | PixArt-Σ XL/2, 2048×2048, text-to-image | `sample_pixart.py` |

---

## Installation

```bash
conda create -n lospattn python=3.10 -y
conda activate lospattn

pip install torch                     
pip install diffusers transformers accelerate huggingface_hub sentencepiece
pip install triton einops numpy pandas
# install prebuilt wheel for easier setup
pip install flash-attn --no-build-isolation
pip install beautifulsoup4 ftfy       
```


## Image Generation

Run all commands from the repository root.

### DiT

```bash
# LoSpAttn
python sample_dit.py --model dit256
python sample_dit.py --model dit512

# Original model (for comparison)
python sample_dit.py --model dit256 --raw
python sample_dit.py --model dit512 --raw
```

### PixArt-Σ

```bash
# LoSpAttn
python sample_pixart.py --model pixart1k --window 512
python sample_pixart.py --model pixart2k --window 2048

# Original model (for comparison)
python sample_pixart.py --model pixart1k --window 512 --raw
python sample_pixart.py --model pixart2k --window 2048 --raw
```

Images are saved as PNGs:
Default location: `images/<model>/`

Each run also prints how much of the attention LoSpAttn replaced:

```
Lowrank: 41.23%
Sparse:  12.05%
Softmax: 46.72%
```