# PhysX-Anything on Google Colab

Generate a simulation-ready 3D asset (URDF/MJCF) from a single iron image.

## Prerequisites
- Colab with **GPU runtime** (Runtime → Change runtime type → T4/A100 GPU)
- Upload your `iron_frameXXX.png` (RGBA cutout from `prep_iron.py`)

## Steps (run each cell in Colab)

### 1. Check GPU
```python
!nvidia-smi
```

### 2. Clone PhysX-Anything
```python
!git clone --recurse-submodules https://github.com/ziangcao0312/PhysX-Anything.git
%cd PhysX-Anything
```

### 3. Install dependencies
```python
# Their setup script (may take 15-20 min — heavy CUDA builds)
!pip install -r requirements.txt
!pip install transformers==4.50.0 qwen-vl-utils 'accelerate>=0.26.0'
```

### 4. Download pretrained models
```python
!python download.py
```

### 5. Upload your iron images (both RGB and RGBA)
```python
from google.colab import files
uploaded = files.upload()   # upload iron_f950_rgb.png AND iron_f950_rgba.png
!mkdir -p demo_rgb demo_rgba
!mv iron_f950_rgb.png  demo_rgb/
!mv iron_f950_rgba.png demo_rgba/
```

### 6. Run the pipeline — TWICE (RGB with bg-removal, RGBA without)

**Run A — RGB bbox crop (let PhysX remove background):**
```python
!python 1_vlm_demo.py \
    --demo_path ./demo_rgb \
    --save_part_ply True \
    --remove_bg True \
    --ckpt ./pretrain/vlm
!python 2_decoder.py
!python 3_split.py
!python 4_simready_gen.py \
    --voxel_define 32 --basepath ./test_rgb \
    --process 0 --fixed_base 0 --deformable 0
```

**Run B — RGBA masked crop (background already removed):**
```python
!python 1_vlm_demo.py \
    --demo_path ./demo_rgba \
    --save_part_ply True \
    --remove_bg False \
    --ckpt ./pretrain/vlm
!python 2_decoder.py
!python 3_split.py
!python 4_simready_gen.py \
    --voxel_define 32 --basepath ./test_rgba \
    --process 0 --fixed_base 0 --deformable 0
```

### 7. Download both results
```python
!zip -r iron_rgb_asset.zip  ./test_rgb
!zip -r iron_rgba_asset.zip ./test_rgba
from google.colab import files
files.download('iron_rgb_asset.zip')
files.download('iron_rgba_asset.zip')
```

## Expected Output
- `iron.urdf` — MuJoCo/robotics-ready
- `iron.xml` (MJCF)
- Part meshes (.ply/.obj)
- Physical parameters (mass, material)

## Notes / Likely Issues
- **Install may fail** on some CUDA-specific packages (kaolin, nvdiffrast, spconv,
  diffoctreerast). Colab's CUDA version must match. If it fails, we may need the
  full GPU pod with a controlled CUDA image.
- **VRAM:** T4 (16GB) may be tight for the 7B VLM + TRELLIS. A100 (Colab Pro)
  is safer.
- The `--remove_bg False` flag is correct because our input is already RGBA
  (background removed via SAM2 mask).
