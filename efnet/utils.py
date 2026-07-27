"""
efnet/utils.py
==============
Utility functions:
  • Face alignment (MTCNN or SCRFD via InsightFace)
  • CASIA-WebFace / LFW download helpers for Colab
  • Custom dataset builder (capture from webcam / files)
  • Image quality filtering (blur detection)
  • Colab-specific helpers (GPU check, Drive mount)
"""

import os
import io
import cv2
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch


# ---------------------------------------------------------------------------
# GPU & Environment helpers
# ---------------------------------------------------------------------------

def check_environment():
    """Print GPU info and recommend settings."""
    print("=" * 55)
    print("  EFNet Environment Check")
    print("=" * 55)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU      : {gpu}")
        print(f"  VRAM     : {mem:.1f} GB")
        if mem >= 14:
            print("  Recommended batch size: 256  (A100/V100)")
        elif mem >= 6:
            print("  Recommended batch size: 128  (T4/P100)")
        else:
            print("  Recommended batch size: 64   (small GPU)")
    else:
        print("  GPU      : NOT AVAILABLE — training will be very slow")
        print("  → In Colab: Runtime > Change runtime type > GPU")

    try:
        import torchvision
        print(f"  PyTorch  : {torch.__version__}")
        print(f"  torchvision: {torchvision.__version__}")
    except Exception:
        pass

    print("=" * 55)


def mount_drive():
    """Mount Google Drive in Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted at /content/drive")
    except ImportError:
        print("Not running in Colab — Google Drive mount skipped.")


# ---------------------------------------------------------------------------
# Dataset download helpers
# ---------------------------------------------------------------------------

def download_lfw(dest_dir: str = '/content/data') -> str:
    """
    Download and extract LFW (Labeled Faces in the Wild).

    LFW is freely available and ~173 MB. We use the funneled (aligned)
    version which gives better baseline accuracy.

    Returns:
        Path to LFW root directory.
    """
    os.makedirs(dest_dir, exist_ok=True)
    lfw_root = os.path.join(dest_dir, 'lfw_funneled')

    if os.path.exists(lfw_root):
        print(f"[LFW] Already exists at '{lfw_root}'")
        return lfw_root

    url  = 'http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz'
    tgz  = os.path.join(dest_dir, 'lfw-funneled.tgz')
    pairs_url = 'http://vis-www.cs.umass.edu/lfw/pairs.txt'

    print("[LFW] Downloading (173 MB)...")
    subprocess.run(['wget', '-q', '-O', tgz, url], check=True)
    print("[LFW] Extracting...")
    subprocess.run(['tar', '-xzf', tgz, '-C', dest_dir], check=True)
    os.remove(tgz)

    pairs_path = os.path.join(dest_dir, 'lfw_pairs.txt')
    subprocess.run(['wget', '-q', '-O', pairs_path, pairs_url], check=True)

    print(f"[LFW] Ready at '{lfw_root}'  pairs → '{pairs_path}'")
    return lfw_root


def download_casia_webface_instructions():
    """Print instructions for downloading CASIA-WebFace."""
    msg = """
CASIA-WebFace Download Instructions
=====================================
CASIA-WebFace (~3.4 GB, 494K images, 10572 identities) is a research
dataset — requires an institutional email to request access.

Option 1 — InsightFace's cleaned MS1MV3 (recommended, 5.5 GB):
    # In Colab:
    !pip install gdown
    !gdown --folder 1dswD2B9nCgPxHpJP2MN-79DjFlvMaHJV -O /content/data/ms1mv3

    This downloads the .rec + .idx files. Then set in TrainConfig:
        data_format = 'rec'
        rec_path    = '/content/data/ms1mv3/train.rec'
        idx_path    = '/content/data/ms1mv3/train.idx'

Option 2 — Use a pre-extracted folder version (if you have the files):
    Upload to Google Drive, then:
        from google.colab import drive
        drive.mount('/content/drive')
        # Your data should be at /content/drive/MyDrive/casia_webface/

Option 3 — Smaller test run with VGGFace2 subset:
    VGGFace2 has a freely downloadable test set (~1.5 GB, 500 identities).
    Good for validating your pipeline before scaling up.

Option 4 — CASIA-WebFace from Kaggle:
    kaggle datasets download -d atulanandjha/lfwpeople
    (This gives LFW, not CASIA, but sufficient for initial testing)

After download, align all images:
    python -c "from efnet.utils import align_dataset; 
               align_dataset('/content/data/raw', '/content/data/aligned')"
"""
    print(msg)


# ---------------------------------------------------------------------------
# Face alignment pipeline
# ---------------------------------------------------------------------------

def align_dataset(src_dir: str, dst_dir: str,
                  det_size: Tuple[int, int] = (320, 320),
                  target_size: int = 112,
                  min_det_score: float = 0.85,
                  skip_existing: bool = True) -> int:
    """
    Align all face images in src_dir and write 112×112 crops to dst_dir.

    Preserves the identity folder structure:
        src_dir/Alice/img001.jpg  →  dst_dir/Alice/img001.jpg

    Requires: pip install insightface onnxruntime-gpu (or onnxruntime)

    Args:
        src_dir        : Root of unaligned dataset.
        dst_dir        : Where to write aligned 112×112 crops.
        det_size       : SCRFD input resolution. Lower = faster, less accurate.
        target_size    : Output face crop size. Always 112 for EFNet.
        min_det_score  : Minimum face detection confidence to accept.
        skip_existing  : Skip files already present in dst_dir.

    Returns:
        Number of successfully aligned images.
    """
    try:
        from insightface.app import FaceAnalysis
        from insightface.utils.face_align import norm_crop
        app = FaceAnalysis(name='buffalo_sc',
                           providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=det_size)
    except ImportError:
        print("InsightFace not found. Install with:")
        print("  pip install insightface onnxruntime")
        return 0

    src   = Path(src_dir)
    dst   = Path(dst_dir)
    count = 0
    skip  = 0
    fail  = 0

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_imgs = [p for p in src.rglob('*') if p.suffix.lower() in IMG_EXTS]

    print(f"[Align] Processing {len(all_imgs)} images "
          f"from '{src_dir}' → '{dst_dir}'")

    for src_path in all_imgs:
        rel_path = src_path.relative_to(src)
        dst_path = dst / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and dst_path.exists():
            skip += 1
            continue

        try:
            img_rgb = np.array(Image.open(src_path).convert('RGB'))
            faces   = app.get(img_rgb)

            if not faces:
                fail += 1
                continue

            best_face = max(
                [f for f in faces if f.det_score >= min_det_score],
                key=lambda f: f.det_score,
                default=None
            )
            if best_face is None:
                fail += 1
                continue

            aligned = norm_crop(img_rgb, best_face.kps, image_size=target_size)
            Image.fromarray(aligned).save(str(dst_path), quality=95)
            count += 1

        except Exception as e:
            fail += 1

        # Progress every 1000 images
        total_done = count + skip + fail
        if total_done % 1000 == 0:
            print(f"  {total_done}/{len(all_imgs)} | "
                  f"aligned={count} skipped={skip} failed={fail}")

    print(f"\n[Align] Done. aligned={count}, skipped={skip}, failed={fail}")
    return count


# ---------------------------------------------------------------------------
# Image quality filter
# ---------------------------------------------------------------------------

def laplacian_variance(img_pil: Image.Image) -> float:
    """
    Measure image sharpness using Laplacian variance.
    Low variance = blurry image. Threshold ≈ 100 works well in practice.
    """
    gray = np.array(img_pil.convert('L'), dtype=np.float32)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def filter_blurry_images(src_dir: str, dst_dir: str,
                         threshold: float = 100.0) -> int:
    """
    Copy non-blurry images from src_dir to dst_dir.
    Returns count of images kept.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    kept = 0
    IMG_EXTS = {'.jpg', '.jpeg', '.png'}

    for src_path in src.rglob('*'):
        if src_path.suffix.lower() not in IMG_EXTS:
            continue
        try:
            img = Image.open(src_path)
            if laplacian_variance(img) >= threshold:
                rel     = src_path.relative_to(src)
                dst_out = dst / rel
                dst_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_path), str(dst_out))
                kept += 1
        except Exception:
            continue

    print(f"[Filter] Kept {kept} sharp images from '{src_dir}'")
    return kept


# ---------------------------------------------------------------------------
# Custom dataset builder (live capture from webcam)
# ---------------------------------------------------------------------------

def capture_custom_dataset(output_dir: str,
                            identities: List[str],
                            shots_per_identity: int = 10,
                            capture_delay_s: float = 0.5):
    """
    Interactively capture face images from webcam for a custom dataset.

    For each identity in the list, press SPACE to capture, ESC to skip.
    Saves to output_dir/identity_name/IMG_XXXX.jpg

    Requires: OpenCV with camera support (works on local machine, not Colab).
    For Colab: use the JavaScript-based capture cell in the notebook instead.

    Args:
        output_dir           : Root directory to save images.
        identities           : List of person names to capture.
        shots_per_identity   : How many photos to take per person.
        capture_delay_s      : Delay between captures (avoid near-duplicate frames).
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam. Use the Colab notebook cell for browser capture.")
        return

    for name in identities:
        save_dir = Path(output_dir) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(save_dir.glob('*.jpg')))
        n_to_capture = max(0, shots_per_identity - existing)

        if n_to_capture == 0:
            print(f"[Capture] '{name}' already has {existing} images. Skipping.")
            continue

        print(f"\n[Capture] Ready to capture {n_to_capture} images for '{name}'")
        print("  SPACE = capture | ESC = skip identity | Q = quit")

        captured = 0
        while captured < n_to_capture:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            cv2.putText(display,
                        f"{name} — {captured}/{n_to_capture}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
            cv2.imshow('EFNet Capture', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                fname = save_dir / f"img_{existing + captured:04d}.jpg"
                cv2.imwrite(str(fname), frame)
                captured += 1
                print(f"  Captured: {fname}")
                time.sleep(capture_delay_s)
            elif key == 27:   # ESC
                print(f"  Skipped '{name}'")
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[Capture] Done. Images saved to '{output_dir}'")


# ---------------------------------------------------------------------------
# Colab JavaScript webcam capture cell
# ---------------------------------------------------------------------------

COLAB_CAPTURE_JS = '''
# Run this cell in Colab to capture face images from your browser webcam.
# It will save images to /content/custom_faces/<name>/

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import os, re
from PIL import Image
import io

def take_photo(filename, quality=0.92):
    js = Javascript("""
    async function takePhoto(quality) {
        const div = document.createElement('div');
        const video = document.createElement('video');
        const capture = document.createElement('button');
        capture.textContent = 'Capture';
        capture.style.cssText = 'padding:8px 16px;margin:8px;font-size:14px';
        div.appendChild(video);
        div.appendChild(capture);
        document.body.appendChild(div);
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        video.srcObject = stream;
        await video.play();
        await new Promise((resolve) => capture.onclick = resolve);
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        stream.getTracks().forEach(t => t.stop());
        div.remove();
        return canvas.toDataURL('image/jpeg', quality);
    }
    """)
    display(js)
    data = eval_js(f'takePhoto({quality})')
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# Usage:
identities = ['Alice', 'Bob', 'Charlie']   # <-- edit these
shots = 5
for name in identities:
    os.makedirs(f'/content/custom_faces/{name}', exist_ok=True)
    for i in range(shots):
        print(f"Capturing {name} — image {i+1}/{shots}")
        path = f'/content/custom_faces/{name}/img_{i+1:04d}.jpg'
        take_photo(path)
        print(f"  Saved: {path}")
print("All captures done!")
'''


def print_colab_capture_cell():
    """Print the Colab webcam capture code for copy-paste."""
    print("Copy the following code into a Colab cell:")
    print("─" * 60)
    print(COLAB_CAPTURE_JS)
    print("─" * 60)
