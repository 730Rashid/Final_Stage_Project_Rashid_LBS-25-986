"""
Multi Model Vectorisation Pipeline for Ablation Study.

Generates embeddings using SigLIP base and ResNet50 to complement the
existing CLIP ViT B/32 embeddings. Each model's output is saved to
data/comparison/ as a separate .npy file.

Models:
  SigLIP base  (768 dim, sigmoid loss, trained on WebLI 10B pairs)
  ResNet50     (2048 dim, ImageNet baseline, no text understanding)

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import sys
import gc
import shutil
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import config

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data"
OUTPUT_DIR = config.COMPARISON_DIR


def find_images(root_dir: Path) -> List[str]:
    """Find all image files in the directory, matching vectorise.py logic."""
    image_paths = []

    for ext in config.IMAGE_EXTENSIONS:
        for path in root_dir.rglob("*{}".format(ext)):
            image_paths.append(str(path))

    image_paths.sort()

    return image_paths


def load_siglip_model(device: str) -> Tuple[Any, Any]:
    """
    Load SigLIP base model and image processor from HuggingFace.

    Uses SiglipImageProcessor directly instead of AutoProcessor
    to avoid the tokeniser resolution bug in transformers when
    no text encoder is needed.
    """
    from transformers import AutoModel, SiglipImageProcessor

    hf_id = config.MODEL_REGISTRY["siglip"]["hf_id"]
    print("  Loading {} ...".format(hf_id))

    model = AutoModel.from_pretrained(hf_id).to(device)
    processor = SiglipImageProcessor.from_pretrained(hf_id)
    model.eval()

    print("SigLIP loaded successfully")

    return model, processor


def load_resnet50_model(device: str) -> Tuple[Any, Any]:
    """Load ResNet50 with the classification head removed (2048 dim output)."""
    import torchvision.models as models
    import torchvision.transforms as transforms

    print("Loading ResNet50 ImageNet pretrained ...")

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    full_model = models.resnet50(weights=weights)

    # Remove the final FC layer to get raw 2048 dim features
    feature_extractor = torch.nn.Sequential(*list(full_model.children())[:-1])
    feature_extractor = feature_extractor.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    print("ResNet50 loaded successfully")

    return feature_extractor, preprocess


def process_images_siglip(model, processor, device, image_paths):
    """Generate 768 dim SigLIP embeddings for all images."""
    # Batch size 4 prevents segfault from PIL memory accumulation on CPU
    batch_size = 4
    print("  Batch size: {} (memory safe)".format(batch_size))

    embeddings = []
    valid_paths = []
    corrupt_files = []
    batch_num = 0

    for i in tqdm(range(0, len(image_paths), batch_size), desc="  SigLIP"):
        batch_paths = image_paths[i:i + batch_size]

        batch_images = []
        batch_valid  = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")

                # Copy pixels and close the file handle immediately
                # to prevent OS file descriptor exhaustion across 17k images
                img_copy = img.copy()
                img.close()

                batch_images.append(img_copy)
                batch_valid.append(path)

            except Exception as e:
                corrupt_files.append((path, str(e)))

        if not batch_images:
            continue

        try:
            inputs = processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            with torch.no_grad():
                # vision_model.pooler_output returns a raw tensor (768 dim)
                outputs = model.vision_model(pixel_values=pixel_values).pooler_output

            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(outputs.cpu().numpy())
            valid_paths.extend(batch_valid)

        except Exception as e:
            print("  Batch error: {}".format(e))

            for path in batch_valid:
                corrupt_files.append((path, str(e)))

        finally:
            for img in batch_images:
                img.close()

            batch_images.clear()
            batch_num += 1

            if batch_num % 50 == 0:
                gc.collect()

    embeddings = np.vstack(embeddings) if embeddings else np.array([])

    print("  Generated {} embeddings (dim={})".format(
        len(embeddings), embeddings.shape[1] if len(embeddings) else 0))

    return embeddings, valid_paths, corrupt_files


def process_images_resnet50(model, preprocess, device, image_paths):
    """Generate 2048 dim ResNet50 embeddings for all images."""
    batch_size = config.MODEL_REGISTRY["resnet50"]["batch_size"]

    print("  Batch size: {}".format(batch_size))

    embeddings = []
    valid_paths = []
    corrupt_files = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="  ResNet50"):
        batch_paths = image_paths[i:i + batch_size]

        batch_tensors = []
        batch_valid   = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_tensors.append(preprocess(img))
                batch_valid.append(path)

            except Exception as e:
                corrupt_files.append((path, str(e)))

        if not batch_tensors:
            continue

        try:
            batch = torch.stack(batch_tensors).to(device)

            with torch.no_grad():
                outputs = model(batch).squeeze(-1).squeeze(-1)

            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(outputs.cpu().numpy())
            valid_paths.extend(batch_valid)

        except Exception as e:
            print("  Batch error: {}".format(e))

            for path in batch_valid:
                corrupt_files.append((path, str(e)))

    embeddings = np.vstack(embeddings) if embeddings else np.array([])

    print("  Generated {} embeddings (dim={})".format(
        len(embeddings), embeddings.shape[1] if len(embeddings) else 0))

    return embeddings, valid_paths, corrupt_files


def save_model_embeddings(model_key, embeddings, paths):
    """Save model embeddings and filenames to the comparison directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "{}_embeddings.npy".format(model_key), embeddings)

    with open(OUTPUT_DIR / "{}_filenames.json".format(model_key), "w") as f:
        json.dump(paths, f, indent=2)

    print("Saved {}_embeddings.npy {}".format(model_key, embeddings.shape))


def copy_clip_embeddings():
    """Copy existing CLIP embeddings into the comparison directory."""
    src_emb = PROJECT_ROOT / "data" / "embeddings" / "embeddings.npy"
    src_fn  = PROJECT_ROOT / "data" / "embeddings" / "filenames.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if src_emb.exists():
        shutil.copy2(src_emb, OUTPUT_DIR / "clip_embeddings.npy")
        print("  Copied CLIP embeddings to comparison directory")

    if src_fn.exists():
        shutil.copy2(src_fn, OUTPUT_DIR / "clip_filenames.json")
        print("  Copied CLIP filenames to comparison directory")


def main():
    """Main entry point. Vectorise with SigLIP and ResNet50."""
    print("Multi Model Vectorisation Pipeline")
    print("Timestamp: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    if not DATASET_PATH.exists():
        print("Error: Dataset not found at {}".format(DATASET_PATH))
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print("Device: {}".format(device))

    image_paths = find_images(DATASET_PATH)
    print("Found {} images".format(len(image_paths)))
    print()

    # Step 1: Copy existing CLIP embeddings
    print("1/3 Copying existing CLIP embeddings...")
    copy_clip_embeddings()
    print()

    # Step 2: Generate SigLIP embeddings
    print("2/3 Generating SigLIP embeddings...")
    siglip_model, siglip_proc = load_siglip_model(device)

    siglip_embs, siglip_paths, siglip_corrupt = process_images_siglip(
        siglip_model, siglip_proc, device, image_paths)

    save_model_embeddings("siglip", siglip_embs, siglip_paths)

    del siglip_model, siglip_proc
    torch.cuda.empty_cache()
    print()

    # Step 3: Generate ResNet50 embeddings
    print("3/3 Generating ResNet50 embeddings...")
    resnet_model, resnet_preprocess = load_resnet50_model(device)

    resnet_embs, resnet_paths, resnet_corrupt = process_images_resnet50(
        resnet_model, resnet_preprocess, device, image_paths)

    save_model_embeddings("resnet50", resnet_embs, resnet_paths)

    del resnet_model
    torch.cuda.empty_cache()
    print()

    # Summary
    print("SigLIP: {} embeddings".format(len(siglip_embs)))
    print("ResNet50: {} embeddings".format(len(resnet_embs)))

    if siglip_corrupt:
        print("SigLIP corrupt: {}".format(len(siglip_corrupt)))

    if resnet_corrupt:
        print("ResNet50 corrupt: {}".format(len(resnet_corrupt)))


if __name__ == "__main__":
    main()
