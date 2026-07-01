"""
Precompute nucleus labels from the Hoechst channel with Cellpose, for training the
brightfield U-Net.

Run this ONCE on a GPU machine. It segments every Hoechst image, extracts nucleus
centroids, and caches them to disk. The training dataset (dataset.CellDataset with
label_cache=...) then loads these centroids and builds the Gaussian-blob targets,
so Cellpose never has to run during training and need not be installed on the
training machine.

The cache key is the image path *relative to data_dir*, so a cache built here is
portable to another machine as long as the same folder layout sits under its own
data_dir.

Usage (GPU box):
    python generate_labels.py --data "<path to data>" --cache label_cache
    python generate_labels.py --data "<path>" --cache label_cache --qc 20
GPU is used automatically when available; pass --cpu to force CPU.

Then copy the `label_cache/` folder next to the training code and run train.py.
"""
import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from skimage.measure import regionprops

from dataset import (
    CellDataset, load_as_float, save_cached_centroids, load_cached_centroids,
    label_cache_key, generate_gaussian_blobs,
)

# Heatmap sigma is owned by dataset.py; we only cache centroids here.


def build_cellpose(gpu, model_type):
    """Return (eval_fn(img) -> label_mask, info_str). Handles Cellpose v3 and v4."""
    import cellpose
    from cellpose import models
    major = int(cellpose.version.split(".")[0])

    # Don't silently fall back to CPU: a GPU labelling run that lands on CPU can
    # take hours. Verify CUDA is actually available before claiming the GPU.
    if gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                print("WARNING: --gpu requested but torch.cuda.is_available() is False. "
                      "This torch build can't see a GPU (need a CUDA build); running on CPU.")
                gpu = False
        except ImportError:
            pass

    if major >= 4:
        # Cellpose-SAM: a single generalist model, diameter-agnostic.
        model = models.CellposeModel(gpu=gpu)
        def run(img):
            return model.eval(img, normalize=True)[0]
        info = f"cellpose {cellpose.version} (CellposeModel, gpu={gpu})"
    else:
        # v3 and earlier: dedicated nuclei model, auto diameter estimation.
        model = models.Cellpose(gpu=gpu, model_type=model_type)
        def run(img):
            return model.eval(img, diameter=None, channels=[0, 0])[0]
        info = f"cellpose {cellpose.version} (model_type={model_type}, gpu={gpu})"
    return run, info


def save_qc_overlay(qc_dir, name, img, centroids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import exposure

    os.makedirs(qc_dir, exist_ok=True)
    # Normalize to [0,1] from the image's own min/max so CLAHE never sees
    # out-of-range floats, regardless of bit depth or input dtype.
    img_norm = exposure.rescale_intensity(img.astype(np.float32), in_range="image",
                                          out_range=(0.0, 1.0))
    vis = exposure.equalize_adapthist(img_norm, clip_limit=0.03)
    heat = generate_gaussian_blobs(img.shape, centroids, sigma=4.0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(vis, cmap="gray")
    ax[0].set_title("Hoechst (CLAHE)")
    ax[0].axis("off")
    ax[1].imshow(vis, cmap="gray")
    ax[1].imshow(heat, cmap="magma", alpha=0.5, vmin=0, vmax=1)
    if len(centroids):
        c = np.asarray(centroids).reshape(-1, 2)
        ax[1].scatter(c[:, 1], c[:, 0], s=6, c="cyan", edgecolors="black", linewidths=0.3)
    ax[1].set_title(f"Cellpose centroids (n={len(centroids)})")
    ax[1].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(qc_dir, f"LBL_{name}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=r"C:\Users\lab\OneDrive - Vandria\Bureau\Cell-Counter\IPSC data",
                   help="Root data dir (recursively searched)")
    p.add_argument("--cache", default="label_cache", help="Output dir for cached centroids")
    p.add_argument("--cpu", action="store_true", help="Force CPU (default is GPU when available)")
    p.add_argument("--model", default="nuclei", help="Cellpose v3 model_type (ignored on v4)")
    p.add_argument("--qc-frac", type=float, default=0.1,
                   help="Fraction of images to save QC overlays for (default 0.1 = 10 percent)")
    p.add_argument("--qc", type=int, default=None,
                   help="Absolute number of QC overlays (overrides --qc-frac; 0 disables)")
    p.add_argument("--seed", type=int, default=0, help="Seed for the random QC sample")
    p.add_argument("--overwrite", action="store_true", help="Re-label images already cached")
    args = p.parse_args()

    dataset = CellDataset(data_dir=args.data, is_train=False)
    if not dataset.pairs:
        print(f"No image pairs found in '{args.data}'.")
        return

    run, info = build_cellpose(gpu=not args.cpu, model_type=args.model)
    print(f"Labelling {len(dataset.pairs)} Hoechst images with {info}")
    os.makedirs(args.cache, exist_ok=True)

    # Randomly choose which images get a QC overlay (default 10% of the dataset).
    n_qc = args.qc if args.qc is not None else round(args.qc_frac * len(dataset.pairs))
    n_qc = max(0, min(n_qc, len(dataset.pairs)))
    qc_indices = set(random.Random(args.seed).sample(range(len(dataset.pairs)), n_qc))
    qc_dir = os.path.join(args.cache, "QC")
    print(f"Saving QC overlays for {n_qc} randomly sampled images.")

    manifest = {}
    skipped = 0
    for i, pair in enumerate(tqdm(dataset.pairs, desc="Cellpose labelling")):
        hoechst_path = pair["hoechst"]
        key = label_cache_key(args.data, hoechst_path)
        out_fp = os.path.join(args.cache, key + ".npy")
        name = os.path.splitext(os.path.basename(hoechst_path))[0]
        want_qc = i in qc_indices

        if os.path.exists(out_fp) and not args.overwrite:
            skipped += 1
            # Still render QC from the cached centroids if this image was sampled.
            if want_qc:
                img = load_as_float(hoechst_path)
                cached = load_cached_centroids(args.cache, args.data, hoechst_path)
                if img is not None and cached is not None:
                    save_qc_overlay(qc_dir, name, img, cached)
            continue

        img = load_as_float(hoechst_path)
        if img is None or np.max(img) == 0:
            centroids = np.zeros((0, 2), dtype=np.float32)
        else:
            masks = run(img)
            centroids = np.array([prop.centroid for prop in regionprops(masks)],
                                 dtype=np.float32).reshape(-1, 2)

        save_cached_centroids(args.cache, args.data, hoechst_path, centroids)
        manifest[key] = {
            "rel": os.path.relpath(hoechst_path, args.data).replace("\\", "/"),
            "count": int(len(centroids)),
        }

        if want_qc and img is not None:
            save_qc_overlay(qc_dir, name, img, centroids)

    # Merge manifest with any prior run so it stays complete across incremental runs.
    manifest_fp = os.path.join(args.cache, "manifest.json")
    if os.path.exists(manifest_fp):
        with open(manifest_fp) as f:
            prior = json.load(f)
        prior.update(manifest)
        manifest = prior
    with open(manifest_fp, "w") as f:
        json.dump(manifest, f, indent=2)

    counts = [m["count"] for m in manifest.values()]
    print(f"\nDone. Newly labelled: {len(dataset.pairs) - skipped}, skipped (cached): {skipped}.")
    if counts:
        print(f"Cache holds {len(counts)} images, "
              f"nuclei/image min/median/max = {min(counts)}/{int(np.median(counts))}/{max(counts)}.")
    print(f"Cache dir: '{args.cache}'  (copy this to the training machine)")


if __name__ == "__main__":
    main()
