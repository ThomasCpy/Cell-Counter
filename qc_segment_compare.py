"""
Standalone QC: compare Hoechst nucleus-seed detection methods side by side.

This does NOT touch the training pipeline. It only renders comparison figures so
you can pick a reliable seeding method before regenerating training labels for a
new cell type.

Methods compared per tile:
  1. Baseline        - dataset.segment_nuclei (triangle + raw-EDT watershed)
  2. Improved        - smoothed EDT + intensity-weighted surface + h_maxima seeds
  3. StarDist        - pretrained '2D_versatile_fluo' (only if `stardist` is installed)

Run:
    CellCounter/Scripts/python.exe qc_segment_compare.py
Edit DATA_DIR / NUM_SAMPLES below, or pass --data and --num on the CLI.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, segmentation, exposure
from skimage.measure import regionprops
import scipy.ndimage as ndi

from dataset import CellDataset, load_as_float, segment_nuclei

# ----------------------------- tunables ------------------------------------
DATA_DIR = r"C:\Users\lab\OneDrive - Vandria\Bureau\Cell-Counter\IPSC data"
NUM_SAMPLES = 12
OUT_DIR = "QC_compare"

MAX_SIZE = 10          # remove_small_objects max_size (matches baseline)
SMOOTH_SIGMA = 2.0     # gaussian sigma applied to the EDT (kills internal bumps)
INTENSITY_FRAC = 0.5   # 0=pure distance, 1=pure Hoechst intensity for the surface
H_MAXIMA = 0.06        # seed depth on the normalized [0,1] surface (raise -> fewer seeds)
# ---------------------------------------------------------------------------


def _hoechst_mask(img, max_size=MAX_SIZE):
    """Triangle threshold + background-artifact refinement (baseline masking, NO erosion)."""
    img_smoothed = filters.gaussian(img, sigma=2.0)
    thresh = filters.threshold_triangle(img_smoothed)
    mask = img_smoothed > thresh

    labeled_mask, num_labels = ndi.label(mask)
    if num_labels > 0:
        sizes = np.bincount(labeled_mask.ravel())
        sizes[0] = 0
        for label in np.where(sizes > 0.05 * img.size)[0]:
            region = labeled_mask == label
            px = img_smoothed[region]
            if px.size > 0 and px.max() > px.min():
                refined = region & (img_smoothed > filters.threshold_triangle(px))
                mask = (mask & ~region) | refined
            else:
                mask = mask & ~region

    mask_cleaned = morphology.remove_small_objects(mask, max_size=max_size)
    return img_smoothed, mask_cleaned


def segment_improved(img, max_size=MAX_SIZE, smooth_sigma=SMOOTH_SIGMA,
                     intensity_frac=INTENSITY_FRAC, h=H_MAXIMA):
    """Smoothed-EDT + intensity-weighted surface + h_maxima seeds. No pre-erosion."""
    if img is None or np.max(img) == 0:
        return np.zeros_like(img, dtype=int), 0

    img_smoothed, mask = _hoechst_mask(img, max_size)
    if not np.any(mask):
        return np.zeros_like(img, dtype=int), 0

    distance = ndi.distance_transform_edt(mask)
    distance = filters.gaussian(distance, sigma=smooth_sigma)

    # Build a watershed surface that blends shape (distance) and brightness (Hoechst).
    d = distance / (distance.max() + 1e-8)
    it = img_smoothed / (img_smoothed.max() + 1e-8)
    surface = ((1.0 - intensity_frac) * d + intensity_frac * it) * mask

    # h_maxima suppresses shallow maxima regardless of spacing -> avoids over-splitting
    # single nuclei while still seeding each member of a touching cluster.
    seeds = morphology.h_maxima(surface, h) * mask
    markers, _ = ndi.label(seeds)
    if markers.max() == 0:
        return np.zeros_like(img, dtype=int), 0

    labels = segmentation.watershed(-surface, markers, mask=mask)
    return labels, int(labels.max())


# --------------------------- optional StarDist -----------------------------
_STARDIST = {"tried": False, "model": None}


def segment_stardist(img):
    """Returns (labels, count) using StarDist, or (None, None) if unavailable."""
    if not _STARDIST["tried"]:
        _STARDIST["tried"] = True
        try:
            from stardist.models import StarDist2D
            _STARDIST["model"] = StarDist2D.from_pretrained("2D_versatile_fluo")
        except Exception as e:  # not installed / no weights / TF missing
            print(f"[StarDist] unavailable ({type(e).__name__}: {e}). "
                  f"Install with: pip install stardist tensorflow  -> panel will appear.")
            _STARDIST["model"] = None

    model = _STARDIST["model"]
    if model is None:
        return None, None

    from csbdeep.utils import normalize
    labels, _ = model.predict_instances(normalize(img, 1, 99.8))
    return labels, int(labels.max())


# ------------------------------- rendering ---------------------------------
def _draw(ax, vis, labels, title):
    ax.imshow(vis, cmap="gray")
    if labels is not None and labels.max() > 0:
        mapping = np.arange(labels.max() + 1)
        np.random.shuffle(mapping[1:])
        overlay = np.ma.masked_where(labels == 0, mapping[labels])
        ax.imshow(overlay, cmap="hsv", alpha=0.4)
        cy = [p.centroid[0] for p in regionprops(labels)]
        cx = [p.centroid[1] for p in regionprops(labels)]
        ax.scatter(cx, cy, s=6, c="white", edgecolors="black", linewidths=0.3)
    ax.set_title(title)
    ax.axis("off")


def run(data_dir, num_samples, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dataset = CellDataset(data_dir=data_dir, is_train=False)
    if not dataset.pairs:
        print(f"No image pairs found in '{data_dir}'.")
        return

    samples = dataset.pairs[:num_samples] if num_samples else dataset.pairs
    for i, pair in enumerate(samples):
        path = pair["hoechst"]
        name = os.path.splitext(os.path.basename(path))[0]
        img = load_as_float(path)
        if img is None:
            continue
        print(f"[{i+1}/{len(samples)}] {name}")

        base_labels, base_n = segment_nuclei(img)
        imp_labels, imp_n = segment_improved(img)
        sd_labels, sd_n = segment_stardist(img)

        vis = exposure.equalize_adapthist(img, clip_limit=0.03)

        panels = [
            ("Hoechst (CLAHE)", None, None),
            (f"Baseline  (n={base_n})", base_labels, base_n),
            (f"Improved  (n={imp_n})", imp_labels, imp_n),
        ]
        if sd_labels is not None:
            panels.append((f"StarDist  (n={sd_n})", sd_labels, sd_n))

        fig, ax = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
        ax[0].imshow(vis, cmap="gray")
        ax[0].set_title(panels[0][0])
        ax[0].axis("off")
        for a, (title, labels, _) in zip(ax[1:], panels[1:]):
            _draw(a, vis, labels, title)

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"CMP_{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved comparison figures to '{out_dir}'.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA_DIR)
    p.add_argument("--num", type=int, default=NUM_SAMPLES)
    p.add_argument("--out", default=OUT_DIR)
    a = p.parse_args()
    run(a.data, a.num, a.out)
