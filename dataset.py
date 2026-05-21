import os
import glob
import re
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from skimage import io, filters, morphology, feature, segmentation, exposure
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def load_as_float(image_path):
    """Load image, scale to float, and dynamically handle hardware bit-depth."""
    if not os.path.exists(image_path):
        return None
        
    img = io.imread(image_path, as_gray=True)
    if np.max(img) == 0:
        return img.astype(np.float32)
        
    img_max = np.max(img)
    
    if img.dtype == np.uint8 or img_max <= 255:
        scale_factor = 255.0
    elif img_max <= 4095:
        scale_factor = 4095.0
    elif img_max <= 16383:
        scale_factor = 16383.0
    else:
        scale_factor = 65535.0
        
    return img.astype(np.float32) / scale_factor

def segment_nuclei(img, max_size=50, use_watershed=True, method='triangle'):
    """Segment Hoechst channel to identify nuclei markers."""
    if img is None or np.max(img) == 0: 
        return None, 0
        
    img_smoothed = filters.gaussian(img, sigma=2.0)
    
    if method == 'triangle':
        thresh = filters.threshold_triangle(img_smoothed)
    elif method == 'li':
        thresh = filters.threshold_li(img_smoothed + 1e-8)
    else:
        thresh = filters.threshold_otsu(img_smoothed)
        
    mask = img_smoothed > thresh

    mask_cleaned = morphology.remove_small_objects(mask, max_size=max_size)
    
    if use_watershed:
        distance = ndi.distance_transform_edt(mask_cleaned)
        coords = feature.peak_local_max(distance, min_distance=15, labels=mask_cleaned)
        marker_mask = np.zeros(distance.shape, dtype=bool)
        marker_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(marker_mask)
        nuclei_labels = segmentation.watershed(-distance, markers, mask=mask_cleaned)
        num_cells = np.max(nuclei_labels)
    else:
        nuclei_labels, num_cells = ndi.label(mask_cleaned)

    return nuclei_labels, num_cells

class CellDataset(Dataset):
    def __init__(self, data_dir, is_train=True, crop_size=(256, 256)):
        self.data_dir = data_dir
        self.is_train = is_train
        self.crop_size = crop_size
        
        search_pattern = os.path.join(data_dir, "*.[tT][iI][fF]*")
        all_files = glob.glob(search_pattern)
        
        # Parse filenames
        pattern = re.compile(r'^(.*_([A-H]\d{2})f(\d{2})d)(\d)(.*)$')
        
        pairs_dict = {}
        for f in all_files:
            match = pattern.search(f)
            if match:
                base_prefix = match.group(1)
                channel = match.group(4)
                suffix = match.group(5)
                
                key = base_prefix + suffix
                if key not in pairs_dict:
                    pairs_dict[key] = {}
                pairs_dict[key][channel] = f
                
        self.pairs = []
        for key, channels in pairs_dict.items():
            if '0' in channels and '1' in channels:
                self.pairs.append({
                    'hoechst': channels['0'],
                    'bf': channels['1']
                })
                
        # Sort pairs deterministically
        self.pairs = sorted(self.pairs, key=lambda x: x['bf'])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        bf_path = self.pairs[idx]['bf']
        hoechst_path = self.pairs[idx]['hoechst']

        # Load images
        bf_img = load_as_float(bf_path)
        hoechst_img = load_as_float(hoechst_path)

        # Generate target mask
        nuclei_labels, _ = segment_nuclei(hoechst_img)
        target_mask = (nuclei_labels > 0).astype(np.float32)

        # Convert to tensors
        bf_tensor = torch.from_numpy(bf_img).unsqueeze(0)
        hoechst_tensor = torch.from_numpy(target_mask).unsqueeze(0)

        # Pad and crop
        _, h, w = bf_tensor.shape

        # Pad if needed
        pad_h = max(0, self.crop_size[0] - h)
        pad_w = max(0, self.crop_size[1] - w)

        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
        if pad_h > 0 or pad_w > 0:
            bf_tensor = TF.pad(bf_tensor, padding, fill=0)
            hoechst_tensor = TF.pad(hoechst_tensor, padding, fill=0)

        _, h, w = bf_tensor.shape

        # Crop
        if self.is_train:
            top = random.randint(0, h - self.crop_size[0])
            left = random.randint(0, w - self.crop_size[1])
        else:
            top = (h - self.crop_size[0]) // 2
            left = (w - self.crop_size[1]) // 2

        bf_tensor = TF.crop(bf_tensor, top, left, self.crop_size[0], self.crop_size[1])
        hoechst_tensor = TF.crop(hoechst_tensor, top, left, self.crop_size[0], self.crop_size[1])

        # Data augmentation
        if self.is_train:
            # Horizontal flip
            if random.random() > 0.5:
                bf_tensor = TF.hflip(bf_tensor)
                hoechst_tensor = TF.hflip(hoechst_tensor)

            # Vertical flip
            if random.random() > 0.5:
                bf_tensor = TF.vflip(bf_tensor)
                hoechst_tensor = TF.vflip(hoechst_tensor)

            # Rotation
            if random.random() > 0.5:
                angle = random.uniform(-180, 180)
                bf_tensor = TF.rotate(bf_tensor, angle)
                hoechst_tensor = TF.rotate(hoechst_tensor, angle)

            # Brightness adjustment
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.7, 1.3) # Darken or lighten by up to 30%
                bf_tensor = TF.adjust_brightness(bf_tensor, brightness_factor)

        return bf_tensor, hoechst_tensor

def run_qc_segmentation(data_dir="data", qc_dir="QC", num_samples=10):
    """Run QC on nuclear segmentation and save side-by-side comparisons."""
    os.makedirs(qc_dir, exist_ok=True)
    
    # Use the same logic as CellDataset to find pairs
    dataset = CellDataset(data_dir=data_dir, is_train=False)
    
    if len(dataset.pairs) == 0:
        print(f"No image pairs found in '{data_dir}'.")
        return
        
    samples = dataset.pairs[:num_samples] if num_samples else dataset.pairs
    
    for i, pair in enumerate(samples):
        hoechst_path = pair['hoechst']
        base_name = os.path.basename(hoechst_path)
        name_no_ext, _ = os.path.splitext(base_name)
        print(f"Processing QC for {base_name}...")
        
        hoechst_img = load_as_float(hoechst_path)
        if hoechst_img is None:
            continue
            
        nuclei_labels, num_cells = segment_nuclei(hoechst_img)
        
        # Apply CLAHE for visualization to enhance contrast without altering raw data
        hoechst_vis = exposure.equalize_adapthist(hoechst_img, clip_limit=0.03)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        ax[0].imshow(hoechst_vis, cmap='gray')
        ax[0].set_title("Hoechst (CLAHE)")
        ax[0].axis('off')
        
        ax[1].imshow(hoechst_vis, cmap='gray')
        mask = nuclei_labels > 0
        ax[1].imshow(mask, cmap='Reds', alpha=0.3)
        ax[1].set_title(f"Segmentation Overlay (Count: {num_cells})")
        ax[1].axis('off')
        
        plt.tight_layout()
        out_path = os.path.join(qc_dir, f"QC_{name_no_ext}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    print(f"Saved {len(samples)} QC images to {qc_dir}")

if __name__ == "__main__":
    run_qc_segmentation(data_dir=r"C:\Users\lab\OneDrive - Vandria\Bureau\Cell counter training dataset\Plate 2 HMC3")