import os
import glob
import re
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
from torchvision import tv_tensors
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
        
    if img.dtype == np.uint8:
        scale_factor = 255.0
    elif img.dtype == np.uint16:
        scale_factor = 65535.0
    else:
        # Fallback to dynamic scaling if dtype is non-standard
        img_max = np.max(img)
        if img_max <= 255:
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

    # Identify if the mask caught large continuous areas (like the well background)
    labeled_mask, num_labels = ndi.label(mask)
    if num_labels > 0:
        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0 # Ignore background
        
        # Consider any region covering more than 5% of the image as a well/background artifact
        large_labels = np.where(component_sizes > 0.05 * img.size)[0]
        
        for label in large_labels:
            large_region = (labeled_mask == label)
            region_pixels = img_smoothed[large_region]
            
            if len(region_pixels) > 0 and region_pixels.max() > region_pixels.min():
                if method == 'triangle':
                    new_thresh = filters.threshold_triangle(region_pixels)
                elif method == 'li':
                    new_thresh = filters.threshold_li(region_pixels + 1e-8)
                else:
                    new_thresh = filters.threshold_otsu(region_pixels)
                
                refined_region = large_region & (img_smoothed > new_thresh)
                mask = (mask & ~large_region) | refined_region
            else:
                mask = mask & ~large_region

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
    def __init__(self, data_dir, is_train=True, crop_size=(256, 256), preload=False):
        self.data_dir = data_dir
        self.is_train = is_train
        self.crop_size = crop_size
        self.preload = preload
        
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
        
        # Setup v2 transforms for training
        if self.is_train:
            self.spatial_transforms = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=180.0),
                v2.ElasticTransform(alpha=50.0, sigma=5.0)
            ])

        self.cache = {}
        if self.preload:
            print(f"Preloading and processing {len(self.pairs)} image pairs into memory...")
            for idx in range(len(self.pairs)):
                self.cache[idx] = self._process_image_pair(idx)

    def _process_image_pair(self, idx):
        bf_path = self.pairs[idx]['bf']
        hoechst_path = self.pairs[idx]['hoechst']

        # Load images
        bf_img = load_as_float(bf_path)
        hoechst_img = load_as_float(hoechst_path)

        # Pseudo-flatfield correction (Background Subtraction)
        bg = filters.gaussian(bf_img, sigma=50.0)
        bf_img = bf_img - bg
        
        # Safe Per-Image Standardization (Z-score)
        std = np.std(bf_img)
        bf_img = (bf_img - np.mean(bf_img)) / max(std, 1e-2)

        # Generate target mask
        nuclei_labels, _ = segment_nuclei(hoechst_img)
        target_mask = (nuclei_labels > 0).astype(np.float32)
        
        return bf_img, target_mask

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.preload and idx in self.cache:
            bf_img, target_mask = self.cache[idx]
        else:
            bf_img, target_mask = self._process_image_pair(idx)

        # Convert to tensors
        bf_tensor = tv_tensors.Image(torch.from_numpy(bf_img).unsqueeze(0))
        hoechst_tensor = tv_tensors.Mask(torch.from_numpy(target_mask).unsqueeze(0))

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
            # Apply all spatial transforms elegantly at once
            bf_tensor, hoechst_tensor = self.spatial_transforms(bf_tensor, hoechst_tensor)

            # Random Gaussian Noise (simulates camera read noise)
            if random.random() > 0.5:
                noise = torch.randn_like(bf_tensor) * 0.01
                bf_tensor = bf_tensor + noise

        return bf_tensor, hoechst_tensor

def run_debug_samples(data_dir="data", debug_dir="debug_samples", num_samples=10):
    """Run dataset pipeline to visualize augmented inputs directly."""
    os.makedirs(debug_dir, exist_ok=True)
    
    # Initialize dataset in training mode to see augmentations
    dataset = CellDataset(data_dir=data_dir, is_train=True)
    
    if len(dataset) == 0:
        print(f"No image pairs found in '{data_dir}'.")
        return
        
    num_to_save = min(num_samples, len(dataset))
    for i in range(num_to_save):
        bf_tensor, hoechst_tensor = dataset[i]
        
        # Convert back to numpy for visualization
        bf_img = bf_tensor.squeeze(0).numpy()
        mask = hoechst_tensor.squeeze(0).numpy()
        
        # Min-Max scale brightfield back to [0, 1] for saving
        bf_min, bf_max = bf_img.min(), bf_img.max()
        if bf_max > bf_min:
            bf_vis = (bf_img - bf_min) / (bf_max - bf_min)
        else:
            bf_vis = bf_img
            
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(bf_vis, cmap='gray')
        ax[0].set_title(f"Model Input (Brightfield)\nMin: {bf_min:.2f}, Max: {bf_max:.2f}")
        ax[0].axis('off')
        
        ax[1].imshow(mask, cmap='Blues', vmin=0, vmax=1)
        ax[1].set_title("Target Mask")
        ax[1].axis('off')
        
        ax[2].imshow(bf_vis, cmap='gray')
        ax[2].imshow(mask, cmap='Reds', alpha=0.3, vmin=0, vmax=1)
        ax[2].set_title("Overlay")
        ax[2].axis('off')
        
        plt.tight_layout()
        out_path = os.path.join(debug_dir, f"debug_sample_{i}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    print(f"Saved {num_to_save} debug samples to '{debug_dir}'")

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
        ax[1].imshow(mask, cmap='Blues', alpha=0.3)
        ax[1].set_title(f"Segmentation Overlay (Count: {num_cells})")
        ax[1].axis('off')
        
        plt.tight_layout()
        out_path = os.path.join(qc_dir, f"QC_{name_no_ext}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    print(f"Saved {len(samples)} QC images to {qc_dir}")

if __name__ == "__main__":
    run_qc_segmentation(data_dir=r"C:\Users\lab\OneDrive - Vandria\Bureau\Cell counter training dataset\Plate 2 HMC3", num_samples=1000)
    
    # Uncomment the line below to test the training dataset augmentations directly
    # run_debug_samples(data_dir=r"C:\Users\lab\OneDrive - Vandria\Bureau\Cell counter training dataset\Plate 2 HMC3", num_samples=10)