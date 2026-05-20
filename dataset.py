import os
import glob
import re
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from skimage import io, filters, morphology, feature, segmentation
import scipy.ndimage as ndi

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

def segment_nuclei(img, min_nucleus_area=500, use_watershed=True, method='triangle'):
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
    mask_cleaned = morphology.remove_small_objects(mask, min_size=min_nucleus_area)
    
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