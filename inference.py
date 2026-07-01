import os
import csv
import glob
import re
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
import random
import onnxruntime as ort
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")  # headless; required for rendering QC overlays off the main thread
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from dataset import load_as_float
from skimage.feature import peak_local_max
import concurrent.futures
from collections import deque

def pad_to_multiple(img, multiple=8):
    """Pad image dimensions to a specified multiple."""
    h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
        
    return img, h, w

def parse_tecan_d300e_xml(file_path):
    """
    parses a Tecan D300e .dtt XML file.
    tracks where the software physically dispensed the fluids and vehicles after randomization.
    returns a dictionary mapping physical well coordinates to treatment strings.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    protocol = root.find('Protocol')

    unit_node = protocol.find('MolarityConcentrationUnit')
    unit = unit_node.text if unit_node is not None else "uM"

    fluid_map = {}
    for fluid in protocol.find('Fluids').findall('Fluid'):
        fluid_id = fluid.get('ID')
        name = fluid.find('Name').text
        fluid_map[fluid_id] = name

    plate = protocol.find('.//Plates/Plate')

    final_destination_map = {}
    randomize_node = plate.find('Randomize')
    if randomize_node is not None:
        for r_well in randomize_node.findall('Well'):
            logical_pos = (int(r_well.get('R')), int(r_well.get('C')))
            physical_pos = (int(r_well.get('ToR')), int(r_well.get('ToC')))
            final_destination_map[logical_pos] = physical_pos

    well_treatments = defaultdict(list)
    
    for well in plate.find('Wells').findall('Well'):
        recipe_r = int(well.get('Row'))
        recipe_c = int(well.get('Col'))
        
        phys_pos = final_destination_map.get((recipe_r, recipe_c), (recipe_r, recipe_c))
        
        for fluid in well.findall('Fluid'):
            f_id = fluid.get('ID')
            concentration = fluid.text
            drug_name = fluid_map.get(f_id, "Unknown")
            well_treatments[phys_pos].append(f"{drug_name} {concentration}{unit}")

    vehicle_wells = set()
    backfill_node = protocol.find('.//Backfills')
    if backfill_node is not None:
        for well in backfill_node.findall('.//Well'):
            recipe_r = int(well.get('R'))
            recipe_c = int(well.get('C'))
            
            phys_pos = final_destination_map.get((recipe_r, recipe_c), (recipe_r, recipe_c))
            vehicle_wells.add(phys_pos)

    treatment_output = {}
    all_used_wells = set(well_treatments.keys()).union(vehicle_wells)

    for phys_r, phys_c in all_used_wells:
        row_letter = chr(65 + phys_r) 
        col_num = phys_c + 1
        well_name = f"{row_letter}{col_num:02d}"
        
        if (phys_r, phys_c) in well_treatments and well_treatments[(phys_r, phys_c)]:
            treatment_output[well_name] = " + ".join(well_treatments[(phys_r, phys_c)])
        else:
            treatment_output[well_name] = "Vehicle"

    return treatment_output

_worker_session = None
_worker_input_name = None

def _get_worker_session(onnx_path):
    global _worker_session, _worker_input_name
    if _worker_session is None:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        _worker_session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
        _worker_input_name = _worker_session.get_inputs()[0].name
    return _worker_session, _worker_input_name

def process_single_file(file_path, predict, threshold, output_dir, save_qc, treatments):
    basename = os.path.basename(file_path)
    pattern = re.compile(r'(?:t(\d{3}))?([A-H]\d{2})f(\d{2})d(\d)', re.IGNORECASE)
    match = pattern.search(basename)
    if not match:
        return None
        
    time_str, well, field, channel = match.groups()
    if channel != '0':
        return None
    time_idx = int(time_str) if time_str else 0
    
    img = load_as_float(file_path)
    if img is None:
        return None
        
    bg = ndi.gaussian_filter(img, sigma=50.0)
    img = img - bg
    std = np.std(img)
    img = (img - np.mean(img)) / max(std, 1e-2)

    padded_img, orig_h, orig_w = pad_to_multiple(img, multiple=8)
    input_tensor = np.expand_dims(padded_img, axis=(0, 1)).astype(np.float32)

    logits = predict(input_tensor)
    logits = logits[:orig_h, :orig_w]

    probs = 1.0 / (1.0 + np.exp(-logits))
    coords = peak_local_max(probs, min_distance=5, threshold_abs=threshold)
    num_cells = len(coords)

    if save_qc:
        qc_dir = os.path.join(output_dir, "QC")
        os.makedirs(qc_dir, exist_ok=True)
        name, _ = os.path.splitext(basename)
        fig, ax = plt.subplots(1, 3, figsize=(16, 6))
        
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title("Original Brightfield")
        ax[0].axis('off')

        ax[1].imshow(probs, cmap='viridis')
        ax[1].set_title("Model Output (Probabilities)")
        ax[1].axis('off')

        ax[2].imshow(img, cmap='gray')
        if num_cells > 0:
            ax[2].plot(coords[:, 1], coords[:, 0], 'r.', markersize=6)
        ax[2].set_title(f"Detections (Count: {num_cells})")
        ax[2].axis('off')

        plt.tight_layout()
        out_img_path = os.path.join(qc_dir, f"{name}_overlay.png")
        plt.savefig(out_img_path, bbox_inches='tight')
        plt.close(fig)
    
    treatment = treatments.get(well, "Unknown") if treatments else "Unknown"
    
    print(f"Processed {basename} - Count: {num_cells}")
    return {
        'File': basename,
        'FilePath': file_path,
        'Well': well,
        'Field': field,
        'Time': time_idx,
        'Treatment': treatment,
        'Count': num_cells
    }

def process_file_worker(args):
    file_path, onnx_path, threshold, output_dir, save_qc, treatments = args
    session, input_name = _get_worker_session(onnx_path)
    predict = lambda t: session.run(None, {input_name: t})[0][0, 0]
    return process_single_file(file_path, predict, threshold, output_dir, save_qc, treatments)


def _pth_path_for(model_path):
    """Derive the sibling .pth weights path from a model argument."""
    if model_path.lower().endswith(".onnx"):
        return model_path[:-len(".onnx")] + ".pth"
    return model_path


class TorchBackend:
    """PyTorch UNet backend. Supports single-image predict() and a batched,
    fully-on-GPU process_batch() used by the high-throughput pipeline."""

    def __init__(self, pth_path):
        import torch
        import torch.nn.functional as F
        from model import UNet

        self.torch = torch
        self.F = F
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_gpu = self.device.type == "cuda"
        self.name = f"PyTorch ({self.device.type})"

        # Fields are all the same size, so let cuDNN pick the fastest conv algo.
        torch.backends.cudnn.benchmark = True

        model = UNet(n_channels=1, n_classes=1)
        model.load_state_dict(torch.load(pth_path, map_location=self.device))
        model.eval().to(self.device)
        self.model = model
        self._gauss_kernels = {}

    def predict(self, input_tensor):
        """Single (1, 1, H, W) float32 array -> 2D logits array (legacy path)."""
        with self.torch.no_grad():
            t = self.torch.from_numpy(input_tensor).to(self.device)
            out = self.model(t)
        return out[0, 0].detach().cpu().numpy()

    def gaussian_blur(self, t, sigma=50.0, truncate=4.0):
        """Separable Gaussian, numerically identical to
        scipy.ndimage.gaussian_filter(mode='reflect'). t is (N, 1, H, W)."""
        torch, F = self.torch, self.F
        radius = int(truncate * sigma + 0.5)
        k = self._gauss_kernels.get(sigma)
        if k is None:
            x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=self.device)
            k = torch.exp(-0.5 * (x / sigma) ** 2)
            k = k / k.sum()
            self._gauss_kernels[sigma] = k

        def sym_pad(x, pad, dim):  # scipy 'reflect' == edge-repeated symmetric
            left = x.narrow(dim, 0, pad).flip(dim)
            right = x.narrow(dim, x.size(dim) - pad, pad).flip(dim)
            return torch.cat([left, x, right], dim=dim)

        t = F.conv2d(sym_pad(t, radius, 3), k.view(1, 1, 1, -1))
        t = F.conv2d(sym_pad(t, radius, 2), k.view(1, 1, -1, 1))
        return t

    def process_batch(self, raw_imgs):
        """raw_imgs: list of equal-shape (H, W) float32 arrays.
        Returns (probs, zscored) numpy arrays of shape (N, H, W), matching the
        per-image preprocessing of the original CPU path."""
        torch, F = self.torch, self.F
        with torch.no_grad():
            t = torch.from_numpy(np.stack(raw_imgs)[:, None]).to(self.device)  # (N,1,H,W)

            # Pseudo-flatfield background subtraction
            t = t - self.gaussian_blur(t, sigma=50.0)

            # Safe per-image z-score
            flat = t.flatten(1)
            mean = flat.mean(dim=1).view(-1, 1, 1, 1)
            std = flat.std(dim=1, unbiased=False).view(-1, 1, 1, 1).clamp_min(1e-2)
            t = (t - mean) / std
            zscored = t[:, 0].detach().to("cpu").numpy()

            # Pad to multiple of 8, run, crop back, sigmoid
            H, W = t.shape[2], t.shape[3]
            pad_h, pad_w = (8 - H % 8) % 8, (8 - W % 8) % 8
            if pad_h or pad_w:
                t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
            probs = torch.sigmoid(self.model(t))[:, 0, :H, :W]
            probs = probs.detach().to("cpu").numpy()

        return probs, zscored


def create_predictor(model_path, backend="auto"):
    """
    Build an inference backend.

    Returns (predict, backend_name, is_gpu, torch_backend) where
    predict(input_tensor) takes a (1, 1, H, W) float32 array and returns a 2D
    logits array, and torch_backend is a TorchBackend (for the batched GPU
    pipeline) or None when running on onnxruntime.

    backend:
      - "torch": load the .pth UNet with PyTorch (uses CUDA if available).
      - "onnx":  load the .onnx model with onnxruntime (CUDA > DirectML > CPU).
      - "auto":  use Torch+CUDA when a .pth and a CUDA GPU are both available,
                 otherwise fall back to onnxruntime.
    """
    pth_path = _pth_path_for(model_path)

    use_torch = backend == "torch"
    if backend == "auto":
        use_torch = os.path.exists(pth_path) and _torch_cuda_available()

    if use_torch:
        if not os.path.exists(pth_path):
            raise FileNotFoundError(
                f"Torch backend requested but weights '{pth_path}' were not found."
            )
        tb = TorchBackend(pth_path)
        return tb.predict, tb.name, tb.is_gpu, tb

    # onnxruntime backend
    providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    active_provider = session.get_providers()[0]
    input_name = session.get_inputs()[0].name

    def predict(input_tensor):
        return session.run(None, {input_name: input_tensor})[0][0, 0]

    is_gpu = active_provider in ('CUDAExecutionProvider', 'DmlExecutionProvider')
    return predict, f"ONNX ({active_provider})", is_gpu, None


def _torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _save_qc_overlay(qc_dir, basename, img, probs, coords, num_cells):
    """Render the 3-panel QC overlay. Uses the Figure API (no pyplot global
    state) so it is safe to call from worker threads."""
    name, _ = os.path.splitext(basename)
    fig = Figure(figsize=(16, 6))
    ax = fig.subplots(1, 3)

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original Brightfield")
    ax[0].axis('off')

    ax[1].imshow(probs, cmap='viridis')
    ax[1].set_title("Model Output (Probabilities)")
    ax[1].axis('off')

    ax[2].imshow(img, cmap='gray')
    if num_cells > 0:
        ax[2].plot(coords[:, 1], coords[:, 0], 'r.', markersize=6)
    ax[2].set_title(f"Detections (Count: {num_cells})")
    ax[2].axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(qc_dir, f"{name}_overlay.png"), bbox_inches='tight')


_FNAME_RE = re.compile(r'(?:t(\d{3}))?([A-H]\d{2})f(\d{2})d(\d)', re.IGNORECASE)


def _parse_brightfield(file_path):
    """Return (well, field, time_idx) for a d0 brightfield file, else None."""
    match = _FNAME_RE.search(os.path.basename(file_path))
    if not match:
        return None
    time_str, well, field, channel = match.groups()
    if channel != '0':
        return None
    return well, field, int(time_str) if time_str else 0


def run_torch_batch_pipeline(backend, all_files, threshold, output_dir, treatments,
                             qc_per_well=3, batch_size=8, num_loaders=12):
    """High-throughput GPU pipeline: threaded disk loading overlapped with
    batched, fully-on-GPU preprocessing + inference, and threaded
    post-processing (peak detection + sampled QC overlays)."""
    # 1. Identify valid brightfield files and group by well.
    items = []  # (file_path, basename, well, field, time_idx)
    wells = defaultdict(list)
    for fp in all_files:
        parsed = _parse_brightfield(fp)
        if parsed is None:
            continue
        well, field, time_idx = parsed
        items.append((fp, os.path.basename(fp), well, field, time_idx))
        wells[well].append(fp)

    # 2. Pick a random sample of fields per well to render QC overlays for.
    qc_set = set()
    if qc_per_well and qc_per_well > 0:
        rng = random.Random(0)
        for well, fps in wells.items():
            qc_set.update(rng.sample(fps, min(qc_per_well, len(fps))))
        os.makedirs(os.path.join(output_dir, "QC"), exist_ok=True)

    print(f"Processing {len(items)} brightfield images on GPU "
          f"(batch_size={batch_size}, loaders={num_loaders}, ~{len(qc_set)} QC overlays)...")

    try:
        from tqdm import tqdm
    except ImportError:  # progress bar is optional
        def tqdm(*a, **k):
            class _Noop:
                def update(self, n=1): pass
                def set_postfix_str(self, s): pass
                def close(self): pass
            return _Noop()

    qc_dir = os.path.join(output_dir, "QC")
    results = []
    post_ex = concurrent.futures.ThreadPoolExecutor(max_workers=num_loaders)
    post_futures = []
    pbar = tqdm(total=len(items), unit="img", desc="Inference")

    def load_raw(item):
        img = load_as_float(item[0])
        if img is None:
            return None
        return item, np.ascontiguousarray(img, dtype=np.float32)

    def postprocess(item, probs, zscored):
        fp, basename, well, field, time_idx = item
        coords = peak_local_max(probs, min_distance=5, threshold_abs=threshold)
        num_cells = len(coords)
        if fp in qc_set:
            _save_qc_overlay(qc_dir, basename, zscored, probs, coords, num_cells)
        treatment = treatments.get(well, "Unknown") if treatments else "Unknown"
        return {'File': basename, 'FilePath': fp, 'Well': well, 'Field': field,
                'Time': time_idx, 'Treatment': treatment, 'Count': num_cells}

    def flush(batch):
        if not batch:
            return
        probs, zscored = backend.process_batch([b[1] for b in batch])
        for i, (item, _) in enumerate(batch):
            post_futures.append(post_ex.submit(postprocess, item, probs[i], zscored[i]))
        pbar.update(len(batch))
        pbar.set_postfix_str(f"well {batch[-1][0][2]}")  # most recently processed well

    # 3. Bounded-prefetch loader feeding shape-grouped GPU batches.
    loader_ex = concurrent.futures.ThreadPoolExecutor(max_workers=num_loaders)
    it = iter(items)
    inflight = deque()
    prefetch = num_loaders * 2 + batch_size
    for _ in range(prefetch):
        nxt = next(it, None)
        if nxt is None:
            break
        inflight.append(loader_ex.submit(load_raw, nxt))

    buckets = defaultdict(list)  # shape -> list of (item, raw_img)
    while inflight:
        loaded = inflight.popleft().result()
        nxt = next(it, None)
        if nxt is not None:
            inflight.append(loader_ex.submit(load_raw, nxt))
        if loaded is None:
            continue
        item, raw = loaded
        bucket = buckets[raw.shape]
        bucket.append((item, raw))
        if len(bucket) >= batch_size:
            flush(bucket)
            buckets[raw.shape] = []

    for bucket in buckets.values():
        flush(bucket)
    pbar.close()
    loader_ex.shutdown()

    # 4. Drain the post-processing (peak detection + sampled QC overlays).
    for f in concurrent.futures.as_completed(post_futures):
        r = f.result()
        if r is not None:
            results.append(r)
    post_ex.shutdown()

    print(f"Processed {len(results)} images across {len(wells)} wells.")
    return results

def generate_plots(well_time_counts, output_dir):
    # 1. Original Plot (Aggregated by Treatment)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=well_time_counts, x='Time', y='Count', hue='Treatment', err_style="bars", marker='o')
    plt.title('Cell Count Over Time')
    plt.ylabel('Total Cell Count (Per Well)')
    plt.xlabel('Time')
    plt.legend(title='Treatment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cell_count_over_time.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved lineplot to {plot_path}.")
    
    # Calculate initial counts for normalization per well
    initial_counts = well_time_counts.loc[well_time_counts.groupby('Well')['Time'].idxmin()]
    initial_counts_dict = initial_counts.set_index('Well')['Count'].to_dict()
    well_time_counts['Initial_Count'] = well_time_counts['Well'].map(initial_counts_dict)
    
    # Calculate Percentage Change, preventing division by zero
    mask = well_time_counts['Initial_Count'] > 0
    well_time_counts['Pct_Change'] = np.nan
    well_time_counts.loc[mask, 'Pct_Change'] = ((well_time_counts.loc[mask, 'Count'] - well_time_counts.loc[mask, 'Initial_Count']) / well_time_counts.loc[mask, 'Initial_Count']) * 100
    
    # 2. Normalized Plot (Aggregated by Treatment)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=well_time_counts, x='Time', y='Pct_Change', hue='Treatment', err_style="bars", marker='o')
    plt.title('Normalized Cell Count Over Time')
    plt.ylabel('Percentage Change from Initial (%)')
    plt.xlabel('Time')
    plt.legend(title='Treatment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path_norm = os.path.join(output_dir, "cell_count_over_time_normalized.png")
    plt.savefig(plot_path_norm, bbox_inches='tight')
    plt.close()
    print(f"Saved normalized lineplot to {plot_path_norm}.")
    
    # 3. Faceted Unnormalized Plot (Individual wells)
    g = sns.relplot(data=well_time_counts, x='Time', y='Count', hue='Well', col='Treatment', col_wrap=4, kind='line', marker='o', height=4, aspect=1)
    g.set_axis_labels("Time", "Total Cell Count")
    g.set_titles("{col_name}")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Cell Count Over Time per Well')
    plot_path_fac = os.path.join(output_dir, "cell_count_per_well_faceted.png")
    g.savefig(plot_path_fac)
    plt.close(g.fig)
    print(f"Saved faceted plot to {plot_path_fac}.")
    
    # 4. Faceted Normalized Plot (Individual wells)
    g_norm = sns.relplot(data=well_time_counts, x='Time', y='Pct_Change', hue='Well', col='Treatment', col_wrap=4, kind='line', marker='o', height=4, aspect=1)
    g_norm.set_axis_labels("Time", "Percentage Change (%)")
    g_norm.set_titles("{col_name}")
    g_norm.fig.subplots_adjust(top=0.9)
    g_norm.fig.suptitle('Normalized Cell Count Over Time per Well')
    plot_path_fac_norm = os.path.join(output_dir, "cell_count_per_well_faceted_normalized.png")
    g_norm.savefig(plot_path_fac_norm)
    plt.close(g_norm.fig)
    print(f"Saved normalized faceted plot to {plot_path_fac_norm}.")

def replot_only(output_dir, dtt_path):
    print(f"Running in plot-only mode using data in {output_dir}")
    
    treatments = {}
    if dtt_path and os.path.exists(dtt_path):
        print(f"Parsing DTT file: {dtt_path}")
        treatments = parse_tecan_d300e_xml(dtt_path)
    else:
        print(f"Warning: DTT file {dtt_path} not found. Cannot update treatments.")
        return

    time_xlsx = os.path.join(output_dir, "counts_per_timepoint.xlsx")
    well_xlsx = os.path.join(output_dir, "per_well_counts.xlsx")
    
    if os.path.exists(time_xlsx):
        all_dfs = []
        xl = pd.ExcelFile(time_xlsx)
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            time_str = sheet_name.replace('Time_', '')
            try:
                t = int(time_str)
            except ValueError:
                t = 0
            df['Time'] = t
            all_dfs.append(df)
            
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df['Treatment'] = combined_df['Well'].map(lambda w: treatments.get(w, "Unknown"))
            
            with pd.ExcelWriter(time_xlsx) as writer:
                for t, group in combined_df.groupby('Time'):
                    t_well_counts = group.groupby(['Well', 'Treatment'])['Count'].sum().reset_index()
                    t_well_counts.to_excel(writer, sheet_name=f'Time_{int(t)}', index=False)
            print(f"Updated per-timepoint sheets in {time_xlsx}.")
            
            well_time_counts = combined_df.groupby(['Time', 'Well', 'Treatment'])['Count'].sum().reset_index()
            generate_plots(well_time_counts, output_dir)
            
    elif os.path.exists(well_xlsx):
        df = pd.read_excel(well_xlsx)
        df['Treatment'] = df['Well'].map(lambda w: treatments.get(w, "Unknown"))
        df.to_excel(well_xlsx, index=False)
        print(f"Updated per well counts in {well_xlsx}.")
    else:
        print(f"No existing Excel files found in {output_dir} to replot.")

def run_inference(onnx_path, input_path, threshold=0.5, output_dir="results", dtt_path=None, save_qc=False, max_workers=4, backend="auto", qc_per_well=3, batch_size=8):
    if os.path.isdir(input_path):
        run_batch_inference(onnx_path, input_path, threshold, output_dir, dtt_path, save_qc, max_workers, backend, qc_per_well, batch_size)
    else:
        run_single_inference(onnx_path, input_path, threshold, output_dir, backend)

def run_single_inference(onnx_path, image_path, threshold=0.5, output_dir="results", backend="auto"):
    print(f"Running inference on: {image_path}")

    predict, backend_name, is_gpu, _ = create_predictor(onnx_path, backend)
    if is_gpu:
        print(f"Successfully loaded model onto GPU via {backend_name}.")
    else:
        print(f"Warning: no GPU backend available. Running on {backend_name}.")

    # Load and preprocess image
    img = load_as_float(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
        
    # Pseudo-flatfield correction (Background Subtraction)
    # Must match the preprocessing used during training
    bg = ndi.gaussian_filter(img, sigma=50.0)
    img = img - bg
    
    # Safe Per-Image Standardization (Z-score)
    std = np.std(img)
    img = (img - np.mean(img)) / max(std, 1e-2)

    # Pad image
    padded_img, orig_h, orig_w = pad_to_multiple(img, multiple=8)
    
    # Prepare input tensor
    input_tensor = np.expand_dims(padded_img, axis=(0, 1)).astype(np.float32)

    # Execute model
    logits = predict(input_tensor)

    # Restore original image dimensions
    logits = logits[:orig_h, :orig_w]
    
    # Post-process
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Extract count via local maxima
    coords = peak_local_max(probs, min_distance=5, threshold_abs=threshold)
    num_cells = len(coords)
    print(f">>> Final Counted Cells: {num_cells} <<<\n")

    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path)
    name, _ = os.path.splitext(base_name)

    # Render results
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original Brightfield")
    ax[0].axis('off')

    ax[1].imshow(probs, cmap='viridis')
    ax[1].set_title("Model Output (Probabilities)")
    ax[1].axis('off')

    ax[2].imshow(img, cmap='gray')
    
    if num_cells > 0:
        ax[2].plot(coords[:, 1], coords[:, 0], 'r.', markersize=6)
        
    ax[2].set_title(f"Detections (Count: {num_cells})")
    ax[2].axis('off')

    plt.tight_layout()
    
    # Save plot
    out_img_path = os.path.join(output_dir, f"{name}_overlay.png")
    plt.savefig(out_img_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {out_img_path}")

    # Record metrics
    csv_path = os.path.join(output_dir, "counts.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Image_Name', 'Cell_Count'])
        writer.writerow([base_name, num_cells])
    print(f"Appended count to {csv_path}\n")

def run_batch_inference(onnx_path, input_dir, threshold=0.5, output_dir="results", dtt_path=None, save_qc=False, max_workers=4, backend="auto", qc_per_well=3, batch_size=8):
    print(f"Running batch inference on directory: {input_dir}")

    predict, backend_name, is_gpu, torch_backend = create_predictor(onnx_path, backend)
    if is_gpu:
        print(f"Successfully loaded model onto GPU via {backend_name}.")
    else:
        print(f"Warning: no GPU backend available. Running on {backend_name}.")

    # onnxruntime CPU parallelises across processes; the onnx GPU path runs serially.
    is_cpu = (torch_backend is None) and (not is_gpu)

    treatments = {}
    if dtt_path and os.path.exists(dtt_path):
        print(f"Parsing DTT file: {dtt_path}")
        treatments = parse_tecan_d300e_xml(dtt_path)

    search_pattern = os.path.join(input_dir, "**", "*.[tT][iI][fF]*")
    all_files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(all_files)} TIFF files under {input_dir}")

    results = []

    os.makedirs(output_dir, exist_ok=True)

    if torch_backend is not None:
        results = run_torch_batch_pipeline(
            torch_backend, all_files, threshold, output_dir, treatments,
            qc_per_well=qc_per_well, batch_size=batch_size, num_loaders=max_workers)
    elif is_cpu:
        workers = min(os.cpu_count() or 1, max_workers)
        print(f"Using ProcessPoolExecutor with {workers} workers to balance speed and memory usage...")
        args_list = [(f, onnx_path, threshold, output_dir, save_qc, treatments) for f in all_files]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(process_file_worker, args_list, chunksize=1):
                if result is not None:
                    results.append(result)
    else:
        for file_path in all_files:
            result = process_single_file(file_path, predict, threshold, output_dir, save_qc, treatments)
            if result is not None:
                results.append(result)

    if not results:
        print("No valid brightfield images found or processed.")
        return
        
    results_by_subfolder = {}
    for result in results:
        rel_dir = os.path.relpath(os.path.dirname(result['FilePath']), input_dir)
        if rel_dir == '.':
            rel_dir = 'root'
        results_by_subfolder.setdefault(rel_dir, []).append(result)

    def sanitize_sheet_name(name):
        invalid_chars = r'[\\\/?*\[\]]'
        safe_name = re.sub(invalid_chars, '_', name)
        if len(safe_name) > 31:
            safe_name = safe_name[:31]
        return safe_name

    subfolder_workbook = os.path.join(output_dir, "per_well_counts_by_subfolder.xlsx")
    sheet_names = {}
    with pd.ExcelWriter(subfolder_workbook) as writer:
        for rel_dir, group in results_by_subfolder.items():
            df_group = pd.DataFrame(group)
            per_well = df_group.groupby(['Well', 'Treatment'])['Count'].sum().reset_index()

            sheet_name = sanitize_sheet_name(rel_dir)
            if sheet_name in sheet_names:
                sheet_name = f"{sheet_name[:24]}_{sheet_names[sheet_name] + 1}"
                sheet_name = sanitize_sheet_name(sheet_name)
                sheet_names[sheet_name] = 1
            else:
                sheet_names[sheet_name] = 1

            per_well.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Saved per-well counts for '{rel_dir}' as sheet '{sheet_name}' in {subfolder_workbook}.")

    combined_out_xlsx = os.path.join(output_dir, "per_well_counts_all_subfolders.xlsx")
    combined_df = pd.DataFrame(results)
    combined_per_well = combined_df.groupby(['Well', 'Treatment'])['Count'].sum().reset_index()
    combined_per_well.to_excel(combined_out_xlsx, index=False)
    print(f"Saved combined per-well counts to {combined_out_xlsx}.")

    if combined_df['Time'].notnull().any():
        combined_df['Time'] = combined_df['Time'].fillna(0)
        well_time_counts = combined_df.groupby(['Time', 'Well', 'Treatment'])['Count'].sum().reset_index()
        generate_plots(well_time_counts, output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run cell counter inference")
    parser.add_argument("--model", type=str, default="Vandria_cell_counter_model_IPSC.pth", help="Path to ONNX model")
    parser.add_argument("--input", type=str, default="./ACA16 data", help="Path to image file or directory")
    parser.add_argument("--dtt", type=str, default="./Tecan files/2026-06-02_VNA _p3_RHO182.tdd", help="Path to Tecan D300e .dtt file")
    parser.add_argument("--threshold", type=float, default=0.2, help="Detection threshold")
    parser.add_argument("--output", type=str, default="Results/ACA16", help="Output directory")
    parser.add_argument("--save-qc", default=False, action="store_true", help="onnx/CPU path only: save a QC overlay for every field")
    parser.add_argument("--qc-per-well", type=int, default=10, help="GPU/torch path: number of randomly sampled fields per well to save QC overlays for (0 disables)")
    parser.add_argument("--batch-size", type=int, default=4, help="GPU/torch path: images per forward pass")
    parser.add_argument("--max-workers", type=int, default=12, help="CPU workers: process pool size (onnx path) or loader/post-processing threads (GPU path)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "torch", "onnx"],
                        help="Inference backend. 'auto' uses PyTorch+CUDA when a .pth and GPU are available, else onnxruntime.")
    parser.add_argument("--plot-only", default=False, action="store_true", help="Do not run inference; only re-parse tecan file and replot existing results")
    args = parser.parse_args()
    
    dtt_path = args.dtt
    if dtt_path and not os.path.exists(dtt_path):
        if os.path.exists(dtt_path + ".dtt"):
            dtt_path += ".dtt"
        elif os.path.exists(dtt_path + ".xml"):
            dtt_path += ".xml"
            
    if args.plot_only:
        replot_only(args.output, dtt_path)
    else:
        if not os.path.exists(args.model):
            print(f"Cannot find {args.model}. Make sure you run train.py first!")
        else:
            run_inference(args.model, args.input, threshold=args.threshold, output_dir=args.output, dtt_path=dtt_path, save_qc=args.save_qc, max_workers=args.max_workers, backend=args.backend, qc_per_well=args.qc_per_well, batch_size=args.batch_size)