import os
import csv
import glob
import re
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
import onnxruntime as ort
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import load_as_float
from skimage.feature import peak_local_max
import concurrent.futures

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
            physical_pos = (int(r_well.get('R')), int(r_well.get('C')))
            logical_pos = (int(r_well.get('ToR')), int(r_well.get('ToC')))
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
        _worker_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        _worker_input_name = _worker_session.get_inputs()[0].name
    return _worker_session, _worker_input_name

def process_single_file(file_path, session, input_name, threshold, output_dir, save_qc, treatments):
    basename = os.path.basename(file_path)
    pattern = re.compile(r't(\d{3})([A-H]\d{2})f(\d{2})d(\d)')
    match = pattern.search(basename)
    if not match:
        return None
        
    time_str, well, field, channel = match.groups()
    if channel != '1':
        return None
    time_idx = int(time_str)
    
    img = load_as_float(file_path)
    if img is None:
        return None
        
    bg = ndi.gaussian_filter(img, sigma=50.0)
    img = img - bg
    std = np.std(img)
    img = (img - np.mean(img)) / max(std, 1e-2)

    padded_img, orig_h, orig_w = pad_to_multiple(img, multiple=8)
    input_tensor = np.expand_dims(padded_img, axis=(0, 1)).astype(np.float32)

    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0, 0]
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
        'Well': well,
        'Field': field,
        'Time': time_idx,
        'Treatment': treatment,
        'Count': num_cells
    }

def process_file_worker(args):
    file_path, onnx_path, threshold, output_dir, save_qc, treatments = args
    session, input_name = _get_worker_session(onnx_path)
    return process_single_file(file_path, session, input_name, threshold, output_dir, save_qc, treatments)

def run_inference(onnx_path, input_path, threshold=0.5, output_dir="results", dtt_path=None, save_qc=False, max_workers=4):
    if os.path.isdir(input_path):
        run_batch_inference(onnx_path, input_path, threshold, output_dir, dtt_path, save_qc, max_workers)
    else:
        run_single_inference(onnx_path, input_path, threshold, output_dir)

def run_single_inference(onnx_path, image_path, threshold=0.5, output_dir="results"):
    print(f"Running inference on: {image_path}")
    
    # Load the ONNX model with DirectML
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    active_provider = session.get_providers()[0]
    if active_provider == 'DmlExecutionProvider':
        print("Successfully loaded model onto GPU via DirectML.")
    else:
        print(f"Warning: DirectML failed to load. Falling back to {active_provider}.")
        
    input_name = session.get_inputs()[0].name

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
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0, 0]

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

def run_batch_inference(onnx_path, input_dir, threshold=0.5, output_dir="results", dtt_path=None, save_qc=False, max_workers=4):
    print(f"Running batch inference on directory: {input_dir}")
    
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    active_provider = session.get_providers()[0]
    
    is_cpu = active_provider != 'DmlExecutionProvider'
    if not is_cpu:
        print("Successfully loaded model onto GPU via DirectML.")
    else:
        print(f"Warning: DirectML failed to load. Falling back to {active_provider}.")
        
    input_name = session.get_inputs()[0].name

    treatments = {}
    if dtt_path and os.path.exists(dtt_path):
        print(f"Parsing DTT file: {dtt_path}")
        treatments = parse_tecan_d300e_xml(dtt_path)

    search_pattern = os.path.join(input_dir, "**", "*.[tT][iI][fF]*")
    all_files = glob.glob(search_pattern, recursive=True)
    
    results = []

    os.makedirs(output_dir, exist_ok=True)

    if is_cpu:
        workers = min(os.cpu_count() or 1, max_workers)
        print(f"Using ProcessPoolExecutor with {workers} workers to balance speed and memory usage...")
        args_list = [(f, onnx_path, threshold, output_dir, save_qc, treatments) for f in all_files]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(process_file_worker, args_list, chunksize=1):
                if result is not None:
                    results.append(result)
    else:
        for file_path in all_files:
            result = process_single_file(file_path, session, input_name, threshold, output_dir, save_qc, treatments)
            if result is not None:
                results.append(result)

    if not results:
        print("No valid brightfield images found or processed.")
        return
        
    df = pd.DataFrame(results)
    has_time = df['Time'].notnull().any()
    
    if not has_time:
        well_counts = df.groupby(['Well', 'Treatment'])['Count'].sum().reset_index()
        out_xlsx = os.path.join(output_dir, "per_well_counts.xlsx")
        well_counts.to_excel(out_xlsx, index=False)
        print(f"\nSaved per well counts to {out_xlsx} (no time indication detected).")
    else:
        df['Time'] = df['Time'].fillna(0)
        out_xlsx = os.path.join(output_dir, "counts_per_timepoint.xlsx")
        with pd.ExcelWriter(out_xlsx) as writer:
            for t, group in df.groupby('Time'):
                t_well_counts = group.groupby(['Well', 'Treatment'])['Count'].sum().reset_index()
                t_well_counts.to_excel(writer, sheet_name=f'Time_{int(t)}', index=False)
        print(f"\nSaved per-timepoint sheets to {out_xlsx}.")
        
        well_time_counts = df.groupby(['Time', 'Well', 'Treatment'])['Count'].sum().reset_index()
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run cell counter inference")
    parser.add_argument("--model", type=str, default="Vandria_cell_counter_model_All.onnx", help="Path to ONNX model")
    parser.add_argument("--input", type=str, default="./Data test run/localhost260602100003", help="Path to image file or directory")
    parser.add_argument("--dtt", type=str, default="./Tecan files/2026-06-02_VNA _p3_RHO182", help="Path to Tecan D300e .dtt file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--output", type=str, default="Results/Test run", help="Output directory")
    parser.add_argument("--save-qc", action="store_true", help="Save QC overlays during batch inference")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of CPU workers for parallel batch inference")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Cannot find {args.model}. Make sure you run train.py first!")
    else:
        run_inference(args.model, args.input, threshold=args.threshold, output_dir=args.output, dtt_path=args.dtt, save_qc=args.save_qc, max_workers=args.max_workers)