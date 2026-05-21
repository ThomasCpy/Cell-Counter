import os
import csv
import glob
import numpy as np
import onnxruntime as ort
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from dataset import load_as_float

def pad_to_multiple(img, multiple=8):
    """Pad image dimensions to a specified multiple."""
    h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
        
    return img, h, w

def run_inference(onnx_path, image_path, threshold=0.5, output_dir="results"):
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
    binary_mask = probs > threshold

    # Extract count
    labeled_mask, num_cells = ndi.label(binary_mask)
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
    ax[2].imshow(binary_mask, cmap='Reds', alpha=0.4)
    ax[2].set_title(f"Overlay (Count: {num_cells})")
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

if __name__ == "__main__":
    onnx_model_path = "cell_counter_model.onnx"
    
    test_image = "data/test_image.tif" 
    
    if not os.path.exists(onnx_model_path):
        print(f"Cannot find {onnx_model_path}. Make sure you run train.py first!")
    else:
        run_inference(onnx_model_path, test_image, threshold=0.5)