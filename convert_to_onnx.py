import torch
import os
from model import UNet

def convert_pth_to_onnx(pth_path, onnx_path, n_channels=1, n_classes=1, image_size=512):
    """
    Convert a PyTorch .pth model to ONNX format.
    
    Args:
        pth_path (str): Path to the .pth checkpoint file
        onnx_path (str): Path where the .onnx file will be saved
        n_channels (int): Number of input channels (default: 1 for grayscale)
        n_classes (int): Number of output classes (default: 1 for cell counting)
        image_size (int): Size of the input image (default: 512)
    """
    
    # Check if the .pth file exists
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model file not found: {pth_path}")
    
    print(f"Loading model from: {pth_path}")
    
    # Initialize the model
    model = UNet(n_channels=n_channels, n_classes=n_classes)
    
    # Load the checkpoint
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # If checkpoint is a dict with 'model_state_dict' key
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # If checkpoint is a dict with 'state_dict' key
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # If checkpoint is the state_dict directly
        model.load_state_dict(checkpoint)
    
    print("Model loaded successfully")
    model.eval()
    
    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, n_channels, image_size, image_size)
    
    print(f"Converting model to ONNX format...")
    print(f"Input shape: {dummy_input.shape}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {2: 'height', 3: 'width'},
            'output': {2: 'height', 3: 'width'}
        },
        verbose=False
    )
    
    print(f"Model successfully converted and saved to: {onnx_path}")
    
    # Verify the ONNX model (optional)
    try:
        import onnx
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        print("ONNX model validation passed!")
    except ImportError:
        print("ONNX package not installed. Skipping validation.")
    except Exception as e:
        print(f"Warning: ONNX validation failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PyTorch .pth model to ONNX format")
    parser.add_argument(
        "--pth-path",
        type=str,
        default="Vandria_cell_counter_model_HMC3.pth",
        help="Path to the .pth model file"
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="Vandria_cell_counter_model_HMC3.onnx",
        help="Path where the .onnx model will be saved"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Input image size for dummy input (default: 512)"
    )
    
    args = parser.parse_args()
    
    # Convert the model
    convert_pth_to_onnx(args.pth_path, args.onnx_path, image_size=args.image_size)
