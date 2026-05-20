import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import CellDataset
from model import UNet

def train_pipeline():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    CROP_SIZE = (256, 256)

    # Initialize model
    model = UNet(n_channels=1, n_classes=1).to(device)

    # Setup DataLoaders
    data_dir = "C:/Users/lab/OneDrive - Vandria/Bureau/Cell counter training dataset/Plate 1"
    os.makedirs(data_dir, exist_ok=True)

    # Load and split dataset
    full_dataset = CellDataset(data_dir=data_dir, is_train=False, crop_size=CROP_SIZE)
    
    if len(full_dataset) == 0:
        print(f"Data directory is empty. Please place matched pairs in '{data_dir}'.")
        return

    random.shuffle(full_dataset.pairs)
    train_size = int(0.8 * len(full_dataset.pairs))

    train_dataset = CellDataset(data_dir=data_dir, is_train=True, crop_size=CROP_SIZE)
    train_dataset.pairs = full_dataset.pairs[:train_size]

    val_dataset = CellDataset(data_dir=data_dir, is_train=False, crop_size=CROP_SIZE)
    val_dataset.pairs = full_dataset.pairs[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Setup loss and optimizer
    pos_weight = torch.tensor([9.0])  #ig we have mainly background
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/cell_counter_experiment_pos_weight_9')

    # Training loop
    epochs = 50
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for bf, hoechst in train_loader:
            bf = bf.to(device)
            hoechst = hoechst.to(device)

            optimizer.zero_grad()
            outputs = model(bf)
            loss = criterion(outputs, hoechst)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(train_loader))
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_bf, val_hoechst in val_loader:
                val_bf = val_bf.to(device)
                val_hoechst = val_hoechst.to(device)
                
                val_outputs = model(val_bf)
                v_loss = criterion(val_outputs, val_hoechst)
                val_loss += v_loss.item()
                
        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Log metrics
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        
        # Log sample images
        if epoch % 5 == 0 or epoch == epochs - 1:
            writer.add_images('Images/Brightfield', val_bf, epoch)
            writer.add_images('Images/GroundTruth', val_hoechst, epoch)
            
            # Generate binary mask for visualization
            preds = (torch.sigmoid(val_outputs) > 0.5).float()
            writer.add_images('Images/Predictions', preds, epoch)

    writer.close()

    # Export to ONNX
    model.eval()
    
    # Trace computational graph
    dummy_input = torch.randn(1, 1, CROP_SIZE[0], CROP_SIZE[1]).to(device)
    onnx_path = "cell_counter_model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
    )
    print(f"Training complete. Model exported to {onnx_path}")

if __name__ == "__main__":
    train_pipeline()