import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import CellDataset
from skimage.feature import peak_local_max
from model import UNet
from tqdm import tqdm


TF_ENABLE_ONEDNN_OPTS=0

class HeatmapLoss(nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        return self.mse(probs, targets)

def train_pipeline():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    CROP_SIZE = (256, 256)
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    NUM_WORKERS = 8

    # Initialize model
    model = UNet(n_channels=1, n_classes=1).to(device)

    # Setup DataLoaders
    data_dir = r"C:\Users\lab\OneDrive - Vandria\Bureau\Cell-Counter\IPSC data"
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Setup loss and optimizer
    criterion = HeatmapLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/cell_counter_IPSC')

    # Training loop
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train")
        for bf, hoechst in train_loader_tqdm:
            bf = bf.to(device)
            hoechst = hoechst.to(device)

            optimizer.zero_grad()
            outputs = model(bf)
            loss = criterion(outputs, hoechst)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(1, len(train_loader))
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_count_error = 0.0
        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Val")
            for val_bf, val_hoechst in val_loader_tqdm:
                val_bf = val_bf.to(device)
                val_hoechst = val_hoechst.to(device)
                
                val_outputs = model(val_bf)
                v_loss = criterion(val_outputs, val_hoechst)
                val_loss += v_loss.item()
                val_loader_tqdm.set_postfix(loss=f"{v_loss.item():.4f}")
                
                # Compute count accuracy using local maxima
                preds = torch.sigmoid(val_outputs).cpu().numpy()
                targets = val_hoechst.cpu().numpy()
                for b in range(preds.shape[0]):
                    pred_coords = peak_local_max(preds[b, 0], min_distance=5, threshold_abs=0.3)
                    target_coords = peak_local_max(targets[b, 0], min_distance=5, threshold_abs=0.5)
                    val_count_error += abs(len(pred_coords) - len(target_coords))

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_mae = val_count_error / max(1, len(val_dataset))
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Count MAE: {val_mae:.2f}")
        
        # Log metrics
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Metrics/Val_Count_MAE', val_mae, epoch)
        
        # Step the scheduler and log learning rate
        scheduler.step(avg_val_loss)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log sample images
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            writer.add_images('Images/Brightfield', val_bf, epoch)
            writer.add_images('Images/GroundTruth', val_hoechst, epoch)
            
            # Generate heatmap for visualization
            preds = torch.sigmoid(val_outputs)
            writer.add_images('Images/Predictions', preds, epoch)

    writer.close()
    
    # Save PyTorch state dictionary for future training
    pth_path = "Vandria_cell_counter_model_IPSC.pth"
    # Transfer model to CPU prior to saving to guarantee compatibility
    # across devices without GPU support.
    torch.save(model.to("cpu").state_dict(), pth_path)
    print(f"Saved PyTorch weights for future training to {pth_path}")

    # Export to ONNX
    model.eval().to("cpu")
    
    # Trace computational graph
    dummy_input = torch.randn(1, 1, CROP_SIZE[0], CROP_SIZE[1]).to("cpu")
    onnx_path = "Vandria_cell_counter_model_IPSC.onnx"

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