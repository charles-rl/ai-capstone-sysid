import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from training_models import CNNLSTMModel

# --- CONFIGURATION ---
DATA_PATH = "../data/processed_actuator_sysid_dataset.npz"
CHKPT_PATH = "../models/best_sysid_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 30
BATCH_SIZE = 64

CONFIG = {
    "in_channels": 4,      # theta, omega, cos(theta), sin(theta)
    "learning_rate": 1e-3,
    "cnn1_dims": 32,
    "cnn2_dims": 32,
    "lstm_dims": 64,
    "weight_decay": 1e-5,
    "clip_value": 1.0
}

# --- DATASET CLASS ---
class SysIDDataset(Dataset):
    def __init__(self, x_data, y_data):
        # PyTorch Conv1d expects shape (Batch, Channels, Sequence Length)
        # We must permute from (N, 600, 4) -> (N, 4, 600)
        self.x = torch.tensor(x_data, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# --- TRAINING LOOP ---
def train():
    os.makedirs(os.path.dirname(CHKPT_PATH), exist_ok=True)
    
    print(f"Using device: {DEVICE}")
    print("Loading data...")
    data = np.load(DATA_PATH)
    
    train_dataset = SysIDDataset(data['X_train'], data['Y_train'])
    val_dataset   = SysIDDataset(data['X_val'],   data['Y_val'])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    
    model = CNNLSTMModel(config=CONFIG, n_params=3, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
    
    train_losses = []
    val_losses = []
    val_mses =[]
    
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(EPOCHS):
        # -- TRAIN PASS --
        model.train()
        epoch_train_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            loss = model.learn(batch_x, batch_y)
            epoch_train_loss += loss * batch_x.size(0)
            
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # -- VAL PASS --
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mse = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                
                mu, sigma = model.forward(batch_x)
                
                # Calculate NLL Loss
                loss = model.loss(mu, batch_y, sigma.pow(2))
                epoch_val_loss += loss.item() * batch_x.size(0)
                
                # Calculate MSE (For the rubric / human readability)
                mse = torch.nn.functional.mse_loss(mu, batch_y)
                epoch_val_mse += mse.item() * batch_x.size(0)
                
        epoch_val_loss /= len(val_loader.dataset)
        epoch_val_mse /= len(val_loader.dataset)
        
        val_losses.append(epoch_val_loss)
        val_mses.append(epoch_val_mse)
        
        print(f"Epoch {epoch+1:02d} | Train NLL: {epoch_train_loss:.4f} | Val NLL: {epoch_val_loss:.4f} | Val MSE: {epoch_val_mse:.4f}")
        
        # -- SAVE BEST MODEL --
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model.save_model()
            print("  --> Saved new best model!")

    # -- PLOT LEARNING CURVES --
    plot_results(train_losses, val_losses, val_mses)

def plot_results(train_losses, val_losses, val_mses):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: NLL Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train NLL Loss')
    plt.plot(epochs, val_losses, label='Val NLL Loss')
    plt.title('Gaussian NLL Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: MSE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_mses, color='green', label='Val MSE')
    plt.title('Validation Mean Squared Error (Rubric Metric)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("../models/training_curves.png")
    plt.show()

if __name__ == "__main__":
    train()