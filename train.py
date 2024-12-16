import os

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from CBBCE import CBBCE
from dataset import RadarDataset
from model import DAFCRadarNet


def calculate_ratio(train_loader, detection_type):
    """Calculate ratio of positive samples in dataset from 2D labels"""
    n1 = 0  # target samples
    n_total = len(train_loader.dataset)
    n_total *= 32 if detection_type == 'range' else 63
    for _, rd_label in train_loader:
        # Get 1D labels by summing across appropriate dimension
        label = (rd_label.sum(dim=-1 if detection_type == "range" else -2) >= 1).float()
        # Count bins with targets
        n1 += torch.sum(label >= 0.9999)
    ratio = n1.item() / n_total
    print("ratio:", ratio, ", n1:", n1.item(), ", n_total:", n_total)
    return ratio


def train_model(model, criterion, train_loader, val_loader, detection_type, epochs=300, learning_rate=1e-3,
                weight_decay=5e-4):
    """
    Train range or Doppler detector

    Args:
        model: Neural network model
        criterion: Loss function
        train_loader: Training data loader
        val_loader: Validation data loader
        detection_type: "range" or "doppler"
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to device
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Adam optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.905, patience=5)  # LR scheduler

    # Training history
    history = {"train_loss": [], "val_loss": []}

    # Early stopping setup
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None

    epoch_pbar = tqdm(range(epochs), desc="Training Progress")
    for epoch in epoch_pbar:
        # Training stage
        model.train()
        train_loss = 0

        for X, rd_label in train_loader:
            # Get input and label
            X = X.to(device)
            rd_label = rd_label.to(device)
            label = (rd_label.sum(dim=-1 if detection_type == "range" else -2) >= 1).float()

            outputs = model(X)
            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation stage
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, rd_label in val_loader:
                # Get input and label
                X = X.to(device)
                rd_label = rd_label.to(device)
                label = (rd_label.sum(dim=-1 if detection_type == "range" else -2) >= 1).float()

                # Forward pass
                outputs = model(X)
                loss = criterion(outputs, label)
                val_loss += loss.item()

        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)

        # Update history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        # Learning rate scheduling
        scheduler.step(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Print epoch results
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Create saving directories
    models_dir = os.path.join("models")
    history_dir = os.path.join("training_history")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(models_dir, f"{detection_type}model.pt")
    torch.save(model.state_dict(), model_path)

    # Save training history
    history_path = os.path.join(history_dir, f"{detection_type}_training_data.pt")
    torch.save({'history': history}, history_path)

    return history


def plot_training_history(history: dict, detection_type: str):
    """Plot training and validation metrics"""
    plt.figure(figsize=(15.12, 9.82))

    # Plot loss
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title(f"{detection_type} Detector Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plots_dir = os.path.join("plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{detection_type}_training_plot.png")
    plt.savefig(plot_path)
    plt.close()


def train():
    # Create datasets with and without targets
    train_dataset_with_targets = RadarDataset(num_samples=1024, n_targets=8, random_n_targets=True)
    train_dataset_no_targets = RadarDataset(num_samples=1024, n_targets=0)

    val_dataset_with_targets = RadarDataset(num_samples=256, n_targets=8, random_n_targets=True)
    val_dataset_no_targets = RadarDataset(num_samples=256, n_targets=0)

    # Combine datasets
    train_dataset = ConcatDataset([train_dataset_with_targets, train_dataset_no_targets])
    val_dataset = ConcatDataset([val_dataset_with_targets, val_dataset_no_targets])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2,
                              pin_memory=torch.cuda.is_available(), persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=2,
                            pin_memory=torch.cuda.is_available(), persistent_workers=True)

    # Train range detector
    print("\nTraining Range Detector:")
    detection_type = "range"
    ratio = calculate_ratio(train_loader, detection_type)
    criterion = CBBCE(ratio)
    range_model = DAFCRadarNet(detection_type)
    range_history = train_model(range_model, criterion, train_loader, val_loader, detection_type)
    plot_training_history(range_history, detection_type)

    # Train Doppler detector
    # print("\nTraining Doppler Detector:")
    # detection_type = "doppler"
    # ratio = calculate_ratio(train_loader, detection_type)
    # criterion = CBBCE(ratio)
    # doppler_model = DAFCRadarNet(detection_type)
    # doppler_history = train_model(doppler_model, criterion, train_loader, val_loader, detection_type)
    # plot_training_history(doppler_history, detection_type)
