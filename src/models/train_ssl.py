"""
Self-Supervised Learning Training Script
Phase 1: Autoencoder pretraining on ALL transactions (unsupervised)
Phase 2: Anomaly scoring via reconstruction error
Phase 3: Optional - Use SSL embeddings as features for supervised model

This is an experimental module for the baseline project.
"""
import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[TRAIN-SSL] ERROR: PyTorch not installed. Install with: pip install torch")


# ================================
# Configuration
# ================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_advanced.parquet"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_advanced.parquet"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
RANDOM_STATE = 42

# SSL Hyperparameters
BATCH_SIZE = 512
LEARNING_RATE = 0.001
PRETRAIN_EPOCHS = 50
EMBEDDING_DIM = 16  # Bottleneck size
HIDDEN_DIMS = [128, 64, 32]  # Encoder/Decoder layers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set random seeds
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)


# ================================
# Autoencoder Model
# ================================

class Autoencoder(nn.Module):
    """
    Simple autoencoder for anomaly detection.
    
    Architecture:
        Encoder: Input → 128 → 64 → 32 → 16 (bottleneck)
        Decoder: 16 → 32 → 64 → 128 → Input (reconstruction)
    
    Fraud transactions → High reconstruction error
    Normal transactions → Low reconstruction error
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], embedding_dim: int):
        super(Autoencoder, self).__init__()
        
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim
        
        # Bottleneck
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder layers (reverse of encoder)
        decoder_layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Returns:
            Tuple of (reconstructed input, embedding).
        """
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)
        return reconstructed, embedding
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the latent embedding (bottleneck representation)."""
        return self.encoder(x)


# ================================
# Data Loading & Preparation
# ================================

def load_all_transactions(
    train_path: str,
    test_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load train and test data for SSL pretraining.
    SSL uses ALL transactions (no labels needed for pretraining).
    
    Args:
        train_path: Path to training data.
        test_path: Path to test data.
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, feature_names).
    """
    print(f"[TRAIN-SSL] Loading data...")
    
    # Load train data
    df_train = pd.read_parquet(train_path)
    target_col = "Class"
    feature_cols = [col for col in df_train.columns if col != target_col]
    
    X_train_raw = df_train[feature_cols].values.astype(np.float64)
    y_train = df_train[target_col].values.astype(np.int64)
    
    # Load test data
    df_test = pd.read_parquet(test_path)
    X_test_raw = df_test[feature_cols].values.astype(np.float64)
    y_test = df_test[target_col].values.astype(np.int64)
    
    print(f"[TRAIN-SSL] Train: {len(X_train_raw):,} rows, Test: {len(X_test_raw):,} rows")
    print(f"[TRAIN-SSL] Features: {len(feature_cols)}")
    print(f"[TRAIN-SSL] Train fraud rate: {y_train.mean()*100:.3f}%")
    print(f"[TRAIN-SSL] Test fraud rate: {y_test.mean()*100:.3f}%")
    
    return X_train_raw, y_train, X_test_raw, y_test, feature_cols


def prepare_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale data for autoencoder training.
    
    Args:
        X_train: Training features.
        X_test: Test features.
    
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler).
    """
    print(f"[TRAIN-SSL] Scaling data...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def create_dataloaders(
    X_train: np.ndarray,
    X_test: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        X_train: Scaled training features.
        X_test: Scaled test features.
        batch_size: Batch size for training.
    
    Returns:
        Tuple of (train_loader, test_loader).
    """
    # Convert to PyTorch tensors
    train_tensor = torch.FloatTensor(X_train)
    test_tensor = torch.FloatTensor(X_test)
    
    # Create datasets
    train_dataset = TensorDataset(train_tensor, train_tensor)  # Input = Target for autoencoder
    test_dataset = TensorDataset(test_tensor, test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, test_loader


# ================================
# Training
# ================================

def train_autoencoder(
    model: Autoencoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = PRETRAIN_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    device: str = DEVICE,
) -> Dict[str, List[float]]:
    """
    Train the autoencoder.
    
    Args:
        model: Autoencoder model.
        train_loader: Training data loader.
        test_loader: Validation data loader.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        device: Device to train on.
    
    Returns:
        Dictionary containing training history.
    """
    print(f"\n[TRAIN-SSL] {'=' * 50}")
    print(f"[TRAIN-SSL] Training Autoencoder on {device.upper()}")
    print(f"[TRAIN-SSL] {'=' * 50}")
    print(f"[TRAIN-SSL] Epochs: {epochs}")
    print(f"[TRAIN-SSL] Batch size: {BATCH_SIZE}")
    print(f"[TRAIN-SSL] Learning rate: {learning_rate}")
    print(f"[TRAIN-SSL] Embedding dim: {EMBEDDING_DIM}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        "train_loss": [],
        "val_loss": [],
    }
    
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            reconstructed, _ = model(batch_inputs)
            loss = criterion(reconstructed, batch_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                reconstructed, _ = model(batch_inputs)
                loss = criterion(reconstructed, batch_targets)
                
                val_loss += loss.item() * batch_inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        history["val_loss"].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"[TRAIN-SSL]   Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Best Val: {best_val_loss:.6f}")
    
    print(f"[TRAIN-SSL] Training complete. Best validation loss: {best_val_loss:.6f}")
    
    return history


# ================================
# Anomaly Scoring
# ================================

def compute_reconstruction_error(
    model: Autoencoder,
    data_loader: DataLoader,
    device: str = DEVICE,
) -> np.ndarray:
    """
    Compute reconstruction error for anomaly scoring.
    
    High error = anomalous (potential fraud).
    
    Args:
        model: Trained autoencoder.
        data_loader: Data loader.
        device: Device.
    
    Returns:
        Array of reconstruction errors.
    """
    model.eval()
    model = model.to(device)
    errors = []
    
    with torch.no_grad():
        for batch_inputs, batch_targets in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            reconstructed, _ = model(batch_inputs)
            
            # Per-sample MSE
            batch_errors = torch.mean((reconstructed - batch_targets) ** 2, dim=1)
            errors.extend(batch_errors.cpu().numpy())
    
    return np.array(errors)


def evaluate_anomaly_detection(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    dataset_name: str,
) -> Dict[str, float]:
    """
    Evaluate anomaly detection performance.
    
    Args:
        y_true: True labels (0=normal, 1=fraud).
        anomaly_scores: Reconstruction errors.
        dataset_name: Name for display.
    
    Returns:
        Dictionary of metrics.
    """
    print(f"\n[TRAIN-SSL] {'=' * 50}")
    print(f"[TRAIN-SSL] Anomaly Detection Results - {dataset_name}")
    print(f"[TRAIN-SSL] {'=' * 50}")
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    avg_precision = average_precision_score(y_true, anomaly_scores)
    
    # Precision-Recall AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, anomaly_scores)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Analyze score distributions
    normal_scores = anomaly_scores[y_true == 0]
    fraud_scores = anomaly_scores[y_true == 1]
    
    print(f"\n[TRAIN-SSL]   ROC-AUC:             {roc_auc:.4f}")
    print(f"[TRAIN-SSL]   PR-AUC:              {pr_auc:.4f}")
    print(f"[TRAIN-SSL]   Avg Precision:       {avg_precision:.4f}")
    print(f"[TRAIN-SSL]   Normal mean error:   {normal_scores.mean():.6f} (+/- {normal_scores.std():.6f})")
    print(f"[TRAIN-SSL]   Fraud mean error:    {fraud_scores.mean():.6f} (+/- {fraud_scores.std():.6f})")
    print(f"[TRAIN-SSL]   Error ratio:         {fraud_scores.mean() / normal_scores.mean():.2f}x")
    
    # Top-k analysis
    k = y_true.sum()
    top_k_indices = np.argsort(anomaly_scores)[::-1][:int(k)]
    fraud_in_top_k = y_true[top_k_indices].sum()
    print(f"[TRAIN-SSL]   Fraud in top-{int(k)}:  {fraud_in_top_k}/{int(k)} ({fraud_in_top_k/k*100:.1f}%)")
    
    return {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "average_precision": round(float(avg_precision), 4),
        "normal_mean_error": round(float(normal_scores.mean()), 6),
        "fraud_mean_error": round(float(fraud_scores.mean()), 6),
        "error_ratio": round(float(fraud_scores.mean() / normal_scores.mean()), 2),
        "fraud_in_top_k_pct": round(float(fraud_in_top_k / k * 100), 1),
    }


# ================================
# Embedding Extraction
# ================================

def extract_embeddings(
    model: Autoencoder,
    X: np.ndarray,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
) -> np.ndarray:
    """
    Extract latent embeddings for all samples.
    These can be used as features for supervised models.
    
    Args:
        model: Trained autoencoder.
        X: Feature matrix (scaled).
        batch_size: Batch size.
        device: Device.
    
    Returns:
        Embedding matrix (n_samples, embedding_dim).
    """
    model.eval()
    model = model.to(device)
    
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings = []
    
    with torch.no_grad():
        for (batch_data,) in loader:
            batch_data = batch_data.to(device)
            batch_embeddings = model.get_embedding(batch_data)
            embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)


# ================================
# Model Saving
# ================================

def save_ssl_model(
    model: Autoencoder,
    scaler: StandardScaler,
    feature_names: List[str],
    models_dir: str,
    model_name: str = "autoencoder",
) -> Dict[str, str]:
    """
    Save SSL model and preprocessing components.
    
    Args:
        model: Trained autoencoder.
        scaler: Fitted StandardScaler.
        feature_names: List of feature names.
        models_dir: Directory to save models.
        model_name: Base name for saved files.
    
    Returns:
        Dictionary of saved paths.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Save PyTorch model state
    model_path = os.path.join(models_dir, f"{model_name}_ssl.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": len(feature_names),
        "hidden_dims": HIDDEN_DIMS,
        "embedding_dim": EMBEDDING_DIM,
        "feature_names": feature_names,
    }, model_path)
    print(f"[TRAIN-SSL] Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(models_dir, f"{model_name}_ssl_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[TRAIN-SSL] Scaler saved to: {scaler_path}")
    
    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
    }


# ================================
# Main Pipeline
# ================================

def run_ssl_training(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    models_dir: Optional[str] = None,
    metrics_dir: Optional[str] = None,
    epochs: int = PRETRAIN_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    embedding_dim: int = EMBEDDING_DIM,
    device: str = DEVICE,
) -> Dict[str, Any]:
    """
    Execute the SSL pipeline.
    
    Args:
        train_path: Path to training data.
        test_path: Path to test data.
        models_dir: Directory to save models.
        metrics_dir: Directory to save metrics.
        epochs: Number of pretraining epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        embedding_dim: Bottleneck dimension.
        device: Device to use.
    
    Returns:
        Dictionary with results.
    """
    if not TORCH_AVAILABLE:
        print("[TRAIN-SSL] ERROR: PyTorch required. Install with: pip install torch")
        return {}
    
    # Resolve paths
    train_path = train_path or str(DEFAULT_TRAIN_PATH)
    test_path = test_path or str(DEFAULT_TEST_PATH)
    models_dir = models_dir or str(DEFAULT_MODELS_DIR)
    metrics_dir = metrics_dir or str(DEFAULT_METRICS_DIR)
    
    print("\n" + "=" * 60)
    print("SELF-SUPERVISED LEARNING - AUTOENCODER")
    print("=" * 60)
    print(f"[TRAIN-SSL] Device: {device.upper()}")
    print(f"[TRAIN-SSL] No labels needed for pretraining")
    print(f"[TRAIN-SSL] Using ALL transactions to learn normal behavior")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, y_test, feature_names = load_all_transactions(
        train_path, test_path
    )
    
    # Scale data
    X_train_scaled, X_test_scaled, scaler = prepare_data(X_train, X_test)
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(X_train_scaled, X_test_scaled, batch_size)
    
    # Create model
    input_dim = len(feature_names)
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        embedding_dim=embedding_dim,
    )
    
    print(f"\n[TRAIN-SSL] Model Architecture:")
    print(f"[TRAIN-SSL]   Input dim:    {input_dim}")
    print(f"[TRAIN-SSL]   Hidden dims:  {HIDDEN_DIMS}")
    print(f"[TRAIN-SSL]   Embedding:    {embedding_dim}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN-SSL]   Total params: {total_params:,}")
    
    # Train autoencoder (Phase 1: Pretraining)
    history = train_autoencoder(
        model, train_loader, test_loader, epochs, learning_rate, device
    )
    
    # Compute anomaly scores
    print(f"\n[TRAIN-SSL] Computing anomaly scores...")
    
    # Train set scores
    train_errors = compute_reconstruction_error(model, train_loader, device)
    train_metrics = evaluate_anomaly_detection(y_train, train_errors, "TRAIN SET")
    
    # Test set scores
    test_errors = compute_reconstruction_error(model, test_loader, device)
    test_metrics = evaluate_anomaly_detection(y_test, test_errors, "TEST SET")
    
    # Compare with LightGBM (the best supervised model)
    print(f"\n[TRAIN-SSL] {'=' * 50}")
    print(f"[TRAIN-SSL] COMPARISON: SSL vs Supervised")
    print(f"[TRAIN-SSL] {'=' * 50}")
    print(f"[TRAIN-SSL]   SSL Autoencoder PR-AUC:  {test_metrics['pr_auc']:.4f}")
    print(f"[TRAIN-SSL]   LightGBM (best) PR-AUC:   0.8121")
    print(f"[TRAIN-SSL]   Difference:                {test_metrics['pr_auc'] - 0.8121:+.4f}")
    
    if test_metrics['pr_auc'] > 0.5:
        print(f"[TRAIN-SSL]   ✅ SSL provides useful signal!")
        print(f"[TRAIN-SSL]   💡 Embeddings could enhance ensemble model")
    else:
        print(f"[TRAIN-SSL]   ⚠️ SSL alone insufficient for fraud detection")
        print(f"[TRAIN-SSL]   💡 Use SSL embeddings as features for LightGBM")
    
    # Extract embeddings for future use
    print(f"\n[TRAIN-SSL] Extracting SSL embeddings...")
    train_embeddings = extract_embeddings(model, X_train_scaled, batch_size, device)
    test_embeddings = extract_embeddings(model, X_test_scaled, batch_size, device)
    
    print(f"[TRAIN-SSL] Train embeddings shape: {train_embeddings.shape}")
    print(f"[TRAIN-SSL] Test embeddings shape:  {test_embeddings.shape}")
    
    # Save everything
    saved_paths = save_ssl_model(model, scaler, feature_names, models_dir)
    
    # Save metrics
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_summary = {
        "model_type": "autoencoder_ssl",
        "pretraining_epochs": epochs,
        "input_dim": input_dim,
        "embedding_dim": embedding_dim,
        "hidden_dims": HIDDEN_DIMS,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "training_history": {
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "best_val_loss": min(history["val_loss"]),
        },
        "comparison": {
            "ssl_pr_auc": test_metrics["pr_auc"],
            "lightgbm_pr_auc": 0.8121,
            "verdict": "SSL as supplementary feature source" if test_metrics["pr_auc"] < 0.8 else "SSL competitive standalone",
        },
    }
    
    metrics_path = os.path.join(metrics_dir, "autoencoder_ssl_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[TRAIN-SSL] Metrics saved to: {metrics_path}")
    
    print(f"\n{'=' * 60}")
    print("SSL TRAINING COMPLETE")
    print(f"{'=' * 60}")
    
    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "saved_paths": saved_paths,
        "train_embeddings": train_embeddings,
        "test_embeddings": test_embeddings,
    }


# ================================
# Entry Point
# ================================

def main():
    """Execute SSL training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Self-Supervised Learning Model (Autoencoder)"
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to training data (default: data/processed/train_advanced.parquet)",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Path to test data (default: data/processed/test_advanced.parquet)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory to save models",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help="Directory to save metrics",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=PRETRAIN_EPOCHS,
        help=f"Number of pretraining epochs (default: {PRETRAIN_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=EMBEDDING_DIM,
        help=f"Embedding dimension (default: {EMBEDDING_DIM})",
    )
    
    args = parser.parse_args()
    
    try:
        results = run_ssl_training(
            train_path=args.train_path,
            test_path=args.test_path,
            models_dir=args.models_dir,
            metrics_dir=args.metrics_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            embedding_dim=args.embedding_dim,
        )
        
        if results:
            print(f"\n[TRAIN-SSL] ✅ SSL experiment complete.")
            print(f"[TRAIN-SSL] Check artifacts/metrics/autoencoder_ssl_metrics.json")
        else:
            print("[TRAIN-SSL] ⚠️ SSL training failed.")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"[TRAIN-SSL] ❌ ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[TRAIN-SSL] ❌ UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()