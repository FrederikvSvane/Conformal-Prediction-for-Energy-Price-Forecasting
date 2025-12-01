#!/usr/bin/env python3
"""
Sequence Length Hyperparameter Search for LSTM Energy Price Forecasting

This script tests multiple sequence lengths using walk-forward cross-validation
and logs MSE, MAE, and RMSE metrics for each configuration.

Usage:
    python seq_len_search.py
"""

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models import LSTM_model
from dataset import EnergyPriceDataset, load_and_preprocess_energy_data

# =============================================================================
# Configuration
# =============================================================================

SEQUENCE_LENGTHS = [24, 48, 72, 96, 120, 144, 168, 192, 240, 336]
# 24=1day, 48=2days, 72=3days, 96=4days, 120=5days, 144=6days, 168=1week, 192=8days, 240=10days, 336=2weeks

NUM_EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001

FEATURE_COLS = [
    'Hour', 'day_nr', 'week_nr', 'year', 'month',
    'day_of_year_sin', 'day_of_year_cos',
    'wind_forecast_dah_mw', 'consumption_forecast_dah_mw',
    'temp_forecast_dah_celcius', 'temp_norm_celcius',
    'heating_demand_interaction', 'temp_deviation',
    'spot_lag1'
]
TARGET_COL = 'spot'

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'energy_data.csv')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')


# =============================================================================
# Setup
# =============================================================================

def setup_logging(timestamp: str) -> logging.Logger:
    """Setup logging to both file and console."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file = os.path.join(RESULTS_DIR, f'seq_len_search_{timestamp}.log')

    logger = logging.getLogger('seq_len_search')
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_device() -> torch.device:
    """Get the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


# =============================================================================
# Walk-Forward CV Folds
# =============================================================================

def create_walk_forward_folds(df: pd.DataFrame) -> list:
    """
    Create monthly walk-forward CV folds.

    Strategy:
    - Minimum 4 months of training data
    - Test on each subsequent month
    - Results in 6 folds for the ~10 month training period
    """
    df = df.copy()
    df['month_year'] = df['date_time'].dt.to_period('M')
    unique_months = df['month_year'].unique()

    min_train_months = 4
    folds = []

    for i in range(min_train_months, len(unique_months)):
        train_months = unique_months[:i]
        test_month = unique_months[i]

        train_mask = df['month_year'].isin(train_months)
        test_mask = df['month_year'] == test_month

        folds.append({
            'train_idx': df[train_mask].index.tolist(),
            'test_idx': df[test_mask].index.tolist(),
            'train_months': [str(m) for m in train_months],
            'test_month': str(test_month)
        })

    return folds


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(predictions: np.ndarray, targets: np.ndarray,
                    scaler_y: StandardScaler) -> dict:
    """
    Compute MSE, MAE, RMSE in original EUR/MWh scale.

    Args:
        predictions: Scaled predictions from model
        targets: Scaled true values
        scaler_y: Fitted scaler for inverse transform

    Returns:
        Dict with 'mse', 'mae', 'rmse' keys
    """
    # Inverse transform to original scale
    pred_orig = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    true_orig = scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()

    mse = mean_squared_error(true_orig, pred_orig)
    mae = mean_absolute_error(true_orig, pred_orig)
    rmse = np.sqrt(mse)

    return {'mse': mse, 'mae': mae, 'rmse': rmse}


# =============================================================================
# CV Training Loop
# =============================================================================

def run_cv_for_seq_len(seq_len: int, df: pd.DataFrame, folds: list,
                        device: torch.device, logger: logging.Logger) -> pd.DataFrame:
    """
    Run full walk-forward CV for one sequence length.

    Args:
        seq_len: Sequence length to test
        df: Full training DataFrame
        folds: List of fold dictionaries
        device: torch device (cuda/cpu)
        logger: Logger instance

    Returns:
        DataFrame with columns [seq_len, fold, epoch, mse, mae, rmse]
    """
    results = []

    for fold_idx, fold in enumerate(folds):
        logger.info(f"  Fold {fold_idx + 1}/{len(folds)}: "
                   f"Train {fold['train_months'][0]}-{fold['train_months'][-1]} → "
                   f"Test {fold['test_month']}")

        # Get fold data
        fold_train = df.loc[fold['train_idx']]
        fold_test = df.loc[fold['test_idx']]

        # Check if we have enough data for this sequence length
        if len(fold_train) <= seq_len or len(fold_test) <= seq_len:
            logger.warning(f"  Skipping fold {fold_idx + 1}: insufficient data for seq_len={seq_len}")
            continue

        # Scale features (fit on fold's training data only)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        train_features = scaler_X.fit_transform(fold_train[FEATURE_COLS])
        train_targets = scaler_y.fit_transform(fold_train[[TARGET_COL]])

        test_features = scaler_X.transform(fold_test[FEATURE_COLS])
        test_targets = scaler_y.transform(fold_test[[TARGET_COL]])

        # Create datasets and loaders
        train_dataset = EnergyPriceDataset(train_features, train_targets, seq_len)
        test_dataset = EnergyPriceDataset(test_features, test_targets, seq_len)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model
        model = LSTM_model(input_size=len(FEATURE_COLS)).to(device)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Train for NUM_EPOCHS, recording metrics each epoch
        for epoch in range(NUM_EPOCHS):
            # Training
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = loss_func(predictions, y_batch)
                loss.backward()
                optimizer.step()

            # Evaluate on fold's test set
            model.eval()
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    pred = model(X_batch)
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(y_batch.numpy())

            # Compute metrics in original scale
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            metrics = compute_metrics(all_preds, all_targets, scaler_y)

            results.append({
                'seq_len': seq_len,
                'fold': fold_idx + 1,
                'epoch': epoch + 1,
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse']
            })

            # Log progress every 25 epochs
            if (epoch + 1) % 25 == 0:
                logger.info(f"    Epoch {epoch + 1}: MSE={metrics['mse']:.2f}, "
                           f"MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")

    return pd.DataFrame(results)


# =============================================================================
# Main
# =============================================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logging(timestamp)

    logger.info("=" * 60)
    logger.info("Sequence Length Hyperparameter Search")
    logger.info("=" * 60)
    logger.info(f"Sequence lengths to test: {SEQUENCE_LENGTHS}")
    logger.info(f"Epochs per fold: {NUM_EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LEARNING_RATE}")

    # Setup device
    device = get_device()
    logger.info(f"Device: {device}")

    # Load data
    logger.info(f"Loading data from {DATA_PATH}...")
    df = load_and_preprocess_energy_data(DATA_PATH, keep_date=True)

    # 80/20 split - keep 20% for EQCI calibration (same as notebook)
    split_idx = int(len(df) * 0.8)
    df_train_full = df.iloc[:split_idx].copy()

    logger.info(f"Training data: {len(df_train_full)} samples")
    logger.info(f"Date range: {df_train_full['date_time'].min()} to {df_train_full['date_time'].max()}")

    # Create walk-forward folds
    folds = create_walk_forward_folds(df_train_full)
    logger.info(f"Created {len(folds)} walk-forward CV folds")

    # Run CV for each sequence length
    all_results = []

    for seq_len in SEQUENCE_LENGTHS:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Testing sequence length: {seq_len} hours ({seq_len/24:.1f} days)")
        logger.info("=" * 60)

        results_df = run_cv_for_seq_len(seq_len, df_train_full, folds, device, logger)
        all_results.append(results_df)

        # Quick summary for this seq_len
        if not results_df.empty:
            avg_by_epoch = results_df.groupby('epoch')['mse'].mean()
            best_epoch = avg_by_epoch.idxmin()
            best_mse = avg_by_epoch.min()
            logger.info(f"  Best avg MSE: {best_mse:.2f} at epoch {best_epoch}")

    # Combine all results
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Save full results to CSV
    csv_path = os.path.join(RESULTS_DIR, f'seq_len_search_results_{timestamp}.csv')
    all_results_df.to_csv(csv_path, index=False)
    logger.info(f"\nFull results saved to: {csv_path}")

    # Generate summary (best epoch per seq_len based on avg MSE across folds)
    summary = {}
    for seq_len in SEQUENCE_LENGTHS:
        seq_data = all_results_df[all_results_df['seq_len'] == seq_len]
        if seq_data.empty:
            continue

        avg_by_epoch = seq_data.groupby('epoch').agg({
            'mse': 'mean',
            'mae': 'mean',
            'rmse': 'mean'
        })
        best_epoch = int(avg_by_epoch['mse'].idxmin())
        best_metrics = avg_by_epoch.loc[best_epoch]

        summary[str(seq_len)] = {
            'best_epoch': best_epoch,
            'mse': round(best_metrics['mse'], 4),
            'mae': round(best_metrics['mae'], 4),
            'rmse': round(best_metrics['rmse'], 4)
        }

    # Save summary to JSON
    json_path = os.path.join(RESULTS_DIR, f'seq_len_search_summary_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {json_path}")

    # Print final summary table
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Seq Len':>10} {'Days':>8} {'Best Epoch':>12} {'MSE':>12} {'MAE':>10} {'RMSE':>10}")
    logger.info("-" * 60)

    best_seq_len = None
    best_mse = float('inf')

    for seq_len_str, metrics in summary.items():
        seq_len = int(seq_len_str)
        days = seq_len / 24
        logger.info(f"{seq_len:>10} {days:>8.1f} {metrics['best_epoch']:>12} "
                   f"{metrics['mse']:>12.2f} {metrics['mae']:>10.2f} {metrics['rmse']:>10.2f}")

        if metrics['mse'] < best_mse:
            best_mse = metrics['mse']
            best_seq_len = seq_len

    logger.info("-" * 60)
    logger.info(f"\nBest sequence length: {best_seq_len} hours ({best_seq_len/24:.1f} days)")
    logger.info(f"Best MSE: {best_mse:.2f}")
    logger.info("")
    logger.info("Search complete!")


if __name__ == '__main__':
    main()
