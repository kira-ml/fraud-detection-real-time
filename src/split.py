"""
split.py
Credit Card Fraud Detection - Time-Aware Data Splitting

This module implements a chronological split of the 48-hour dataset into
training (first 80%) and test (remaining 20%) sets based on the Time column.
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for data splitting"""
    # Paths
    BASELINE_FEATURES_PATH = r"D:\fraud-detection-real-time\data\processed\processedfeatures_baseline.parquet"
    ADVANCED_FEATURES_PATH = r"D:\fraud-detection-real-time\data\processed\processedfeatures_advanced.parquet"
    OUTPUT_DIR = r"D:\fraud-detection-real-time\src\..\data\processed"
    
    # Split parameters
    TRAIN_RATIO = 0.80
    MAX_TIME_SECONDS = 172792  # From module description (48 hours / 172,800 + small offset)
    
    # Feature column name
    TIME_COLUMN = "Time"  # Note: May not exist in engineered features, will handle this
    
    # Output file names
    TRAIN_OUTPUT = "train_baseline.parquet"
    TEST_OUTPUT = "test_baseline.parquet"
    SPLIT_CONFIG = "split_config.json"

# ============================================================================
# DATA SPLITTER
# ============================================================================

class ChronologicalSplitter:
    """
    Time-aware data splitter that preserves chronological order.
    Splits at 80% of the time range for training and 20% for testing.
    """
    
    def __init__(self, config: Config):
        """Initialize splitter with configuration"""
        self.config = config
        self.train_df = None
        self.test_df = None
        self.split_stats = {}
        
    def load_features(self, file_path: str) -> pd.DataFrame:
        """Load features from parquet file"""
        print(f"Loading features from: {file_path}")
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def compute_split_threshold(self, df: pd.DataFrame) -> float:
        """
        Compute the split timestamp based on 80% of max Time value.
        Falls back to self.config.MAX_TIME_SECONDS if Time column is missing.
        """
        if self.config.TIME_COLUMN in df.columns:
            max_time = df[self.config.TIME_COLUMN].max()
            split_threshold = self.config.TRAIN_RATIO * max_time
            print(f"Using Time column: max={max_time:.0f}, split at {split_threshold:.0f}")
            return split_threshold
        else:
            # Use the known max from module description
            print(f"Time column not found. Using fixed max time: {self.config.MAX_TIME_SECONDS}")
            split_threshold = self.config.TRAIN_RATIO * self.config.MAX_TIME_SECONDS
            print(f"Split threshold: {split_threshold:.0f} seconds")
            return split_threshold
    
    def split_dataframe(self, df: pd.DataFrame, split_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into train and test based on Time threshold.
        """
        # Ensure data is sorted chronologically
        if self.config.TIME_COLUMN in df.columns:
            df = df.sort_values(self.config.TIME_COLUMN).reset_index(drop=True)
        
        # Split based on Time column if available, otherwise use row-based split
        if self.config.TIME_COLUMN in df.columns:
            train_df = df[df[self.config.TIME_COLUMN] <= split_threshold].copy()
            test_df = df[df[self.config.TIME_COLUMN] > split_threshold].copy()
            
            # Verify no temporal leakage
            if len(train_df) > 0 and len(test_df) > 0:
                max_train_time = train_df[self.config.TIME_COLUMN].max()
                min_test_time = test_df[self.config.TIME_COLUMN].min()
                print(f"Max train time: {max_train_time:.0f}, Min test time: {min_test_time:.0f}")
                if max_train_time < min_test_time:
                    print("✅ No temporal leakage: all train times < all test times")
                else:
                    print("⚠️ Temporal leakage detected! Check split logic.")
        else:
            # Fallback: row-based split (80/20) if Time column not present
            split_idx = int(self.config.TRAIN_RATIO * len(df))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            print("⚠️ Time column not found - using row-based 80/20 split (temporal ordering not guaranteed)")
        
        return train_df, test_df
    
    def compute_statistics(self, df: pd.DataFrame, label: str) -> Dict[str, Any]:
        """Compute statistics for a split to verify class distribution"""
        if 'Class' not in df.columns:
            return {"error": "Class column not found in dataframe"}
        
        total = len(df)
        fraud_count = (df['Class'] == 1).sum()
        legit_count = (df['Class'] == 0).sum()
        fraud_ratio = fraud_count / total if total > 0 else 0
        
        stats = {
            f"{label}_total": total,
            f"{label}_fraud": fraud_count,
            f"{label}_legit": legit_count,
            f"{label}_fraud_ratio": fraud_ratio,
            f"{label}_fraud_ratio_pct": fraud_ratio * 100
        }
        
        return stats
    
    def save_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Save train and test splits to parquet files"""
        # Create output directory
        output_dir = Path(self.config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save parquet files
        train_path = output_dir / self.config.TRAIN_OUTPUT
        test_path = output_dir / self.config.TEST_OUTPUT
        
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        print(f"✅ Train set saved to: {train_path}")
        print(f"✅ Test set saved to: {test_path}")
        
        # Save split configuration
        config_path = output_dir / self.config.SPLIT_CONFIG
        with open(config_path, 'w') as f:
            json.dump({
                "split_ratio": self.config.TRAIN_RATIO,
                "split_threshold": float(self.split_stats.get("split_threshold", 0)),
                "train_fraud_ratio": self.split_stats.get("train_fraud_ratio_pct", 0),
                "test_fraud_ratio": self.split_stats.get("test_fraud_ratio_pct", 0),
                "train_size": int(self.split_stats.get("train_total", 0)),
                "test_size": int(self.split_stats.get("test_total", 0))
            }, f, indent=4)
        print(f"✅ Split config saved to: {config_path}")
    
    def run_splitting(self, input_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete data splitting pipeline.
        """
        # Load data
        df = self.load_features(input_path)
        
        # Compute split threshold
        split_threshold = self.compute_split_threshold(df)
        self.split_stats["split_threshold"] = split_threshold
        
        # Perform split
        train_df, test_df = self.split_dataframe(df, split_threshold)
        
        # Compute statistics
        train_stats = self.compute_statistics(train_df, "train")
        test_stats = self.compute_statistics(test_df, "test")
        
        # Merge statistics
        self.split_stats.update(train_stats)
        self.split_stats.update(test_stats)
        
        # Print statistics
        print("\n" + "=" * 50)
        print("SPLIT STATISTICS")
        print("=" * 50)
        print(f"Training set: {train_stats['train_total']:,} rows, "
              f"{train_stats['train_fraud_ratio_pct']:.4f}% fraud")
        print(f"Test set:     {test_stats['test_total']:,} rows, "
              f"{test_stats['test_fraud_ratio_pct']:.4f}% fraud")
        print(f"Total:        {train_stats['train_total'] + test_stats['test_total']:,} rows")
        
        # Check if fraud ratio is preserved
        if 'train_fraud_ratio' in train_stats and 'test_fraud_ratio' in test_stats:
            diff = abs(train_stats['train_fraud_ratio'] - test_stats['test_fraud_ratio'])
            if diff < 0.0005:  # Within 0.05%
                print("✅ Fraud ratio preserved between splits")
            else:
                print(f"⚠️ Fraud ratio difference: {diff:.6f}")
        
        # Save splits
        self.save_splits(train_df, test_df)
        
        return train_df, test_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for data splitting"""
    print("\n" + "#" * 60)
    print("# CREDIT CARD FRAUD DETECTION - DATA SPLITTING")
    print("#" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Initialize splitter
    splitter = ChronologicalSplitter(config)
    
    # Path to baseline features
    baseline_path = config.BASELINE_FEATURES_PATH
    
    # Run splitting for baseline features
    print(f"\n{'=' * 60}")
    print("SPLITTING BASELINE FEATURES")
    print(f"{'=' * 60}")
    train_baseline, test_baseline = splitter.run_splitting(baseline_path)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Train set size: {len(train_baseline):,}")
    print(f"Test set size: {len(test_baseline):,}")
    print(f"Train fraud ratio: {(train_baseline['Class'] == 1).mean() * 100:.4f}%")
    print(f"Test fraud ratio: {(test_baseline['Class'] == 1).mean() * 100:.4f}%")
    print(f"✅ Data splitting complete!")
    
    return train_baseline, test_baseline

if __name__ == "__main__":
    train_data, test_data = main()