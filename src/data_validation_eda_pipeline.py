"""
Credit Card Fraud Detection - Data Validation & EDA Pipeline
Production-level script for comprehensive data quality assessment and exploratory analysis
"""

import os
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for production
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for the data validation and EDA pipeline"""
    
    # File paths
    RAW_DATA_PATH = r"D:\fraud-detection-real-time\data\raw\creditcard.csv"
    OUTPUT_DIR = r"D:\fraud-detection-real-time\data\data_logging"
    
    # Output subdirectories
    VALIDATION_DIR = "validation_results"
    EDA_DIR = "eda_results"
    VIZ_DIR = "visualizations"
    
    # Data quality thresholds
    MISSING_THRESHOLD = 0.2  # 20% missing values threshold
    OUTLIER_STD_THRESHOLD = 3  # Z-score threshold for outliers
    CARDINALITY_THRESHOLD = 50  # Max unique values for categorical consideration
    CORRELATION_THRESHOLD = 0.95  # High correlation threshold
    
    # Visualization settings
    FIGURE_DPI = 150
    FIGURE_SIZE = (12, 8)
    STYLE = 'seaborn-v0_8-darkgrid'
    
    # Random state for reproducibility
    RANDOM_STATE = 42
    
    # Logging configuration
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Set up logging configuration"""
    logger = logging.getLogger('DataValidationEDA')
    logger.setLevel(config.LOG_LEVEL)
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config.LOG_LEVEL)
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    
    return logger

# ============================================================================
# DATA LOADING
# ============================================================================

class DataLoader:
    """Handle data loading operations"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the credit card dataset
        Returns:
            pd.DataFrame: Loaded dataset
        """
        self.logger.info(f"Loading data from: {self.config.RAW_DATA_PATH}")
        
        try:
            if not os.path.exists(self.config.RAW_DATA_PATH):
                raise FileNotFoundError(f"File not found: {self.config.RAW_DATA_PATH}")
            
            # Load data with appropriate parameters
            df = pd.read_csv(
                self.config.RAW_DATA_PATH,
                low_memory=False,
                na_values=['', ' ', 'NA', 'N/A', 'null', 'NULL', 'None']
            )
            
            self.logger.info(f"Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            self.logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

# ============================================================================
# DATA VALIDATION & QUALITY ASSESSMENT
# ============================================================================

class DataValidator:
    """Comprehensive data validation and quality assessment"""
    
    def __init__(self, df: pd.DataFrame, config: Config, logger: logging.Logger):
        self.df = df
        self.config = config
        self.logger = logger
        self.validation_results = {}
        self.quality_issues = []
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Execute all validation checks"""
        self.logger.info("Starting comprehensive data validation...")
        
        try:
            self.check_basic_info()
            self.check_missing_values()
            self.check_duplicates()
            self.check_data_types()
            self.check_outliers()
            self.check_cardinality()
            self.check_constant_features()
            self.check_data_correlations()
            self.check_class_balance()
            self.check_statistical_properties()
            
            self.logger.info("Data validation completed successfully")
            return self.validation_results
            
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            traceback.print_exc()
            raise
    
    def check_basic_info(self):
        """Basic dataset information"""
        self.validation_results['basic_info'] = {
            'dataset_shape': self.df.shape,
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'column_names': self.df.columns.tolist(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        self.logger.info(f"Basic Info - Shape: {self.df.shape}, Memory: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    def check_missing_values(self):
        """Check for missing values in each column"""
        missing_stats = pd.DataFrame({
            'column': self.df.columns,
            'missing_count': self.df.isnull().sum().values,
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).values,
            'dtype': self.df.dtypes.values
        })
        
        missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values(
            'missing_percentage', ascending=False
        )
        
        # Identify problematic columns
        high_missing_columns = missing_stats[
            missing_stats['missing_percentage'] > (self.config.MISSING_THRESHOLD * 100)
        ]['column'].tolist()
        
        self.validation_results['missing_values'] = {
            'columns_with_missing': len(missing_stats),
            'high_missing_columns': high_missing_columns,
            'missing_details': missing_stats.to_dict('records'),
            'total_missing_values': self.df.isnull().sum().sum()
        }
        
        if high_missing_columns:
            self.quality_issues.append(
                f"HIGH_MISSING_VALUES: Columns {high_missing_columns} have >{self.config.MISSING_THRESHOLD*100}% missing values"
            )
            self.logger.warning(f"Columns with high missing values: {high_missing_columns}")
    
    def check_duplicates(self):
        """Check for duplicate rows"""
        duplicate_count = self.df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(self.df)) * 100
        
        self.validation_results['duplicates'] = {
            'duplicate_rows': duplicate_count,
            'duplicate_percentage': duplicate_percentage,
            'has_duplicates': duplicate_count > 0
        }
        
        if duplicate_count > 0:
            self.quality_issues.append(
                f"DUPLICATE_ROWS: {duplicate_count} duplicate rows found ({duplicate_percentage:.2f}%)"
            )
            self.logger.warning(f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.2f}%)")
    
    def check_data_types(self):
        """Validate data types and identify potential issues"""
        dtype_analysis = {}
        
        for col in self.df.columns:
            current_dtype = str(self.df[col].dtype)
            
            # Check if numeric columns contain non-numeric values
            if self.df[col].dtype in ['int64', 'float64']:
                # Check for infinite values
                inf_count = np.isinf(self.df[col]).sum() if self.df[col].dtype == 'float64' else 0
                
                dtype_analysis[col] = {
                    'dtype': current_dtype,
                    'is_numeric': True,
                    'has_inf': inf_count > 0,
                    'inf_count': inf_count,
                    'min_value': self.df[col].min() if not self.df[col].isnull().all() else None,
                    'max_value': self.df[col].max() if not self.df[col].isnull().all() else None
                }
            else:
                dtype_analysis[col] = {
                    'dtype': current_dtype,
                    'is_numeric': False,
                    'unique_values': self.df[col].nunique()
                }
        
        self.validation_results['data_types'] = dtype_analysis
    
    def check_outliers(self):
        """Detect outliers using IQR and Z-score methods"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_analysis = {}
        
        for col in numeric_cols:
            if col == 'Class':  # Skip target variable
                continue
            
            # IQR method
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = self.df[
                (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            ][col].count()
            
            # Z-score method
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            zscore_outliers = np.sum(z_scores > self.config.OUTLIER_STD_THRESHOLD)
            
            outlier_analysis[col] = {
                'iqr_outliers': int(iqr_outliers),
                'iqr_outlier_percentage': (iqr_outliers / len(self.df)) * 100,
                'zscore_outliers': int(zscore_outliers),
                'zscore_outlier_percentage': (zscore_outliers / len(self.df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        # Identify highly problematic columns
        high_outlier_cols = [
            col for col, stats in outlier_analysis.items() 
            if stats['iqr_outlier_percentage'] > 10
        ]
        
        self.validation_results['outliers'] = outlier_analysis
        
        if high_outlier_cols:
            self.quality_issues.append(
                f"HIGH_OUTLIERS: Columns {high_outlier_cols} have >10% outliers"
            )
            self.logger.warning(f"Columns with high outliers: {high_outlier_cols}")
    
    def check_cardinality(self):
        """Check cardinality of features"""
        cardinality_analysis = {}
        
        for col in self.df.columns:
            n_unique = self.df[col].nunique()
            cardinality_ratio = n_unique / len(self.df)
            
            cardinality_analysis[col] = {
                'unique_values': n_unique,
                'cardinality_ratio': cardinality_ratio,
                'is_high_cardinality': n_unique > self.config.CARDINALITY_THRESHOLD,
                'is_potential_id': cardinality_ratio > 0.9
            }
        
        # Identify potential ID columns
        id_columns = [
            col for col, stats in cardinality_analysis.items() 
            if stats['is_potential_id']
        ]
        
        self.validation_results['cardinality'] = cardinality_analysis
        
        if id_columns:
            self.quality_issues.append(
                f"POTENTIAL_ID_COLUMNS: {id_columns} might be ID columns (cardinality > 90%)"
            )
            self.logger.warning(f"Potential ID columns detected: {id_columns}")
    
    def check_constant_features(self):
        """Check for constant or quasi-constant features"""
        constant_analysis = {}
        
        for col in self.df.columns:
            value_counts = self.df[col].value_counts()
            dominant_ratio = value_counts.iloc[0] / len(self.df) if len(value_counts) > 0 else 0
            
            constant_analysis[col] = {
                'is_constant': self.df[col].nunique() == 1,
                'is_quasi_constant': dominant_ratio > 0.99,
                'dominant_value': value_counts.index[0] if len(value_counts) > 0 else None,
                'dominant_ratio': dominant_ratio
            }
        
        constant_cols = [col for col, stats in constant_analysis.items() if stats['is_constant']]
        quasi_constant_cols = [col for col, stats in constant_analysis.items() if stats['is_quasi_constant']]
        
        self.validation_results['constant_features'] = constant_analysis
        
        if constant_cols:
            self.quality_issues.append(f"CONSTANT_FEATURES: {constant_cols} have constant values")
        if quasi_constant_cols:
            self.quality_issues.append(f"QUASI_CONSTANT_FEATURES: {quasi_constant_cols} have >99% same value")
    
    def check_data_correlations(self):
        """Check for highly correlated features"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = self.df[numeric_cols].corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > self.config.CORRELATION_THRESHOLD:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            self.validation_results['correlations'] = {
                'correlation_matrix': corr_matrix.to_dict(),
                'high_correlation_pairs': high_corr_pairs,
                'num_high_corr_pairs': len(high_corr_pairs)
            }
            
            if high_corr_pairs:
                self.quality_issues.append(
                    f"HIGH_CORRELATIONS: {len(high_corr_pairs)} feature pairs with correlation >{self.config.CORRELATION_THRESHOLD}"
                )
        else:
            self.validation_results['correlations'] = {
                'error': 'Not enough numeric columns for correlation analysis'
            }
    
    def check_class_balance(self):
        """Check class balance for fraud detection"""
        if 'Class' in self.df.columns:
            class_dist = self.df['Class'].value_counts()
            class_percentages = (class_dist / len(self.df)) * 100
            
            self.validation_results['class_balance'] = {
                'class_distribution': class_dist.to_dict(),
                'class_percentages': class_percentages.to_dict(),
                'is_imbalanced': True if min(class_percentages) < 1 else False,
                'minority_class_ratio': min(class_percentages)
            }
            
            if min(class_percentages) < 1:
                self.quality_issues.append(
                    f"CLASS_IMBALANCE: Minority class represents only {min(class_percentages):.3f}% of data"
                )
                self.logger.warning(f"Severe class imbalance detected: {min(class_percentages):.3f}%")
    
    def check_statistical_properties(self):
        """Compute basic statistical properties"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        stats_summary = {}
        for col in numeric_cols:
            if col != 'Class':
                stats_summary[col] = {
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'skewness': self.df[col].skew(),
                    'kurtosis': self.df[col].kurtosis(),
                    'normality_test': self._test_normality(self.df[col].dropna())
                }
        
        self.validation_results['statistical_properties'] = stats_summary
    
    def _test_normality(self, data: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """Test for normality using Shapiro-Wilk test (sample limited)"""
        if len(data) > 5000:
            data = data.sample(5000, random_state=self.config.RANDOM_STATE)
        
        try:
            statistic, p_value = stats.shapiro(data)
            return {
                'test': 'Shapiro-Wilk',
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': p_value > alpha
            }
        except:
            return {'test': 'Shapiro-Wilk', 'error': 'Test could not be performed'}

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

class ExploratoryDataAnalyzer:
    """Comprehensive exploratory data analysis"""
    
    def __init__(self, df: pd.DataFrame, config: Config, logger: logging.Logger, output_dir: Path):
        self.df = df
        self.config = config
        self.logger = logger
        self.output_dir = output_dir
        self.eda_results = {}
        
        # Set style for all plots
        plt.style.use(self.config.STYLE)
    
    def run_all_analyses(self):
        """Execute all EDA analyses"""
        self.logger.info("Starting exploratory data analysis...")
        
        try:
            # Create visualizations directory
            viz_dir = os.path.join(self.output_dir, self.config.VIZ_DIR)
            os.makedirs(viz_dir, exist_ok=True)
            
            # Generate all analyses
            self.analyze_basic_statistics()
            self.analyze_distributions(viz_dir)
            self.analyze_correlations(viz_dir)
            self.analyze_class_distribution(viz_dir)
            self.analyze_time_features(viz_dir)
            self.analyze_amount_distribution(viz_dir)
            self.analyze_feature_relationships(viz_dir)
            self.perform_pca_analysis(viz_dir)
            
            self.logger.info("EDA completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during EDA: {str(e)}")
            traceback.print_exc()
            raise
    
    def analyze_basic_statistics(self):
        """Calculate comprehensive basic statistics"""
        # Numeric columns statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        basic_stats = {}
        for col in numeric_cols:
            basic_stats[col] = {
                'count': int(self.df[col].count()),
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                '25%': float(self.df[col].quantile(0.25)),
                '50%': float(self.df[col].median()),
                '75%': float(self.df[col].quantile(0.75)),
                'max': float(self.df[col].max()),
                'variance': float(self.df[col].var()),
                'range': float(self.df[col].max() - self.df[col].min())
            }
        
        self.eda_results['basic_statistics'] = basic_stats
    
    def analyze_distributions(self, viz_dir: str):
        """Analyze and visualize feature distributions"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Exclude 'Class' and 'Time' for distribution plots
        plot_cols = [col for col in numeric_cols if col not in ['Class', 'Time']]
        
        if plot_cols:
            # Sample of features for distribution plots
            sample_cols = plot_cols[:min(20, len(plot_cols))]
            
            n_cols = 5
            n_rows = (len(sample_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
            axes = axes.flatten()
            
            for idx, col in enumerate(sample_cols):
                ax = axes[idx]
                data = self.df[col].dropna()
                ax.hist(data, bins=50, alpha=0.7, edgecolor='black', density=True)
                ax.set_title(f'{col}\nμ={data.mean():.2f}, σ={data.std():.2f}', fontsize=10)
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.tick_params(labelsize=8)
            
            # Hide unused subplots
            for idx in range(len(sample_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(viz_dir, 'feature_distributions.png'),
                dpi=self.config.FIGURE_DPI,
                bbox_inches='tight'
            )
            plt.close()
            
            self.logger.info("Feature distributions plotted")
    
    def analyze_correlations(self, viz_dir: str):
        """Analyze and visualize correlations"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # Correlation with Class
            if 'Class' in numeric_cols:
                correlations = self.df[numeric_cols].corr()['Class'].sort_values(ascending=False)
                
                # Plot top correlations
                top_corr = correlations.drop('Class').head(20)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = ['red' if c < 0 else 'blue' for c in top_corr.values]
                top_corr.abs().sort_values().plot(kind='barh', color=colors, ax=ax)
                ax.set_title('Top 20 Features Correlated with Fraud (Class)', fontsize=14)
                ax.set_xlabel('Correlation Strength', fontsize=12)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(viz_dir, 'class_correlations.png'),
                    dpi=self.config.FIGURE_DPI,
                    bbox_inches='tight'
                )
                plt.close()
            
            # Full correlation heatmap
            sample_cols = numeric_cols[:min(30, len(numeric_cols))]
            corr_matrix = self.df[sample_cols].corr()
            
            fig, ax = plt.subplots(figsize=(16, 12))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix, mask=mask, cmap='coolwarm', center=0,
                annot=False, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}
            )
            ax.set_title('Feature Correlation Heatmap', fontsize=16)
            plt.tight_layout()
            plt.savefig(
                os.path.join(viz_dir, 'correlation_heatmap.png'),
                dpi=self.config.FIGURE_DPI,
                bbox_inches='tight'
            )
            plt.close()
            
            self.logger.info("Correlation analysis completed")
            
            # Store top correlations in results
            if 'Class' in numeric_cols:
                self.eda_results['top_features_correlated_with_class'] = correlations.to_dict()
    
    def analyze_class_distribution(self, viz_dir: str):
        """Analyze class (fraud) distribution"""
        if 'Class' in self.df.columns:
            class_counts = self.df['Class'].value_counts()
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Bar plot
            colors = ['#2ecc71', '#e74c3c']
            axes[0].bar(class_counts.index.astype(str), class_counts.values, color=colors)
            axes[0].set_title('Transaction Class Distribution', fontsize=14)
            axes[0].set_xlabel('Class (0: Legitimate, 1: Fraudulent)')
            axes[0].set_ylabel('Count')
            
            # Add value labels
            for i, v in enumerate(class_counts.values):
                axes[0].text(i, v + 1000, str(v), ha='center', fontweight='bold')
            
            # Pie chart
            plt.rcParams['font.size'] = 12
            wedges, texts, autotexts = axes[1].pie(
                class_counts.values,
                labels=['Legitimate', 'Fraudulent'],
                autopct='%1.2f%%',
                colors=colors,
                startangle=90,
                explode=(0, 0.1)
            )
            autotexts[0].set_color('white')
            autotexts[1].set_color('white')
            axes[1].set_title('Transaction Class Proportion', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(viz_dir, 'class_distribution.png'),
                dpi=self.config.FIGURE_DPI,
                bbox_inches='tight'
            )
            plt.close()
            
            self.eda_results['class_distribution_analysis'] = {
                'legitimate_count': int(class_counts.get(0, 0)),
                'fraudulent_count': int(class_counts.get(1, 0)),
                'fraud_ratio': float(class_counts.get(1, 0) / len(self.df) * 100)
            }
    
    def analyze_time_features(self, viz_dir: str):
        """Analyze time-related features"""
        if 'Time' in self.df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            
            # Time distribution
            axes[0, 0].hist(self.df['Time'], bins=100, edgecolor='black', alpha=0.7)
            axes[0, 0].set_title('Transaction Time Distribution', fontsize=12)
            axes[0, 0].set_xlabel('Time (seconds)')
            axes[0, 0].set_ylabel('Frequency')
            
            # Fraud vs Legitimate over time
            fraud_data = self.df[self.df['Class'] == 1]
            legit_data = self.df[self.df['Class'] == 0]
            
            axes[0, 1].hist(legit_data['Time'], bins=100, alpha=0.6, label='Legitimate', color='green')
            axes[0, 1].hist(fraud_data['Time'], bins=100, alpha=0.6, label='Fraudulent', color='red')
            axes[0, 1].set_title('Transaction Time by Class', fontsize=12)
            axes[0, 1].set_xlabel('Time (seconds)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            
            # Fraud rate over time (by hour)
            self.df['Hour'] = self.df['Time'] // 3600
            hourly_fraud_rate = self.df.groupby('Hour')['Class'].mean() * 100
            
            axes[1, 0].plot(hourly_fraud_rate.index, hourly_fraud_rate.values, marker='o', linewidth=2)
            axes[1, 0].set_title('Fraud Rate by Hour', fontsize=12)
            axes[1, 0].set_xlabel('Hour')
            axes[1, 0].set_ylabel('Fraud Rate (%)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Transaction volume by hour
            hourly_volume = self.df.groupby('Hour').size()
            axes[1, 1].bar(hourly_volume.index, hourly_volume.values, edgecolor='black', alpha=0.7)
            axes[1, 1].set_title('Transaction Volume by Hour', fontsize=12)
            axes[1, 1].set_xlabel('Hour')
            axes[1, 1].set_ylabel('Number of Transactions')
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(viz_dir, 'time_analysis.png'),
                dpi=self.config.FIGURE_DPI,
                bbox_inches='tight'
            )
            plt.close()
            
            self.logger.info("Time feature analysis completed")
    
    def analyze_amount_distribution(self, viz_dir: str):
        """Analyze transaction amount distribution"""
        if 'Amount' in self.df.columns:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Full distribution
            axes[0, 0].hist(self.df['Amount'], bins=100, edgecolor='black', alpha=0.7)
            axes[0, 0].set_title('Full Amount Distribution', fontsize=12)
            axes[0, 0].set_xlabel('Amount')
            axes[0, 0].set_ylabel('Frequency')
            
            # Log-transformed distribution
            log_amount = np.log1p(self.df['Amount'])
            axes[0, 1].hist(log_amount, bins=100, edgecolor='black', alpha=0.7, color='orange')
            axes[0, 1].set_title('Log-Transformed Amount Distribution', fontsize=12)
            axes[0, 1].set_xlabel('Log(Amount + 1)')
            axes[0, 1].set_ylabel('Frequency')
            
            # Amount by class
            fraud_amount = self.df[self.df['Class'] == 1]['Amount']
            legit_amount = self.df[self.df['Class'] == 0]['Amount']
            
            axes[0, 2].boxplot([legit_amount, fraud_amount], labels=['Legitimate', 'Fraudulent'])
            axes[0, 2].set_title('Amount Distribution by Class', fontsize=12)
            axes[0, 2].set_ylabel('Amount')
            
            # Amount statistics by class
            amount_stats = self.df.groupby('Class')['Amount'].agg(['mean', 'median', 'std', 'min', 'max'])
            axes[1, 0].axis('tight')
            axes[1, 0].axis('off')
            table = axes[1, 0].table(
                cellText=amount_stats.round(2).values,
                colLabels=amount_stats.columns,
                rowLabels=['Legitimate', 'Fraudulent'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            axes[1, 0].set_title('Amount Statistics by Class', fontsize=12, pad=20)
            
            # Percentile analysis
            percentiles = [50, 75, 90, 95, 99]
            legit_percentiles = [np.percentile(legit_amount, p) for p in percentiles]
            fraud_percentiles = [np.percentile(fraud_amount, p) for p in percentiles]
            
            x = np.arange(len(percentiles))
            width = 0.35
            axes[1, 1].bar(x - width/2, legit_percentiles, width, label='Legitimate', alpha=0.8)
            axes[1, 1].bar(x + width/2, fraud_percentiles, width, label='Fraudulent', alpha=0.8)
            axes[1, 1].set_xlabel('Percentile')
            axes[1, 1].set_ylabel('Amount')
            axes[1, 1].set_title('Amount Percentiles Comparison', fontsize=12)
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(percentiles)
            axes[1, 1].legend()
            
            # Scatter plot of Amount vs Time
            sample_idx = np.random.choice(len(self.df), min(10000, len(self.df)), replace=False)
            sample_df = self.df.iloc[sample_idx]
            
            axes[1, 2].scatter(
                sample_df[sample_df['Class'] == 0]['Time'],
                sample_df[sample_df['Class'] == 0]['Amount'],
                alpha=0.3, s=1, color='green', label='Legitimate'
            )
            axes[1, 2].scatter(
                sample_df[sample_df['Class'] == 1]['Time'],
                sample_df[sample_df['Class'] == 1]['Amount'],
                alpha=0.5, s=10, color='red', label='Fraudulent'
            )
            axes[1, 2].set_xlabel('Time (seconds)')
            axes[1, 2].set_ylabel('Amount')
            axes[1, 2].set_title('Amount vs Time (Sample)', fontsize=12)
            axes[1, 2].legend()
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(viz_dir, 'amount_analysis.png'),
                dpi=self.config.FIGURE_DPI,
                bbox_inches='tight'
            )
            plt.close()
            
            self.logger.info("Amount analysis completed")
    
    def analyze_feature_relationships(self, viz_dir: str):
        """Analyze relationships between features and target"""
        if 'Class' in self.df.columns:
            # Get top features correlated with Class
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            correlations = self.df[numeric_cols].corr()['Class'].abs().sort_values(ascending=False)
            top_features = correlations.drop('Class').head(6).index.tolist()
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for idx, feature in enumerate(top_features):
                ax = axes[idx]
                
                fraud_vals = self.df[self.df['Class'] == 1][feature]
                legit_vals = self.df[self.df['Class'] == 0][feature]
                
                ax.hist(legit_vals, bins=50, alpha=0.6, label='Legitimate', density=True, color='green')
                ax.hist(fraud_vals, bins=50, alpha=0.6, label='Fraudulent', density=True, color='red')
                ax.set_title(f'{feature} Distribution by Class\nCorr: {correlations[feature]:.4f}', fontsize=12)
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(viz_dir, 'feature_class_relationships.png'),
                dpi=self.config.FIGURE_DPI,
                bbox_inches='tight'
            )
            plt.close()
            
            self.logger.info("Feature relationships analyzed")
    
    def perform_pca_analysis(self, viz_dir: str):
        """Perform PCA analysis for dimensionality visualization"""
        # Select features (exclude Time, Amount, and Class)
        pca_features = self.df.select_dtypes(include=[np.number]).columns
        pca_features = [col for col in pca_features if col not in ['Time', 'Amount', 'Class']]
    
        if len(pca_features) > 1:
            # Sample if dataset is large
            if len(self.df) > 50000:
                sample_df = self.df.sample(50000, random_state=self.config.RANDOM_STATE)
            else:
                sample_df = self.df
        
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(sample_df[pca_features])
        
            # PCA - Fixed initialization
            pca = PCA(n_components=2, random_state=self.config.RANDOM_STATE)
            X_pca = pca.fit_transform(X_scaled)
        
            # Plot
            fig, ax = plt.subplots(figsize=(12, 10))
        
            scatter = ax.scatter(
                X_pca[:, 0],
                X_pca[:, 1],
                c=sample_df['Class'].values,
                cmap='coolwarm',
                alpha=0.5,
                s=1
            )
        
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
            ax.set_title('PCA Visualization of Transactions', fontsize=14)
        
            legend = ax.legend(*scatter.legend_elements(), title="Class")
            ax.add_artist(legend)
        
            plt.tight_layout()
            plt.savefig(
                os.path.join(viz_dir, 'pca_visualization.png'),
                dpi=self.config.FIGURE_DPI,
                bbox_inches='tight'
            )
            plt.close()
        
            self.eda_results['pca_analysis'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_variance_explained': float(sum(pca.explained_variance_ratio_))
            }
        
            self.logger.info("PCA analysis completed")

# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generate comprehensive data validation and EDA reports"""
    
    def __init__(self, config: Config, logger: logging.Logger, output_dir: Path):
        
        self.config = config
        self.logger = logger
        self.output_dir = output_dir
    
    def generate_validation_report(
        self,
        validation_results: Dict[str, Any],
        quality_issues: List[str]
    ):
        """Generate data validation report"""
        self.logger.info("Generating data validation report...")
        
        report_path = os.path.join(self.output_dir, self.config.VALIDATION_DIR, 'data_validation_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA VALIDATION & QUALITY ASSESSMENT REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {self.config.RAW_DATA_PATH}\n\n")
            
            # Basic Information
            if 'basic_info' in validation_results:
                f.write("-" * 80 + "\n")
                f.write("BASIC DATASET INFORMATION\n")
                f.write("-" * 80 + "\n")
                info = validation_results['basic_info']
                f.write(f"Total Rows: {info['total_rows']:,}\n")
                f.write(f"Total Columns: {info['total_columns']}\n")
                f.write(f"Memory Usage: {info['memory_usage_mb']:.2f} MB\n\n")
            
            # Missing Values
            if 'missing_values' in validation_results:
                f.write("-" * 80 + "\n")
                f.write("MISSING VALUES ANALYSIS\n")
                f.write("-" * 80 + "\n")
                mv = validation_results['missing_values']
                f.write(f"Total Missing Values: {mv['total_missing_values']:,}\n")
                f.write(f"Columns with Missing Values: {mv['columns_with_missing']}\n\n")
                
                if mv['columns_with_missing'] > 0:
                    f.write("Detailed Missing Values:\n")
                    for detail in mv['missing_details']:
                        f.write(f"  - {detail['column']}: {detail['missing_count']:,} ({detail['missing_percentage']:.2f}%)\n")
                    f.write("\n")
            
            # Duplicates
            if 'duplicates' in validation_results:
                f.write("-" * 80 + "\n")
                f.write("DUPLICATE ANALYSIS\n")
                f.write("-" * 80 + "\n")
                dup = validation_results['duplicates']
                f.write(f"Duplicate Rows: {dup['duplicate_rows']:,} ({dup['duplicate_percentage']:.2f}%)\n\n")
            
            # Outliers
            if 'outliers' in validation_results:
                f.write("-" * 80 + "\n")
                f.write("OUTLIER ANALYSIS (Top 10 Features)\n")
                f.write("-" * 80 + "\n")
                
                outlier_items = sorted(
                    validation_results['outliers'].items(),
                    key=lambda x: x[1]['iqr_outlier_percentage'],
                    reverse=True
                )[:10]
                
                for col, stats in outlier_items:
                    f.write(f"\n{col}:\n")
                    f.write(f"  - IQR Method: {stats['iqr_outliers']:,} outliers ({stats['iqr_outlier_percentage']:.2f}%)\n")
                    f.write(f"  - Z-Score Method: {stats['zscore_outliers']:,} outliers ({stats['zscore_outlier_percentage']:.2f}%)\n")
            
            # Class Balance
            if 'class_balance' in validation_results:
                f.write("\n" + "-" * 80 + "\n")
                f.write("CLASS BALANCE ANALYSIS\n")
                f.write("-" * 80 + "\n")
                cb = validation_results['class_balance']
                for cls, count in cb['class_distribution'].items():
                    percentage = cb['class_percentages'][cls]
                    f.write(f"Class {cls}: {count:,} ({percentage:.3f}%)\n")
                f.write(f"Class Imbalance Detected: {'Yes' if cb['is_imbalanced'] else 'No'}\n")
            
            # Quality Issues Summary
            f.write("\n" + "=" * 80 + "\n")
            f.write("DATA QUALITY ISSUES SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            if quality_issues:
                f.write(f"Total Issues Found: {len(quality_issues)}\n\n")
                for i, issue in enumerate(quality_issues, 1):
                    f.write(f"{i}. {issue}\n")
            else:
                f.write("No significant data quality issues found.\n")
            
            # High Correlations
            if 'correlations' in validation_results and 'high_correlation_pairs' in validation_results['correlations']:
                f.write("\n" + "-" * 80 + "\n")
                f.write("HIGH CORRELATIONS (>0.95)\n")
                f.write("-" * 80 + "\n")
                for pair in validation_results['correlations']['high_correlation_pairs']:
                    f.write(f"  - {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}\n")
        
        self.logger.info(f"Validation report saved to: {report_path}")
    
    def generate_eda_report(self, eda_results: Dict[str, Any]):
        """Generate EDA report"""
        self.logger.info("Generating EDA report...")
        
        report_path = os.path.join(self.output_dir, self.config.EDA_DIR, 'exploratory_data_analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Basic Statistics
            if 'basic_statistics' in eda_results:
                f.write("-" * 80 + "\n")
                f.write("BASIC STATISTICS (Key Features)\n")
                f.write("-" * 80 + "\n\n")
                
                key_features = ['Amount', 'Time'] if 'Amount' in eda_results['basic_statistics'] else []
                key_features.extend([k for k in eda_results['basic_statistics'].keys() 
                                   if k not in key_features][:5])
                
                for feature in key_features[:10]:
                    if feature in eda_results['basic_statistics']:
                        stats = eda_results['basic_statistics'][feature]
                        f.write(f"\n{feature}:\n")
                        f.write(f"  Mean: {stats['mean']:.4f}\n")
                        f.write(f"  Std: {stats['std']:.4f}\n")
                        f.write(f"  Min: {stats['min']:.4f}\n")
                        f.write(f"  Max: {stats['max']:.4f}\n")
                        f.write(f"  Median: {stats['50%']:.4f}\n")
                        f.write(f"  Range: {stats['range']:.4f}\n")
            
            # Class Distribution
            if 'class_distribution_analysis' in eda_results:
                f.write("\n" + "-" * 80 + "\n")
                f.write("CLASS DISTRIBUTION\n")
                f.write("-" * 80 + "\n")
                cd = eda_results['class_distribution_analysis']
                f.write(f"Legitimate Transactions: {cd['legitimate_count']:,}\n")
                f.write(f"Fraudulent Transactions: {cd['fraudulent_count']:,}\n")
                f.write(f"Fraud Ratio: {cd['fraud_ratio']:.4f}%\n")
            
            # PCA Analysis
            if 'pca_analysis' in eda_results:
                f.write("\n" + "-" * 80 + "\n")
                f.write("PCA ANALYSIS\n")
                f.write("-" * 80 + "\n")
                pca = eda_results['pca_analysis']
                f.write(f"Total Variance Explained (2 components): {pca['total_variance_explained']:.4f}\n")
                f.write(f"PC1: {pca['explained_variance_ratio'][0]:.4f}\n")
                f.write(f"PC2: {pca['explained_variance_ratio'][1]:.4f}\n")
            
            # Top Features Correlated with Class
            if 'top_features_correlated_with_class' in eda_results:
                f.write("\n" + "-" * 80 + "\n")
                f.write("TOP FEATURES CORRELATED WITH FRAUD\n")
                f.write("-" * 80 + "\n")
                
                correlations = eda_results['top_features_correlated_with_class']
                top_features = sorted(
                    [(k, v) for k, v in correlations.items() if k != 'Class'],
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:15]
                
                for feature, corr in top_features:
                    f.write(f"  {feature}: {corr:.4f}\n")
        
        self.logger.info(f"EDA report saved to: {report_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class DataValidationEDAPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.df = None
        self.validation_results = None
        self.eda_results = None
        self.quality_issues = []
        
        # Create output directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary output directories"""
        dirs = [
            self.config.OUTPUT_DIR,
            os.path.join(self.config.OUTPUT_DIR, self.config.VALIDATION_DIR),
            os.path.join(self.config.OUTPUT_DIR, self.config.EDA_DIR),
            os.path.join(self.config.OUTPUT_DIR, self.config.VIZ_DIR)
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        self.logger.info(f"Output directories created at: {self.config.OUTPUT_DIR}")
    
    def run(self):
        """Execute the complete pipeline"""
        start_time = datetime.now()
        self.logger.info("=" * 80)
        self.logger.info("STARTING DATA VALIDATION & EDA PIPELINE")
        self.logger.info("=" * 80)
        
        try:
            # Step 1: Load Data
            loader = DataLoader(self.config, self.logger)
            self.df = loader.load_data()
            
            # Step 2: Data Validation
            self.logger.info("\n" + "=" * 40)
            self.logger.info("PHASE 1: Data Validation & Quality Assessment")
            self.logger.info("=" * 40)
            
            validator = DataValidator(self.df, self.config, self.logger)
            self.validation_results = validator.run_all_validations()
            self.quality_issues = validator.quality_issues
            
            # Step 3: Exploratory Data Analysis
            self.logger.info("\n" + "=" * 40)
            self.logger.info("PHASE 2: Exploratory Data Analysis")
            self.logger.info("=" * 40)
            
            output_dir = Path(self.config.OUTPUT_DIR)
            eda_analyzer = ExploratoryDataAnalyzer(
                self.df, self.config, self.logger, output_dir
            )
            eda_analyzer.run_all_analyses()
            self.eda_results = eda_analyzer.eda_results
            
            # Step 4: Generate Reports
            self.logger.info("\n" + "=" * 40)
            self.logger.info("PHASE 3: Report Generation")
            self.logger.info("=" * 40)
            
            report_generator = ReportGenerator(self.config, self.logger, output_dir)
            report_generator.generate_validation_report(
                self.validation_results, self.quality_issues
            )
            report_generator.generate_eda_report(self.eda_results)
            
            # Final Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"Total Duration: {duration:.2f} seconds")
            self.logger.info(f"Output Directory: {self.config.OUTPUT_DIR}")
            self.logger.info(f"Quality Issues Found: {len(self.quality_issues)}")
            
            # Print summary to console
            print("\n" + "=" * 80)
            print("EXECUTION SUMMARY")
            print("=" * 80)
            print(f"Dataset Shape: {self.df.shape}")
            print(f"Quality Issues Found: {len(self.quality_issues)}")
            print(f"Visualizations Generated: {len(os.listdir(os.path.join(self.config.OUTPUT_DIR, self.config.VIZ_DIR)))}")
            print(f"Reports Generated in: {self.config.OUTPUT_DIR}")
            print("-" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            traceback.print_exc()
            return False

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    # Create and run pipeline
    pipeline = DataValidationEDAPipeline(config)
    success = pipeline.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)