"""
Dynamic Resonance Rooting (DRR) Implementation with UCI Human Activity Recognition Dataset

This script downloads, preprocesses, and analyzes the UCI Human Activity Recognition dataset
using the Dynamic Resonance Rooting framework. The dataset contains multivariate time series
data from smartphone accelerometer and gyroscope sensors.

Dataset: Human Activity Recognition Using Smartphones
Source: UCI Machine Learning Repository
Features: 6 raw sensor signals (3-axis accelerometer + 3-axis gyroscope)
Sampling Rate: 50 Hz
Activities: Walking, Walking_Upstairs, Walking_Downstairs, Sitting, Standing, Laying
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve
import zipfile
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class UCIHARDataProcessor:
    """
    Processor for UCI Human Activity Recognition Dataset for DRR Analysis
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
        self.dataset_path = os.path.join(data_dir, 'UCI_HAR_Dataset.zip')
        self.extracted_path = os.path.join(data_dir, 'UCI HAR Dataset')
        self.processed_file = os.path.join(data_dir, 'uci_har_processed.csv')
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Dataset parameters
        self.sampling_rate = 50  # Hz
        self.window_size = 128   # samples per window
        self.activities = {
            1: 'WALKING',
            2: 'WALKING_UPSTAIRS', 
            3: 'WALKING_DOWNSTAIRS',
            4: 'SITTING',
            5: 'STANDING',
            6: 'LAYING'
        }
        
        # Raw sensor columns for DRR analysis
        self.sensor_columns = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]
    
    def download_dataset(self):
        """Download the UCI HAR dataset"""
        if not os.path.exists(self.dataset_path):
            print(f"Downloading UCI HAR Dataset from {self.dataset_url}")
            print("This may take a few minutes...")
            urlretrieve(self.dataset_url, self.dataset_path)
            print(f"Dataset downloaded to {self.dataset_path}")
        else:
            print("Dataset already downloaded")
    
    def extract_dataset(self):
        """Extract the downloaded dataset"""
        if not os.path.exists(self.extracted_path):
            print("Extracting dataset...")
            with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print(f"Dataset extracted to {self.extracted_path}")
        else:
            print("Dataset already extracted")
    
    def load_raw_signals(self, data_type='train'):
        """
        Load raw sensor signals for DRR analysis
        
        Args:
            data_type (str): 'train' or 'test'
            
        Returns:
            tuple: (signals_df, labels_df, subjects_df)
        """
        signals_path = os.path.join(self.extracted_path, data_type, 'Inertial Signals')
        labels_path = os.path.join(self.extracted_path, data_type, f'y_{data_type}.txt')
        subjects_path = os.path.join(self.extracted_path, data_type, f'subject_{data_type}.txt')
        
        # Load labels and subjects
        labels = pd.read_csv(labels_path, header=None, names=['activity_id'])
        subjects = pd.read_csv(subjects_path, header=None, names=['subject_id'])
        
        # Load raw signals
        signal_files = {
            'body_acc_x': f'body_acc_x_{data_type}.txt',
            'body_acc_y': f'body_acc_y_{data_type}.txt', 
            'body_acc_z': f'body_acc_z_{data_type}.txt',
            'body_gyro_x': f'body_gyro_x_{data_type}.txt',
            'body_gyro_y': f'body_gyro_y_{data_type}.txt',
            'body_gyro_z': f'body_gyro_z_{data_type}.txt',
            'total_acc_x': f'total_acc_x_{data_type}.txt',
            'total_acc_y': f'total_acc_y_{data_type}.txt',
            'total_acc_z': f'total_acc_z_{data_type}.txt'
        }
        
        signals_data = {}
        for signal_name, filename in signal_files.items():
            filepath = os.path.join(signals_path, filename)
            # Each row is a 128-sample window
            signal_data = pd.read_csv(filepath, sep=r'\s+', header=None)
            signals_data[signal_name] = signal_data
        
        return signals_data, labels, subjects
    
    def create_continuous_timeseries(self, signals_data, labels, subjects, max_samples=10000):
        """
        Convert windowed data to continuous time series for DRR analysis
        
        Args:
            signals_data (dict): Dictionary of signal DataFrames
            labels (pd.DataFrame): Activity labels
            subjects (pd.DataFrame): Subject IDs
            max_samples (int): Maximum number of samples to include
            
        Returns:
            pd.DataFrame: Continuous time series data
        """
        print("Creating continuous time series for DRR analysis...")
        
        # Initialize lists to store continuous data
        continuous_data = []
        
        # Process each window
        n_windows = min(len(labels), max_samples // self.window_size)
        
        for window_idx in range(n_windows):
            # Get metadata for this window
            activity_id = labels.iloc[window_idx]['activity_id']
            subject_id = subjects.iloc[window_idx]['subject_id']
            activity_name = self.activities[activity_id]
            
            # Extract signals for this window
            window_data = {}
            for signal_name, signal_df in signals_data.items():
                window_data[signal_name] = signal_df.iloc[window_idx].values
            
            # Create time series for this window
            for sample_idx in range(self.window_size):
                sample_data = {
                    'timestamp': window_idx * self.window_size + sample_idx,
                    'time_seconds': (window_idx * self.window_size + sample_idx) / self.sampling_rate,
                    'window_id': window_idx,
                    'subject_id': subject_id,
                    'activity_id': activity_id,
                    'activity_name': activity_name
                }
                
                # Add sensor values
                for signal_name in self.sensor_columns:
                    sample_data[signal_name] = window_data[signal_name][sample_idx]
                
                continuous_data.append(sample_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(continuous_data)
        
        print(f"Created continuous time series with {len(df)} samples")
        print(f"Duration: {df['time_seconds'].max():.2f} seconds")
        print(f"Sampling rate: {self.sampling_rate} Hz")
        
        return df
    
    def preprocess_for_drr(self, df):
        """
        Preprocess the continuous time series for DRR analysis
        
        Args:
            df (pd.DataFrame): Continuous time series data
            
        Returns:
            pd.DataFrame: Preprocessed data ready for DRR
        """
        print("Preprocessing data for DRR analysis...")
        
        # Remove any NaN values
        df_clean = df.dropna()
        
        # Normalize sensor signals to [-1, 1] range for better DRR analysis
        sensor_data = df_clean[self.sensor_columns].copy()
        
        # Apply z-score normalization to each sensor
        for col in self.sensor_columns:
            mean_val = sensor_data[col].mean()
            std_val = sensor_data[col].std()
            df_clean[f'{col}_normalized'] = (sensor_data[col] - mean_val) / std_val
        
        # Create magnitude signals for better resonance detection
        df_clean['acc_magnitude'] = np.sqrt(
            df_clean['body_acc_x']**2 + 
            df_clean['body_acc_y']**2 + 
            df_clean['body_acc_z']**2
        )
        
        df_clean['gyro_magnitude'] = np.sqrt(
            df_clean['body_gyro_x']**2 + 
            df_clean['body_gyro_y']**2 + 
            df_clean['body_gyro_z']**2
        )
        
        # Add derived features for DRR analysis
        df_clean['acc_gyro_correlation'] = (
            df_clean['acc_magnitude'] * df_clean['gyro_magnitude']
        )
        
        print(f"Preprocessed data shape: {df_clean.shape}")
        return df_clean
    
    def save_processed_data(self, df):
        """Save processed data to CSV"""
        df.to_csv(self.processed_file, index=False)
        print(f"Processed data saved to: {self.processed_file}")
        
        # Save summary statistics
        summary_file = self.processed_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("UCI HAR Dataset - DRR Processing Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {len(df)}\n")
            f.write(f"Duration: {df['time_seconds'].max():.2f} seconds\n")
            f.write(f"Sampling Rate: {self.sampling_rate} Hz\n")
            f.write(f"Number of Subjects: {df['subject_id'].nunique()}\n")
            f.write(f"Activities: {', '.join(df['activity_name'].unique())}\n\n")
            
            f.write("Sensor Columns for DRR Analysis:\n")
            for col in self.sensor_columns:
                f.write(f"  - {col}\n")
            
            f.write("\nDerived Features:\n")
            f.write("  - acc_magnitude\n")
            f.write("  - gyro_magnitude\n")
            f.write("  - acc_gyro_correlation\n")
            f.write("  - *_normalized versions of all sensors\n")
        
        print(f"Summary saved to: {summary_file}")
    
    def visualize_data(self, df, n_samples=2000):
        """Create visualizations of the processed data"""
        print("Creating data visualizations...")
        
        # Sample data for visualization
        df_viz = df.head(n_samples).copy()
        
        # Create subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('UCI HAR Dataset - Sensor Data for DRR Analysis', fontsize=16)
        
        # Plot 1: Accelerometer signals
        axes[0, 0].plot(df_viz['time_seconds'], df_viz['body_acc_x'], label='X-axis', alpha=0.7)
        axes[0, 0].plot(df_viz['time_seconds'], df_viz['body_acc_y'], label='Y-axis', alpha=0.7)
        axes[0, 0].plot(df_viz['time_seconds'], df_viz['body_acc_z'], label='Z-axis', alpha=0.7)
        axes[0, 0].set_title('Body Accelerometer Signals')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Acceleration (g)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Gyroscope signals
        axes[0, 1].plot(df_viz['time_seconds'], df_viz['body_gyro_x'], label='X-axis', alpha=0.7)
        axes[0, 1].plot(df_viz['time_seconds'], df_viz['body_gyro_y'], label='Y-axis', alpha=0.7)
        axes[0, 1].plot(df_viz['time_seconds'], df_viz['body_gyro_z'], label='Z-axis', alpha=0.7)
        axes[0, 1].set_title('Body Gyroscope Signals')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Angular Velocity (rad/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Magnitude signals
        axes[1, 0].plot(df_viz['time_seconds'], df_viz['acc_magnitude'], label='Acc Magnitude', alpha=0.8)
        axes[1, 0].plot(df_viz['time_seconds'], df_viz['gyro_magnitude'], label='Gyro Magnitude', alpha=0.8)
        axes[1, 0].set_title('Signal Magnitudes')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Magnitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Activity distribution
        activity_counts = df['activity_name'].value_counts()
        axes[1, 1].bar(activity_counts.index, activity_counts.values)
        axes[1, 1].set_title('Activity Distribution')
        axes[1, 1].set_xlabel('Activity')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 5: Correlation matrix
        corr_cols = ['body_acc_x', 'body_acc_y', 'body_acc_z', 
                     'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
        corr_matrix = df[corr_cols].corr()
        im = axes[2, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[2, 0].set_title('Sensor Correlation Matrix')
        axes[2, 0].set_xticks(range(len(corr_cols)))
        axes[2, 0].set_yticks(range(len(corr_cols)))
        axes[2, 0].set_xticklabels([col.replace('body_', '') for col in corr_cols], rotation=45)
        axes[2, 0].set_yticklabels([col.replace('body_', '') for col in corr_cols])
        plt.colorbar(im, ax=axes[2, 0])
        
        # Plot 6: Power spectral density
        from scipy import signal
        f, psd = signal.welch(df_viz['acc_magnitude'], fs=self.sampling_rate, nperseg=256)
        axes[2, 1].semilogy(f, psd)
        axes[2, 1].set_title('Power Spectral Density - Acc Magnitude')
        axes[2, 1].set_xlabel('Frequency (Hz)')
        axes[2, 1].set_ylabel('PSD')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = os.path.join(self.data_dir, 'uci_har_visualization.png')
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {viz_file}")
        plt.show()
    
    def process_dataset(self, max_samples=10000):
        """Complete processing pipeline"""
        print("Starting UCI HAR dataset processing for DRR analysis...")
        print("=" * 60)
        
        # Step 1: Download and extract
        self.download_dataset()
        self.extract_dataset()
        
        # Step 2: Load raw signals
        print("Loading training data...")
        train_signals, train_labels, train_subjects = self.load_raw_signals('train')
        
        # Step 3: Create continuous time series
        df_continuous = self.create_continuous_timeseries(
            train_signals, train_labels, train_subjects, max_samples
        )
        
        # Step 4: Preprocess for DRR
        df_processed = self.preprocess_for_drr(df_continuous)
        
        # Step 5: Save processed data
        self.save_processed_data(df_processed)
        
        # Step 6: Create visualizations
        self.visualize_data(df_processed)
        
        print("\nDataset processing complete!")
        print(f"Processed data available at: {self.processed_file}")
        
        return df_processed


def create_drr_config(data_file, config_file='config.yml'):
    """
    Create or update DRR configuration file for UCI HAR dataset
    """
    
    # DRR configuration for UCI HAR dataset
    config = {
        'data': {
            'source': 'file',
            'file_path': data_file,
            'sampling_rate': 50,
            'duration': None,  # Use full dataset
            'dt': 0.02  # 1/50 Hz
        },
        
        'datasets': {
            'uci_har': {
                'path': data_file,
                'sampling_rate': 50,
                'columns': [
                    'body_acc_x', 'body_acc_y', 'body_acc_z',
                    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                    'acc_magnitude', 'gyro_magnitude', 'acc_gyro_correlation'
                ],
                'description': 'UCI Human Activity Recognition Dataset - Smartphone Sensor Data'
            }
        },
        
        'analysis': {
            'methods': ['resonance_detection', 'phase_space', 'spectral_analysis'],
            'window_size': 512,
            'overlap': 0.5,
            'frequency_range': [0.1, 25.0]  # Hz
        },
        
        'resonance_detection': {
            'min_frequency': 0.1,
            'max_frequency': 25.0,
            'prominence_threshold': 0.1,
            'persistence_threshold': 0.05
        },
        
        'phase_space': {
            'embedding_dimension': 3,
            'time_delay': 1,
            'method': 'takens'
        },
        
        'visualization': {
            'save_plots': True,
            'plot_format': 'png',
            'dpi': 300,
            'output_dir': 'results'
        },
        
        'output': {
            'save_results': True,
            'results_dir': 'results',
            'export_format': ['csv', 'json']
        }
    }
    
    # Save configuration
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"DRR configuration saved to: {config_file}")
    return config


def run_drr_analysis_demo(data_file):
    """
    Demonstrate basic DRR analysis concepts on the UCI HAR data
    """
    print("Running DRR Analysis Demo...")
    print("=" * 40)
    
    # Load the processed data
    df = pd.read_csv(data_file)
    
    # Select DRR analysis columns
    drr_columns = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'acc_magnitude', 'gyro_magnitude'
    ]
    
    # Basic DRR analysis steps
    print("1. Dynamic Resonance Detection:")
    from scipy import signal
    
    # Analyze each sensor for dominant frequencies
    for col in drr_columns[:3]:  # Just first 3 for demo
        data = df[col].values
        
        # Power spectral density
        f, psd = signal.welch(data, fs=50, nperseg=512)
        
        # Find peaks (resonances)
        peaks, properties = signal.find_peaks(psd, prominence=np.max(psd)*0.1)
        
        print(f"  {col}: Found {len(peaks)} resonant frequencies")
        if len(peaks) > 0:
            dominant_freq = f[peaks[np.argmax(psd[peaks])]]
            print(f"    Dominant frequency: {dominant_freq:.2f} Hz")
    
    print("\n2. Phase Space Reconstruction:")
    # Simple phase space reconstruction example
    data = df['acc_magnitude'].values[:1000]  # First 1000 samples
    
    # Time delay embedding
    delay = 10
    embed_dim = 3
    
    embedded = np.zeros((len(data) - (embed_dim-1)*delay, embed_dim))
    for i in range(embed_dim):
        embedded[:, i] = data[i*delay:len(data)-(embed_dim-1-i)*delay]
    
    print(f"  Created {embed_dim}D phase space with {len(embedded)} points")
    
    print("\n3. Resonance Depth Calculation:")
    # Calculate persistence of resonances across different activities
    activities = df['activity_name'].unique()
    
    for activity in activities[:3]:  # First 3 activities for demo
        activity_data = df[df['activity_name'] == activity]['acc_magnitude'].values
        
        if len(activity_data) > 100:
            f, psd = signal.welch(activity_data, fs=50, nperseg=128)
            peaks, _ = signal.find_peaks(psd, prominence=np.max(psd)*0.05)
            
            # Simple resonance depth metric
            if len(peaks) > 0:
                depth = np.mean(psd[peaks]) / np.mean(psd)
                print(f"  {activity}: Resonance depth = {depth:.3f}")
    
    print("\nDRR Demo complete! This shows the basic concepts.")
    print("For full DRR analysis, integrate with your DRR framework.")


def main():
    """Main execution function"""
    print("UCI HAR Dataset Integration with DRR Framework")
    print("=" * 50)
    
    # Initialize processor
    processor = UCIHARDataProcessor()
    
    # Process the dataset
    df = processor.process_dataset(max_samples=20000)  # Limit for demo
    
    # Create DRR configuration
    config = create_drr_config(processor.processed_file)
    
    # Run demo analysis
    run_drr_analysis_demo(processor.processed_file)
    
    print("\n" + "=" * 50)
    print("INTEGRATION COMPLETE!")
    print("=" * 50)
    print(f"📁 Processed data: {processor.processed_file}")
    print(f"⚙️  Configuration: config.yml")
    print(f"📊 Visualization: {os.path.join(processor.data_dir, 'uci_har_visualization.png')}")
    print("\nNext steps:")
    print("1. Review the processed data and configuration")
    print("2. Run your DRR framework with: python scripts/run_analysis.py --config config.yml")
    print("3. Explore the results in the 'results' directory")
    print("\nThe dataset contains rich multivariate sensor data perfect for")
    print("Dynamic Resonance Rooting analysis of human movement patterns!")


if __name__ == "__main__":
    main()
