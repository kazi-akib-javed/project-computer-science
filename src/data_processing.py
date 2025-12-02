"""
Data Processing Module for Customer Churn Prediction
This module handles data loading, cleaning, and basic preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Class for processing customer churn data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self, file_path):
        """
        Load data from CSV file
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataset
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            
            # Drop customerID column if it exists (identifier, not a feature)
            if 'customerID' in df.columns:
                df = df.drop(columns=['customerID'])
                print(f"Dropped 'customerID' column. New shape: {df.shape}")
            
            return df
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with handled missing values
        """
        print("\nHandling missing values...")
        initial_missing = df.isnull().sum().sum()
        print(f"Missing values before handling: {initial_missing}")
        
        # Convert TotalCharges to numeric (handles empty strings/spaces)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = df['TotalCharges'].replace([' ', ''], np.nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            totalcharges_missing = df['TotalCharges'].isnull().sum()
            if totalcharges_missing > 0:
                print(f"Converted 'TotalCharges' to numeric. Found {totalcharges_missing} missing values.")
        
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled {col} missing values with median: {median_val:.2f}")
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled {col} missing values with mode: {mode_val}")
        
        final_missing = df.isnull().sum().sum()
        print(f"Missing values after handling: {final_missing}")
        return df
    
    def encode_categorical_variables(self, df, target_column='Churn'):
        """
        Encode categorical variables using Label Encoding
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_column : str
            Name of the target column
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with encoded categorical variables
        """
        print("\nEncoding categorical variables...")
        df_encoded = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != target_column:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        # Encode target variable
        if target_column in df.columns:
            le_target = LabelEncoder()
            df_encoded[target_column] = le_target.fit_transform(df_encoded[target_column])
            self.label_encoders[target_column] = le_target
        
        return df_encoded
    
    def split_data(self, df, target_column='Churn', test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_column : str
            Name of the target column
        test_size : float
            Proportion of test set
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test
        """
        print("\nSplitting data into train and test sets...")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.feature_columns = X.columns.tolist()
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Class distribution in training set:\n{y_train.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def process(self, file_path, target_column='Churn', test_size=0.2, random_state=42, save_processed=False):
        """
        Complete data processing pipeline
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        target_column : str
            Name of the target column
        test_size : float
            Proportion of test set
        random_state : int
            Random seed for reproducibility
        save_processed : bool
            Whether to save processed data to data/processed/ folder
            
        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test (ready for feature engineering)
        """
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical variables
        df = self.encode_categorical_variables(df, target_column)
        
        # Save processed data if requested
        if save_processed:
            import os
            os.makedirs('data/processed', exist_ok=True)
            processed_file = 'data/processed/cleaned_data.csv'
            df.to_csv(processed_file, index=False)
            print(f"\nProcessed data saved to {processed_file}")
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(
            df, target_column, test_size, random_state
        )
        
        print("\nData processing completed successfully!")
        return X_train, X_test, y_train, y_test

