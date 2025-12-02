"""
Feature Engineering Module for Customer Churn Prediction
This module creates new features, handles encoding, scaling, and class imbalance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Class for feature engineering and preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        self.feature_columns = None
        
    def create_tenure_groups(self, df):
        """
        Create tenure groups from tenure months
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with 'tenure' column
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with tenure groups
        """
        if 'tenure' in df.columns:
            df = df.copy()
            df['TenureGroup'] = pd.cut(
                df['tenure'],
                bins=[0, 12, 24, 48, 72, float('inf')],
                labels=['0-12', '13-24', '25-48', '49-72', '73+']
            )
            print("Created tenure groups")
        return df
    
    def calculate_service_counts(self, df):
        """
        Calculate count of services subscribed by each customer
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with service count feature
        """
        service_columns = [
            'PhoneService', 'MultipleLines', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies'
        ]
        
        df = df.copy()
        available_services = [col for col in service_columns if col in df.columns]
        
        if available_services:
            df['ServiceCount'] = df[available_services].sum(axis=1)
            print(f"Created service count feature from {len(available_services)} services")
        
        return df
    
    def calculate_charge_ratios(self, df):
        """
        Calculate charge ratios (MonthlyCharges/TotalCharges)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with charge ratio feature
        """
        df = df.copy()
        
        if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
            # Avoid division by zero
            df['ChargeRatio'] = np.where(
                df['TotalCharges'] > 0,
                df['MonthlyCharges'] / df['TotalCharges'],
                0
            )
            print("Created charge ratio feature")
        
        return df
    
    def apply_one_hot_encoding(self, df, categorical_columns=None):
        """
        Apply one-hot encoding to categorical variables
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        categorical_columns : list
            List of categorical column names
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with one-hot encoded features
        """
        df = df.copy()
        
        if categorical_columns is None:
            # Auto-detect categorical columns (object type or low cardinality)
            categorical_columns = []
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].nunique() < 10:
                    categorical_columns.append(col)
        
        # Remove target column if present
        if 'Churn' in categorical_columns:
            categorical_columns.remove('Churn')
        
        available_categorical = [col for col in categorical_columns if col in df.columns]
        
        if available_categorical:
            df_encoded = pd.get_dummies(df, columns=available_categorical, prefix=available_categorical)
            print(f"Applied one-hot encoding to {len(available_categorical)} columns")
            return df_encoded
        
        return df
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.array
            Training features
        X_test : pd.DataFrame or np.array
            Test features
            
        Returns:
        --------
        tuple
            Scaled training and test features
        """
        print("\nScaling features...")
        
        # Convert to numpy if DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
            X_test_array = X_test.values
            self.feature_columns = X_train.columns.tolist()
        else:
            X_train_array = X_train
            X_test_array = X_test
        
        X_train_scaled = self.scaler.fit_transform(X_train_array)
        X_test_scaled = self.scaler.transform(X_test_array)
        
        return X_train_scaled, X_test_scaled
    
    def apply_smote(self, X_train, y_train):
        """
        Apply SMOTE to balance classes
        
        Parameters:
        -----------
        X_train : np.array or pd.DataFrame
            Training features
        y_train : np.array or pd.Series
            Training labels
            
        Returns:
        --------
        tuple
            Balanced training features and labels
        """
        print("\nApplying SMOTE for class balancing...")
        print(f"Class distribution before SMOTE:\n{pd.Series(y_train).value_counts()}")
        
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)
        
        print(f"Class distribution after SMOTE:\n{pd.Series(y_train_balanced).value_counts()}")
        
        return X_train_balanced, y_train_balanced
    
    def engineer_features(self, X_train, X_test, y_train, apply_smote=True):
        """
        Complete feature engineering pipeline
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        y_train : pd.Series
            Training labels
        apply_smote : bool
            Whether to apply SMOTE for class balancing
            
        Returns:
        --------
        tuple
            X_train_scaled, X_test_scaled, y_train_balanced (or y_train), y_test
        """
        # Convert to DataFrame if needed
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
        
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        # Create new features on training data
        X_train_fe = self.create_tenure_groups(X_train)
        X_train_fe = self.calculate_service_counts(X_train_fe)
        X_train_fe = self.calculate_charge_ratios(X_train_fe)
        
        # Create same features on test data
        X_test_fe = self.create_tenure_groups(X_test)
        X_test_fe = self.calculate_service_counts(X_test_fe)
        X_test_fe = self.calculate_charge_ratios(X_test_fe)
        
        # Convert categorical features to numeric
        for col in X_train_fe.columns:
            if X_train_fe[col].dtype == 'object' or X_train_fe[col].dtype.name == 'category':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X_train_fe[col] = le.fit_transform(X_train_fe[col].astype(str))
                X_test_fe[col] = le.transform(X_test_fe[col].astype(str))
        
        # Update feature columns
        self.feature_columns = X_train_fe.columns.tolist()
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train_fe, X_test_fe)
        
        # Apply SMOTE if requested
        if apply_smote:
            X_train_balanced, y_train_balanced = self.apply_smote(X_train_scaled, y_train)
            return X_train_balanced, X_test_scaled, y_train_balanced, None
        else:
            return X_train_scaled, X_test_scaled, y_train, None

