"""
Model Training Module for Customer Churn Prediction
This module implements five classification algorithms: Logistic Regression, Decision Tree, 
Random Forest, XGBoost, and Neural Network
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, make it optional
XGBOOST_AVAILABLE = False
xgb = None
try:
    import xgboost as xgb
    _ = xgb.XGBClassifier()
    XGBOOST_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    XGBOOST_AVAILABLE = False
    xgb = None

# Try to import TensorFlow, make it optional (for advanced neural networks)
TENSORFLOW_AVAILABLE = False
tf = None
keras = None
layers = None
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except (ImportError, Exception):
    TENSORFLOW_AVAILABLE = False


class ModelTrainer:
    """Class for training multiple classification models"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.best_params = {}
        
    def initialize_models(self):
        """Initialize all classification models"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE and xgb is not None:
            self.models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        # Add Neural Network using scikit-learn's MLPClassifier
        self.models['Neural Network'] = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            verbose=False
        )
        
        print("Models initialized successfully!")
    
    def create_neural_network(self, input_dim):
        """
        Create a neural network model using scikit-learn's MLPClassifier
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
            
        Returns:
        --------
        MLPClassifier
            Neural network model
        """
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            verbose=False
        )
        
        return model
    
    def train_models(self, X_train, y_train, use_hyperparameter_tuning=False):
        """
        Train all models on the training data
        
        Parameters:
        -----------
        X_train : np.array or pd.DataFrame
            Training features
        y_train : np.array or pd.Series
            Training labels
        use_hyperparameter_tuning : bool
            Whether to use hyperparameter tuning
            
        Returns:
        --------
        dict
            Dictionary of trained models
        """
        if not self.models:
            self.initialize_models()
        
        print("\n" + "="*60)
        print("Training Models")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            if use_hyperparameter_tuning and name in ['Random Forest', 'XGBoost', 'Neural Network']:
                # Hyperparameter tuning for key models
                if name == 'Random Forest':
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5, 10]
                    }
                elif name == 'XGBoost':
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 5, 7]
                    }
                elif name == 'Neural Network':
                    param_grid = {
                        'hidden_layer_sizes': [(64, 32), (64, 32, 16), (128, 64)],
                        'learning_rate_init': [0.001, 0.01, 0.1],
                        'alpha': [0.0001, 0.001, 0.01]
                    }
                
                random_search = RandomizedSearchCV(
                    model, param_grid, cv=5, scoring='roc_auc', 
                    n_iter=10, random_state=42, n_jobs=-1
                )
                random_search.fit(X_train, y_train)
                self.trained_models[name] = random_search.best_estimator_
                self.best_params[name] = random_search.best_params_
                print(f"Best parameters: {random_search.best_params_}")
            else:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
            
            print(f"{name} training completed!")
        
        print("\n" + "="*60)
        print("All models trained successfully!")
        print("="*60)
        
        return self.trained_models
    
    def cross_validate(self, X_train, y_train, cv=5):
        """
        Perform cross-validation on all models
        
        Parameters:
        -----------
        X_train : np.array or pd.DataFrame
            Training features
        y_train : np.array or pd.Series
            Training labels
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        pd.DataFrame
            Cross-validation results
        """
        if not self.trained_models:
            print("Models not trained yet. Training models first...")
            self.train_models(X_train, y_train)
        
        print("\n" + "="*60)
        print("Cross-Validation Results")
        print("="*60)
        
        cv_results = {}
        y_train_np = np.array(y_train).ravel()
        
        for name, model in self.trained_models.items():
            print(f"\nCross-validating {name}...")
            
            scores = cross_val_score(model, X_train, y_train_np, cv=cv, scoring='roc_auc')
            
            cv_results[name] = {
                'Mean AUC-ROC': scores.mean(),
                'Std AUC-ROC': scores.std(),
                'Min AUC-ROC': scores.min(),
                'Max AUC-ROC': scores.max()
            }
            print(f"Mean AUC-ROC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        cv_df = pd.DataFrame(cv_results).T
        cv_df = cv_df.sort_values('Mean AUC-ROC', ascending=False)
        
        print("\n" + "="*60)
        print("Cross-Validation Summary")
        print("="*60)
        print(cv_df)
        
        return cv_df
    
    def save_models(self, directory='models'):
        """
        Save trained models to disk
        
        Parameters:
        -----------
        directory : str
            Directory to save models
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.trained_models.items():
            filename = f"{directory}/{name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
        
        print(f"\nAll models saved to {directory}/ directory")
    
    def load_models(self, directory='models'):
        """
        Load trained models from disk
        
        Parameters:
        -----------
        directory : str
            Directory containing saved models
        """
        import os
        import glob
        
        model_files = glob.glob(f"{directory}/*.pkl")
        
        for file_path in model_files:
            model_name = os.path.basename(file_path).replace('.pkl', '').replace('_', ' ').title()
            self.trained_models[model_name] = joblib.load(file_path)
            print(f"Loaded {model_name} from {file_path}")
        
        print(f"\nLoaded {len(self.trained_models)} models from {directory}/ directory")

