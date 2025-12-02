"""
Model Evaluation Module for Customer Churn Prediction
This module evaluates and compares the performance of different models and provides SHAP interpretability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Class for evaluating and comparing model performance"""
    
    def __init__(self):
        self.evaluation_results = {}
        self.shap_explainers = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a single model
        
        Parameters:
        -----------
        model : sklearn model or keras model
            Trained model
        X_test : np.array or pd.DataFrame
            Test features
        y_test : np.array or pd.Series
            Test labels
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate ROC AUC if probabilities are available
        roc_auc = None
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                roc_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Confusion Matrix': cm,
            'Predictions': y_pred,
            'Probabilities': y_pred_proba
        }
        
        self.evaluation_results[model_name] = results
        
        return results
    
    def evaluate_all_models(self, models, X_test, y_test):
        """
        Evaluate all models
        
        Parameters:
        -----------
        models : dict
            Dictionary of trained models
        X_test : np.array or pd.DataFrame
            Test features
        y_test : np.array or pd.Series
            Test labels
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with evaluation results
        """
        print("\n" + "="*60)
        print("Evaluating All Models")
        print("="*60)
        
        results_list = []
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            results = self.evaluate_model(model, X_test, y_test, name)
            results_list.append({
                'Model': name,
                'Accuracy': results['Accuracy'],
                'Precision': results['Precision'],
                'Recall': results['Recall'],
                'F1-Score': results['F1-Score'],
                'ROC-AUC': results['ROC-AUC']
            })
            print(f"Accuracy: {results['Accuracy']:.4f}")
            print(f"F1-Score: {results['F1-Score']:.4f}")
            if results['ROC-AUC']:
                print(f"AUC-ROC: {results['ROC-AUC']:.4f}")
        
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('ROC-AUC', ascending=False)
        
        print("\n" + "="*60)
        print("Evaluation Summary")
        print("="*60)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def plot_confusion_matrices(self, models, X_test, y_test, figsize=(15, 10)):
        """
        Plot confusion matrices for all models
        
        Parameters:
        -----------
        models : dict
            Dictionary of trained models
        X_test : np.array or pd.DataFrame
            Test features
        y_test : np.array or pd.Series
            Test labels
        figsize : tuple
            Figure size
        """
        n_models = len(models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, y_pred):.4f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrices saved to 'confusion_matrices.png'")
        plt.close()
    
    def plot_roc_curves(self, models, X_test, y_test, figsize=(10, 8)):
        """
        Plot ROC curves for all models
        
        Parameters:
        -----------
        models : dict
            Dictionary of trained models
        X_test : np.array or pd.DataFrame
            Test features
        y_test : np.array or pd.Series
            Test labels
        figsize : tuple
            Figure size
        """
        plt.figure(figsize=figsize)
        
        for name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    continue
                
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
            except Exception as e:
                print(f"Could not plot ROC for {name}: {e}")
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("ROC curves saved to 'roc_curves.png'")
        plt.close()
    
    def plot_metrics_comparison(self, results_df, figsize=(12, 6)):
        """
        Plot comparison of metrics across models
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame with evaluation results
        figsize : tuple
            Figure size
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        if 'ROC-AUC' in results_df.columns:
            metrics.append('ROC-AUC')
        
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        
        for idx, metric in enumerate(metrics):
            if metric in results_df.columns:
                results_df_sorted = results_df.sort_values(metric, ascending=True)
                axes[idx].barh(results_df_sorted['Model'], results_df_sorted[metric])
                axes[idx].set_xlabel(metric)
                axes[idx].set_title(metric)
                axes[idx].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("Metrics comparison saved to 'metrics_comparison.png'")
        plt.close()
    
    def apply_shap_analysis(self, model, X_test, model_name, feature_names=None, max_samples=100):
        """
        Apply SHAP analysis to a model
        
        Parameters:
        -----------
        model : sklearn model or keras model
            Trained model
        X_test : np.array or pd.DataFrame
            Test features
        model_name : str
            Name of the model
        feature_names : list
            List of feature names
        max_samples : int
            Maximum number of samples to use for SHAP (for performance)
            
        Returns:
        --------
        shap.Explainer
            SHAP explainer object
        """
        print(f"\nApplying SHAP analysis to {model_name}...")
        
        # Limit samples for performance
        if len(X_test) > max_samples:
            sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_sample = X_test[sample_indices]
        else:
            X_sample = X_test
        
        try:
            explainer = shap.Explainer(model, X_sample)
            
            shap_values = explainer(X_sample)
            self.shap_explainers[model_name] = {
                'explainer': explainer,
                'shap_values': shap_values,
                'feature_names': feature_names
            }
            
            # Plot SHAP summary
            plt.figure(figsize=(10, 8))
            shap.plots.beeswarm(shap_values, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.savefig(f'shap_summary_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"SHAP analysis completed for {model_name}")
            print(f"SHAP plot saved to 'shap_summary_{model_name.replace(' ', '_').lower()}.png'")
            
            return explainer
            
        except Exception as e:
            print(f"Error applying SHAP to {model_name}: {e}")
            return None
    
    def print_detailed_report(self, model, X_test, y_test, model_name):
        """
        Print detailed classification report
        
        Parameters:
        -----------
        model : sklearn model or keras model
            Trained model
        X_test : np.array or pd.DataFrame
            Test features
        y_test : np.array or pd.Series
            Test labels
        model_name : str
            Name of the model
        """
        y_pred = model.predict(X_test)
        
        print("\n" + "="*60)
        print(f"Detailed Report for {model_name}")
        print("="*60)
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

