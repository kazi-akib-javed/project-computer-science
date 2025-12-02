"""
Main Script for Customer Churn Prediction Project
This script orchestrates the complete machine learning pipeline as described in the report
"""

import os
import sys
import pandas as pd
import numpy as np
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator


def main():
    """
    Main function to run the complete customer churn prediction pipeline
    """
    print("="*70)
    print("Customer Churn Prediction Using Machine Learning")
    print("A Comparative Analysis of Classification Algorithms")
    print("="*70)
    
    # Configuration
    if os.path.exists('./data/raw/telco_churn.csv'):
        DATA_FILE = './data/raw/telco_churn.csv'
    elif os.path.exists('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
        DATA_FILE = './data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    else:
        DATA_FILE = './data/raw/telco_churn.csv'
    TARGET_COLUMN = 'Churn'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    USE_HYPERPARAMETER_TUNING = False
    APPLY_SMOTE = True
    
    # Step 1: Data Processing
    print("\n" + "="*70)
    print("STEP 1: DATA PROCESSING")
    print("="*70)
    
    processor = DataProcessor()
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"\nError: Data file '{DATA_FILE}' not found.")
        print("Please ensure the dataset is in the data/ directory.")
        sys.exit(1)
    
    # Process data
    result = processor.process(
        DATA_FILE, 
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        save_processed=True
    )
    
    if result is None:
        print("Error in data processing. Exiting...")
        sys.exit(1)
    
    X_train, X_test, y_train, y_test = result
    
    # Step 2: Feature Engineering
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    
    feature_engineer = FeatureEngineer()
    X_train_fe, X_test_fe, y_train_balanced, _ = feature_engineer.engineer_features(
        X_train, X_test, y_train, apply_smote=APPLY_SMOTE
    )
    
    # Step 3: Model Training
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    
    trainer = ModelTrainer()
    trainer.train_models(
        X_train_fe, 
        y_train_balanced if APPLY_SMOTE else y_train, 
        use_hyperparameter_tuning=USE_HYPERPARAMETER_TUNING
    )
    
    # Cross-validation
    print("\n" + "="*70)
    print("CROSS-VALIDATION")
    print("="*70)
    cv_results = trainer.cross_validate(
        X_train_fe, 
        y_train_balanced if APPLY_SMOTE else y_train, 
        cv=5
    )
    
    # Step 4: Model Evaluation
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    evaluator = ModelEvaluator()
    results_df = evaluator.evaluate_all_models(
        trainer.trained_models, 
        X_test_fe, 
        y_test
    )
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    evaluator.plot_confusion_matrices(trainer.trained_models, X_test_fe, y_test)
    evaluator.plot_roc_curves(trainer.trained_models, X_test_fe, y_test)
    evaluator.plot_metrics_comparison(results_df)
    
    # SHAP Analysis for best model
    print("\n" + "="*70)
    print("SHAP INTERPRETABILITY ANALYSIS")
    print("="*70)
    
    best_model_name = results_df.iloc[0]['Model']
    best_model = trainer.trained_models[best_model_name]
    
    # Get feature names if available
    feature_names = feature_engineer.feature_columns if hasattr(feature_engineer, 'feature_columns') else None
    
    evaluator.apply_shap_analysis(
        best_model, 
        X_test_fe, 
        best_model_name,
        feature_names=feature_names,
        max_samples=100
    )
    
    # Detailed report for best model
    print("\n" + "="*70)
    print("BEST MODEL DETAILED REPORT")
    print("="*70)
    
    evaluator.print_detailed_report(best_model, X_test_fe, y_test, best_model_name)
    
    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    trainer.save_models('models')
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    results_df.to_csv('model_evaluation_results.csv', index=False)
    cv_results.to_csv('cross_validation_results.csv')
    print("Results saved to CSV files")
    
    # Summary
    print("\n" + "="*70)
    print("PROJECT SUMMARY")
    print("="*70)
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
    print(f"Best F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
    if results_df.iloc[0]['ROC-AUC']:
        print(f"Best AUC-ROC: {results_df.iloc[0]['ROC-AUC']:.4f}")
    print(f"\nAll models evaluated: {len(results_df)}")
    print(f"Visualizations saved: confusion_matrices.png, roc_curves.png, metrics_comparison.png")
    print(f"SHAP plots saved: shap_summary_{best_model_name.replace(' ', '_').lower()}.png")
    print(f"Models saved in: models/ directory")
    print(f"Results saved: model_evaluation_results.csv, cross_validation_results.csv")
    
    print("\n" + "="*70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()
