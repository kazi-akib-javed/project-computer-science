# Customer Churn Prediction Using Machine Learning

## Overview

This project implements an intelligent machine learning-based customer churn prediction system that analyzes customer behavior patterns to identify at-risk customers before they leave. The system uses multiple classification algorithms including Logistic Regression, Decision Trees, Random Forest, XGBoost, and Neural Networks, with advanced techniques like SMOTE for class imbalance handling and SHAP for model interpretability. The solution enables data-driven retention strategies by providing both accurate predictions and actionable business insights.

## Problem Statement

In the telecommunications industry, customer retention directly impacts profitability. Studies indicate that reducing churn by just 5% can increase profits by 25-95%, making churn prediction a critical business priority. The challenge is compounded by competitive markets where customers can easily switch providers, leading to annual churn rates of 15-30% in densely populated urban markets.

Traditional approaches to churn management rely on reactive measures—contacting customers only after they signal intent to leave. By this stage, retention efforts often prove ineffective and costly. The relationship between early prediction and successful retention is clear: identifying at-risk customers months in advance enables proactive interventions through targeted offers, improved service, or personalized communication.

A machine learning system that accurately predicts churn while providing interpretable insights would enable data-driven retention strategies. Integration of multiple algorithms and interpretability techniques is necessary because business stakeholders require both accurate predictions and understandable explanations for operational decisions.

## Key Features

- **Advanced Classification Algorithms**: Implements 5 state-of-the-art algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost, Neural Network)

- **Class Imbalance Handling**: Uses SMOTE (Synthetic Minority Over-sampling Technique) to address imbalanced datasets where non-churners significantly outnumber churners

- **Model Interpretability**: SHAP (SHapley Additive exPlanations) analysis provides feature-level explanations for predictions, enabling actionable business insights

- **Comprehensive Feature Engineering**: Creates derived features including tenure groups, service counts, and charge ratios to improve predictive power

- **Robust Evaluation Framework**: 5-fold cross-validation with multiple metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)

- **Interactive Analysis**: Jupyter notebooks for step-by-step exploration and experimentation

- **Visual Analytics**: Generates confusion matrices, ROC curves, metrics comparison charts, and SHAP plots

- **Model Persistence**: Saves trained models for future predictions and deployment

- **Modular Architecture**: Clean separation of concerns with maintainable utility and processing modules

- **Automated Pipeline**: End-to-end automation from data loading to model evaluation and interpretation

## System Architecture

### Core Components

1. **Data Processing Module** (`src/data_processing.py`)

   - CSV data loading with automatic customer ID removal

   - Missing value imputation (median for numerical, mode for categorical)

   - TotalCharges conversion and handling of empty strings

   - Label encoding for categorical variables

   - Train-test split with stratified sampling (80/20)

   - Processed data export to `data/processed/` folder

2. **Feature Engineering Module** (`src/feature_engineering.py`)

   - **Tenure Grouping**: Creates categorical groups from tenure months (0-12, 13-24, 25-48, 49-72, 73+)

   - **Service Count Calculation**: Aggregates number of services subscribed per customer

   - **Charge Ratio Computation**: Calculates MonthlyCharges/TotalCharges ratio

   - **SMOTE Application**: Synthetic minority oversampling to balance classes (4139:1495 → 4139:4139)

   - **Feature Scaling**: StandardScaler normalization for numerical features

   - **Backward Compatibility**: Handles both DataFrame and array inputs

3. **Model Training Module** (`src/model_training.py`)

   - **Logistic Regression**: Linear classifier with L2 regularization

   - **Decision Tree**: Non-parametric tree-based classifier

   - **Random Forest**: Ensemble of 100 decision trees with bootstrap aggregation

   - **XGBoost**: Extreme gradient boosting with regularization (optional, requires OpenMP)

   - **Neural Network**: Multi-layer perceptron (64-32-16 neurons) with dropout and early stopping

   - **Hyperparameter Tuning**: RandomizedSearchCV for optimal parameter selection

   - **Cross-Validation**: 5-fold stratified CV with ROC-AUC scoring

   - **Model Persistence**: Pickle serialization for all models

4. **Evaluation & Analytics Module** (`src/evaluation.py`)

   - **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC calculation

   - **Confusion Matrix Generation**: Visual matrices for all models

   - **ROC Curve Analysis**: Comparative ROC curves with AUC scores

   - **Metrics Comparison**: Side-by-side comparison charts

   - **SHAP Interpretability**: Feature importance analysis using Shapley values

   - **Detailed Reporting**: Classification reports and confusion matrix breakdowns

   - **CSV/JSON Export**: Comprehensive results logging

5. **Main Pipeline** (`main.py`)

   - **Automated Execution**: Complete pipeline from data loading to model evaluation

   - **Smart Path Detection**: Automatic data file location (data/raw/ or data/)

   - **Configurable Parameters**: SMOTE toggle, hyperparameter tuning, test size

   - **Progress Tracking**: Step-by-step console output with progress indicators

   - **Result Consolidation**: All outputs saved to organized directories

6. **Interactive Notebooks** (`notebooks/`)

   - **01_data_cleaning.ipynb**: Data loading and cleaning workflow

   - **02_exploratory_analysis.ipynb**: EDA with visualizations and insights

   - **03_feature_engineering.ipynb**: Feature creation and SMOTE application

   - **04_model_training.ipynb**: Model training and cross-validation

   - **05_model_evaluation.ipynb**: Evaluation, visualization, and SHAP analysis

### Technical Stack

- **Machine Learning**: scikit-learn 1.2.2+ (Logistic Regression, Decision Tree, Random Forest, MLPClassifier)

- **Gradient Boosting**: XGBoost 1.7.5+ (optional, requires OpenMP library)

- **Deep Learning**: TensorFlow 2.12.0+ (optional, for advanced neural networks, requires Python 3.11-3.12)

- **Class Imbalance**: imbalanced-learn 0.10.1+ (SMOTE implementation)

- **Model Interpretability**: SHAP 0.41.0+ (Shapley Additive exPlanations)

- **Data Processing**: Pandas 1.5.3+, NumPy 1.24.3+

- **Visualization**: Matplotlib 3.7.1+, Seaborn 0.12.2+

- **Model Persistence**: Joblib 1.2.0+

- **Interactive Analysis**: Jupyter 1.0.0+

- **Architecture**: Modular design with separated concerns

- **Language**: Python 3.9+

### Algorithm Comparison

| Algorithm | Type | Interpretability | Best For | Performance |
|-----------|------|------------------|----------|-------------|
| **Logistic Regression** | Linear | High | Baseline, interpretable coefficients | Fast, reliable |
| **Decision Tree** | Tree-based | Very High | Understanding decision rules | Moderate |
| **Random Forest** | Ensemble | Medium | High accuracy, feature importance | Excellent |
| **XGBoost** | Gradient Boosting | Medium | State-of-the-art performance | Excellent |
| **Neural Network** | Deep Learning | Low (needs SHAP) | Complex patterns, non-linear | Very Good |

### Model Performance Metrics

| Metric | Description | Benchmark | Interpretation |
|--------|-------------|-----------|----------------|
| **Accuracy** | Overall correctness | > 0.70 | Percentage of correct predictions |
| **Precision** | Positive prediction accuracy | > 0.60 | Of predicted churners, how many actually churn |
| **Recall** | True positive rate | > 0.65 | Of actual churners, how many we identify |
| **F1-Score** | Harmonic mean | > 0.60 | Balanced precision-recall metric |
| **ROC-AUC** | Classification ability | > 0.75 | Area under ROC curve (higher is better) |

## Installation and Setup

### Prerequisites

```bash
# Core dependencies
pip install pandas>=1.5.3
pip install numpy>=1.24.3
pip install scikit-learn>=1.2.2
pip install matplotlib>=3.7.1
pip install seaborn>=0.12.2
pip install joblib>=1.2.0
pip install imbalanced-learn>=0.10.1
pip install shap>=0.41.0
pip install jupyter>=1.0.0
```

### Optional Dependencies

```bash
# For XGBoost (requires OpenMP library on macOS: brew install libomp)
pip install xgboost>=1.7.5

# For TensorFlow Neural Networks (requires Python 3.11-3.12, not 3.14+)
pip install tensorflow>=2.12.0
```

### Quick Start

1. **Clone the repository**

2. **Install dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your dataset**: 
   - Place CSV file in `data/raw/telco_churn.csv`
   - Or use existing file at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

4. **Run the system**: 
   ```bash
   python main.py
   ```

## Folder Structure

### Input Data Folders

```
data/
├── raw/                   # Original dataset files
│   └── telco_churn.csv   # IBM Telco Customer Churn dataset
└── processed/            # Cleaned and processed data
    └── cleaned_data.csv  # Processed dataset (auto-generated)
```

### Source Code

```
src/
├── data_processing.py     # Data loading, cleaning, encoding
├── feature_engineering.py # Feature creation, SMOTE, scaling
├── model_training.py      # Model training and cross-validation
└── evaluation.py          # Evaluation metrics and SHAP analysis
```

### Interactive Analysis

```
notebooks/
├── 01_data_cleaning.ipynb           # Data preprocessing workflow
├── 02_exploratory_analysis.ipynb    # EDA with visualizations
├── 03_feature_engineering.ipynb     # Feature engineering steps
├── 04_model_training.ipynb          # Model training process
└── 05_model_evaluation.ipynb        # Evaluation and interpretation
```

### Output Folders

```
models/                    # Trained model files
├── logistic_regression.pkl
├── decision_tree.pkl
├── random_forest.pkl
├── neural_network.pkl
└── [xgboost.pkl]         # If XGBoost is available

# Root directory outputs (auto-generated)
├── model_evaluation_results.csv      # Performance metrics for all models
├── cross_validation_results.csv      # CV scores with mean/std/min/max
├── confusion_matrices.png           # Confusion matrices for all models
├── roc_curves.png                   # ROC curve comparison
├── metrics_comparison.png           # Side-by-side metrics comparison
└── shap_summary_[model_name].png    # SHAP feature importance plot
```

## Usage

### Automated Pipeline (Recommended)

**Complete Pipeline Execution:**

```bash
# Simple run - processes everything automatically
python main.py

# With custom configuration (edit main.py)
# - Toggle SMOTE: APPLY_SMOTE = True/False
# - Enable hyperparameter tuning: USE_HYPERPARAMETER_TUNING = True
# - Adjust test size: TEST_SIZE = 0.2
```

**What Happens:**

1. **Data Processing**: Loads data, handles missing values, encodes categorical variables
2. **Feature Engineering**: Creates new features, applies SMOTE, scales features
3. **Model Training**: Trains 4-5 models (depending on optional dependencies)
4. **Cross-Validation**: Performs 5-fold CV on all models
5. **Evaluation**: Calculates metrics, generates visualizations
6. **SHAP Analysis**: Applies interpretability analysis to best model
7. **Results Export**: Saves all models, metrics, and visualizations

### Interactive Notebooks

**Step-by-Step Analysis:**

```bash
# Launch Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

**Notebook Workflow:**

1. **01_data_cleaning.ipynb**: Explore data loading and cleaning
2. **02_exploratory_analysis.ipynb**: Analyze data distributions and correlations
3. **03_feature_engineering.ipynb**: Understand feature creation and SMOTE
4. **04_model_training.ipynb**: Train models and view CV results
5. **05_model_evaluation.ipynb**: Evaluate models and interpret with SHAP

### Programmatic Usage

**Custom Pipeline:**

```python
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator

# 1. Process data
processor = DataProcessor()
X_train, X_test, y_train, y_test = processor.process(
    'data/raw/telco_churn.csv',
    target_column='Churn',
    test_size=0.2,
    random_state=42,
    save_processed=True
)

# 2. Engineer features
feature_engineer = FeatureEngineer()
X_train_fe, X_test_fe, y_train_balanced, _ = feature_engineer.engineer_features(
    X_train, X_test, y_train, apply_smote=True
)

# 3. Train models
trainer = ModelTrainer()
trainer.train_models(
    X_train_fe, 
    y_train_balanced, 
    use_hyperparameter_tuning=False
)

# 4. Cross-validation
cv_results = trainer.cross_validate(X_train_fe, y_train_balanced, cv=5)

# 5. Evaluate models
evaluator = ModelEvaluator()
results_df = evaluator.evaluate_all_models(
    trainer.trained_models, 
    X_test_fe, 
    y_test
)

# 6. SHAP analysis
best_model_name = results_df.iloc[0]['Model']
best_model = trainer.trained_models[best_model_name]
evaluator.apply_shap_analysis(
    best_model, 
    X_test_fe, 
    best_model_name,
    feature_names=feature_engineer.feature_columns,
    max_samples=100
)

# 7. Save models
trainer.save_models('models')
```

## Output Data

### Model Evaluation Results (`model_evaluation_results.csv`)

Contains performance metrics for all models:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7395 | 0.7986 | 0.7395 | 0.7533 | 0.8457 |
| Random Forest | 0.7736 | 0.7738 | 0.7736 | 0.7737 | 0.8234 |
| Neural Network | 0.7388 | 0.7676 | 0.7388 | 0.7485 | 0.7920 |
| Decision Tree | 0.7523 | 0.7542 | 0.7523 | 0.7532 | 0.6860 |

### Cross-Validation Results (`cross_validation_results.csv`)

Contains 5-fold CV statistics:

| Model | Mean AUC-ROC | Std AUC-ROC | Min AUC-ROC | Max AUC-ROC |
|-------|--------------|-------------|-------------|-------------|
| Random Forest | 0.9341 | 0.0339 | 0.8889 | 0.9654 |
| Neural Network | 0.8810 | 0.0138 | 0.8625 | 0.8994 |
| Logistic Regression | 0.8564 | 0.0066 | 0.8497 | 0.8668 |
| Decision Tree | 0.7982 | 0.0629 | 0.7058 | 0.8595 |

### Visualizations

**Confusion Matrices** (`confusion_matrices.png`):
- Grid layout showing confusion matrices for all models
- True Positives, True Negatives, False Positives, False Negatives
- Accuracy displayed for each model

**ROC Curves** (`roc_curves.png`):
- Comparative ROC curves with AUC scores
- Random classifier baseline (diagonal line)
- Model performance ranking

**Metrics Comparison** (`metrics_comparison.png`):
- Side-by-side bar charts for Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Easy visual comparison across models

**SHAP Summary Plot** (`shap_summary_[model_name].png`):
- Feature importance visualization
- Shows how each feature contributes to predictions
- Enables business interpretation and actionability

### Saved Models

All trained models are saved as pickle files in `models/` directory:
- `logistic_regression.pkl`: Logistic Regression model
- `decision_tree.pkl`: Decision Tree model
- `random_forest.pkl`: Random Forest model
- `neural_network.pkl`: Neural Network (MLPClassifier) model
- `xgboost.pkl`: XGBoost model (if available)

**Loading Saved Models:**

```python
from src.model_training import ModelTrainer

trainer = ModelTrainer()
trainer.load_models('models')

# Use for predictions
model = trainer.trained_models['Random Forest']
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

## Current Phase Status

**Phase 3 - Complete ML Pipeline with Interpretability**

**Core System (Complete):**

- **Multi-algorithm architecture** with 5 classification algorithms
- **Advanced feature engineering** with tenure groups, service counts, charge ratios
- **Class imbalance handling** through SMOTE oversampling
- **Comprehensive evaluation** with multiple metrics and cross-validation
- **Model interpretability** using SHAP values for business insights
- **Modular code architecture** with clean separation of concerns
- **Interactive notebooks** for step-by-step exploration
- **Automated pipeline** from data loading to model evaluation
- **Visual analytics** with confusion matrices, ROC curves, and SHAP plots
- **Model persistence** for future predictions and deployment

**Recent Enhancements (Complete):**

- **Smart path detection** for flexible data file location
- **Processed data export** to `data/processed/` folder
- **Enhanced feature engineering** with automatic categorical handling
- **Neural Network implementation** using scikit-learn MLPClassifier (works with all Python versions)
- **Optional dependency handling** for XGBoost and TensorFlow
- **Comprehensive documentation** with detailed usage examples
- **Jupyter notebook integration** for interactive analysis
- **SHAP interpretability** for model explanations

**Next Phase Goals:**

- Real-time prediction API for production deployment
- Database integration for persistent model storage
- Web-based dashboard with live monitoring
- Automated model retraining pipeline
- A/B testing framework for model comparison
- Cost-sensitive learning for business optimization
- Customer segmentation based on churn risk
- Integration with CRM systems for automated interventions

## Technical Considerations

### Model Selection Strategy

The system employs a comparative approach:

1. **Baseline Models**: Logistic Regression and Decision Tree provide interpretable baselines
2. **Ensemble Methods**: Random Forest and XGBoost offer superior accuracy
3. **Deep Learning**: Neural Network captures complex non-linear patterns
4. **Best Model Selection**: Based on ROC-AUC score (primary metric for imbalanced data)

### Class Imbalance Handling

The system addresses class imbalance through:

- **SMOTE Oversampling**: Synthetic minority class generation
- **Stratified Sampling**: Maintains class distribution in train-test split
- **Weighted Metrics**: F1-Score and ROC-AUC account for class imbalance
- **Threshold Tuning**: ROC-AUC enables optimal threshold selection

### Model Interpretability

SHAP values provide:

- **Feature Importance**: Which features drive churn predictions
- **Individual Explanations**: Why specific customers are predicted to churn
- **Business Actionability**: Clear insights for retention strategies
- **Model Trust**: Transparent decision-making process

### Performance Optimization

The system is optimized for:

- **Efficient Processing**: Vectorized operations with NumPy/Pandas
- **Memory Management**: Streaming processing for large datasets
- **Parallel Execution**: Cross-validation and hyperparameter tuning use multiple cores
- **Model Caching**: Saved models eliminate retraining overhead

### Privacy and Compliance

- **Anonymized Data**: No personally identifiable information required
- **Local Processing**: All computation happens locally
- **Data Minimization**: Only necessary features are processed
- **Configurable Retention**: Processed data can be deleted after model training

### Scalability

The architecture supports:

- **Large Datasets**: Efficient processing of 10,000+ customer records
- **Additional Features**: Modular design allows easy feature addition
- **New Algorithms**: Simple integration of additional classifiers
- **Production Deployment**: Model persistence enables API integration

## Dataset Information

**IBM Telco Customer Churn Dataset**

- **Size**: 7,043 customer records
- **Features**: 20 attributes (19 features + 1 target)
- **Churn Rate**: 26.5% (1,869 churned, 5,174 retained)
- **Features Include**:
  - Demographics: Gender, SeniorCitizen, Partner, Dependents
  - Services: PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV, StreamingMovies
  - Account: Contract, PaymentMethod, Tenure, MonthlyCharges, TotalCharges
  - Target: Churn (Yes/No)

## Results Interpretation

### Key Metrics

- **Accuracy > 0.70**: Model correctly predicts churn status for majority of customers
- **Precision > 0.60**: Of customers predicted to churn, at least 60% actually churn
- **Recall > 0.65**: System identifies at least 65% of actual churners
- **F1-Score > 0.60**: Balanced performance between precision and recall
- **ROC-AUC > 0.75**: Strong discriminative ability (benchmark from literature)

### Business Impact

- **Early Intervention**: Identify at-risk customers months in advance
- **Targeted Campaigns**: Focus retention efforts on high-probability churners
- **Cost Optimization**: Reduce unnecessary retention spending on low-risk customers
- **Revenue Protection**: Prevent revenue loss from customer churn
- **Data-Driven Decisions**: SHAP values guide feature-based retention strategies

## Author

Kazi Akib Javed

## License

This project is for educational purposes.
