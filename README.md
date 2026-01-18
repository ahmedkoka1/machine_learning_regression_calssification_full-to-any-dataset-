# Machine Learning Classification and Regression App

A comprehensive Streamlit-based web application for performing machine learning tasks on any dataset. This app supports both classification and regression problems, with automated data preprocessing, feature engineering, and model training.

## Features

### Data Upload and Exploration
- Upload CSV files through an intuitive interface
- Display dataset overview with customizable row and column selection
- Interactive data visualization with scatter plots and histograms

### Automated Data Preprocessing
- **Outlier Handling**: Clamps outliers using IQR method
- **Null Value Management**: Removes columns with >80% nulls, imputes remaining nulls
- **Redundant Feature Removal**: Drops features with >80% single value
- **Correlation Analysis**: Removes highly correlated features (threshold: 0.8)
- **Skewness Correction**: Applies log transformation for skewed numerical features
- **Feature Scaling**: Uses StandardScaler or MinMaxScaler based on normality tests
- **Categorical Encoding**: Supports OneHot, Label, Ordinal, and Binary encoding

### Machine Learning Models

#### Regression Models
- Linear Regression
- XGBoost Regressor

#### Classification Models
- K-Nearest Neighbors (KNN)
- Logistic Regression
- XGBoost Classifier
- Support Vector Machine (SVM)

### Advanced Features
- Handles imbalanced datasets using SMOTETomek oversampling
- Automatic feature selection and engineering
- Model performance metrics (RMSE, MSE, R² for regression; Accuracy, Precision, Recall, F1 for classification)
- Caching for efficient data loading

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd machine_learning_full
```

2. Install required packages:
```bash
pip install streamlit pandas plotly scikit-learn xgboost imbalanced-learn scipy numpy
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run "machine_learning_classifcation and regression _anydataset.py"
```

2. Open your browser and navigate to the provided local URL (usually http://localhost:8501)

3. Upload your CSV dataset

4. Explore the data using the visualization tabs

5. Select your target column

6. Choose between regression or classification tasks

7. Select and train your preferred model

8. View performance metrics and results

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly
- Imbalanced-learn
- SciPy

## Project Structure

```
machine_learning_full/
│
├── machine_learning_classifcation and regression _anydataset.py  # Main application
├── README.md                                                    # Project documentation
└── uploaded_files/                                              # Directory for uploaded datasets (created automatically)
```

## Data Preprocessing Pipeline

1. **Data Loading**: CSV upload with caching
2. **Initial Exploration**: Display and visualization
3. **Outlier Clamping**: IQR-based outlier handling
4. **Null Handling**: Column removal and imputation
5. **Redundancy Check**: Remove features with high single-value ratios
6. **Correlation Analysis**: Feature selection based on correlation
7. **Skewness Correction**: Log transformation for skewed features
8. **Normality Testing**: Shapiro-Wilk test for scaling method selection
9. **Feature Scaling**: Standard or MinMax scaling
10. **Categorical Encoding**: Multiple encoding strategies
11. **Train-Test Split**: 80-20 split with optional stratification
12. **Imbalance Handling**: SMOTETomek for classification tasks
13. **Model Training and Evaluation**

## Model Configuration

### XGBoost Regressor
- n_estimators: 150
- learning_rate: 0.02
- max_depth: 5

### XGBoost Classifier
- n_estimators: 100
- learning_rate: 0.05
- max_depth: 5
- eval_metric: logloss

### SVM (Balanced)
- kernel: rbf
- C: 0.5
- gamma: scale

### SVM (Imbalanced)
- kernel: poly
- degree: 3
- C: 1
- gamma: scale

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit for the web interface
- Utilizes scikit-learn, XGBoost, and other ML libraries
- Data visualization powered by Plotly
