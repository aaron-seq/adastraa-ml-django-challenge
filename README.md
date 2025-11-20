# AdAstraa AI - ML + Django Challenge

## 📋 Project Overview

This project is a 24-hour technical challenge for the Machine Learning Engineer (Full Stack) role at AdAstraa AI. The goal is to build a Django web application that predicts `Sale_Amount` from messy marketing campaign data.

## 🎯 Problem Statement

Build a Django web application that:
- Trains a machine learning model to predict `Sale_Amount` using provided dataset
- Hosts the trained model inside the Django backend
- Allows users to upload a `test.csv` file (without `Sale_Amount` column)
- Generates predictions and provides a downloadable CSV with `Predicted_Sale_Amount` column

## 📊 Dataset Overview

The dataset contains raw and intentionally messy marketing campaign data with the following columns:

- **Ad_ID**: Unique ID of the ad campaign
- **Campaign_Name**: Name of the campaign (includes typos and variations)
- **Clicks**: Number of clicks
- **Impressions**: Number of ad impressions
- **Cost**: Total ad cost (inconsistent formatting)
- **Leads**: Leads generated
- **Conversions**: Actual conversions
- **Conversion Rate**: Conversions ÷ Clicks (may be incorrect or missing)
- **Sale_Amount**: Revenue generated (target variable)
- **Ad_Date**: Ad date (mixed formats like YYYY/MM/DD, DD-MM-YY)
- **Location**: City (spelling and casing variations)
- **Device**: Mobile/Desktop/Tablet (mixed casing)
- **Keyword**: Trigger keyword (contains typos)

### ⚠️ Data Quality Issues (Intentional)

The dataset simulates real-world advertising data with:
- Inconsistent date formats
- Typos in names and keywords
- Duplicate and missing rows
- Inconsistent casing in categorical fields
- Incorrect Conversion Rate values

## 🛠️ Tech Stack

- **Backend**: Django 4.x
- **ML Libraries**: scikit-learn, pandas, numpy
- **Data Processing**: pandas, numpy
- **Model**: (To be determined based on performance)
- **Database**: SQLite (default Django)

## 📁 Project Structure

```
adastraa-ml-django-challenge/
├── ml_prediction/           # Django project
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── predictor/               # Django app for predictions
│   ├── migrations/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   └── templates/
├── ml_models/               # ML model and preprocessing
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── model.pkl
│   └── preprocessor.pkl
├── data/                    # Dataset files
│   ├── train_data.csv
│   └── test_data.csv (example)
├── requirements.txt
├── manage.py
└── README.md
```

## 🔧 Setup Instructions

### Prerequisites

- Python 3.8+
- pip
- virtualenv (recommended)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/aaron-seq/adastraa-ml-django-challenge.git
   cd adastraa-ml-django-challenge
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if not already trained)
   ```bash
   python ml_models/train_model.py
   ```

5. **Run database migrations**
   ```bash
   python manage.py migrate
   ```

6. **Start the Django development server**
   ```bash
   python manage.py runserver
   ```

7. **Access the application**
   - Open your browser and navigate to `http://localhost:8000`

## 🧹 Data Cleaning & Preprocessing Approach

### 1. Date Standardization
- Parse mixed date formats (YYYY/MM/DD, DD-MM-YY, etc.) into a standard datetime format
- Extract temporal features: day_of_week, month, quarter, days_since_epoch

### 2. Text Normalization
- Convert all text fields to lowercase for consistency
- Apply fuzzy matching to correct common typos in Campaign_Name, Location, and Keyword
- Standardize categorical values (Device: mobile/desktop/tablet)

### 3. Missing Value Handling
- Identify missing values across all columns
- Use appropriate imputation strategies:
  - Numeric: median/mean imputation
  - Categorical: mode imputation or 'unknown' category

### 4. Duplicate Detection
- Identify and remove duplicate rows based on Ad_ID or combination of features

### 5. Feature Engineering
- Calculate correct Conversion Rate: Conversions / Clicks
- Create derived features:
  - CTR (Click-Through Rate): Clicks / Impressions
  - CPL (Cost Per Lead): Cost / Leads
  - CPC (Cost Per Click): Cost / Clicks
  - ROI: (Sale_Amount - Cost) / Cost

### 6. Outlier Treatment
- Identify outliers using IQR method or Z-score
- Handle outliers through capping, transformation, or removal

### 7. Encoding
- Label Encoding for ordinal categorical variables
- One-Hot Encoding for nominal categorical variables
- Feature scaling for numeric variables

## 🤖 Machine Learning Approach

### Model Selection Strategy

Will evaluate multiple regression algorithms:

1. **Linear Regression** (Baseline)
2. **Random Forest Regressor**
3. **XGBoost**
4. **LightGBM**
5. **Gradient Boosting Regressor**

### Model Evaluation Metrics

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

### Cross-Validation

- 5-fold cross-validation for robust performance estimation
- Train-test split: 80-20

### Feature Importance Analysis

- Analyze feature importance to understand key predictors
- Document insights for business interpretation

## 🌐 Django Application Features

### Core Functionality

1. **File Upload Interface**
   - Clean, user-friendly upload form
   - Validation for CSV format
   - Error handling for invalid files

2. **Data Processing Pipeline**
   - Apply same preprocessing as training
   - Handle edge cases gracefully

3. **Prediction Generation**
   - Load pre-trained model
   - Generate predictions for uploaded data
   - Add `Predicted_Sale_Amount` column

4. **Download Results**
   - Generate downloadable CSV with predictions
   - Preserve all original columns

### Optional Features (if time permits)

- Basic data visualization dashboard
- Feature importance charts
- Model performance metrics display
- Prediction confidence intervals

## 📦 Deliverables

- ✅ Public GitHub repository with complete code
- ✅ Working Django application
- ✅ Trained ML model with preprocessing pipeline
- ✅ Comprehensive README with setup instructions
- ✅ requirements.txt with all dependencies
- ✅ Documentation of data cleaning approach
- ✅ Documentation of modeling decisions

## 🚀 Deployment (Optional)

If time permits, deploy to:
- Heroku
- Railway
- Render
- PythonAnywhere

## 🔮 Future Improvements

### With More Time:

1. **Model Enhancements**
   - Hyperparameter tuning using GridSearchCV/RandomizedSearchCV
   - Ensemble methods combining multiple models
   - Deep learning approaches (Neural Networks)

2. **Feature Engineering**
   - More sophisticated temporal features
   - Interaction features between variables
   - Text embeddings for campaign names and keywords

3. **UI/UX**
   - React/Vue.js frontend
   - Real-time prediction progress tracking
   - Interactive visualizations with Plotly/D3.js

4. **Error Handling**
   - More robust validation
   - Detailed error messages
   - Logging system

### Production Scaling:

1. **Architecture**
   - Microservices architecture
   - Separate model serving with TensorFlow Serving or FastAPI
   - Load balancing and auto-scaling

2. **Database**
   - PostgreSQL for production
   - Redis for caching
   - Database connection pooling

3. **Model Management**
   - MLflow for experiment tracking
   - Model versioning and A/B testing
   - Automated retraining pipeline

4. **Monitoring**
   - Model performance monitoring
   - Data drift detection
   - Real-time alerting

5. **Security**
   - API authentication and rate limiting
   - Input sanitization
   - HTTPS and secure file handling

6. **CI/CD**
   - Automated testing
   - Docker containerization
   - Kubernetes orchestration

## 📝 Assumptions & Limitations

### Assumptions:

1. All test data follows the same distribution as training data
2. The same data quality issues exist in test data as training data
3. Missing values in test data can be handled with same imputation strategy
4. The relationship between features and target remains stable over time

### Limitations:

1. Model trained on limited dataset size
2. No real-time model updates
3. Limited error handling for extreme edge cases
4. Single model approach (no ensemble in v1)
5. Basic UI without advanced visualizations

## 👨‍💻 Author

**Aaron Sequeira**
- GitHub: [@aaron-seq](https://github.com/aaron-seq)
- Email: aaronsequeira12@gmail.com

## 📄 License

This project is created for the AdAstraa AI technical assessment.

## 🙏 Acknowledgments

- AdAstraa AI for the challenging problem statement
- The eCommerce and digital marketing domain for real-world data scenarios

---

**Submission Details**
- **Challenge**: AdAstraa AI – 24h ML + Django Challenge
- **Role**: Machine Learning Engineer (Full Stack)
- **Timeline**: 24 hours
- **Contact**: mayur@adastraa.ai