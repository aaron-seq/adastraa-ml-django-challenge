# AdAstraa AI - ML + Django Challenge

## Project Overview

This project is a 24-hour technical challenge for the Machine Learning Engineer (Full Stack) role at AdAstraa AI. The goal is to build a Django web application that predicts Sale_Amount from messy marketing campaign data.

## Problem Statement

Build a Django web application that:
- Trains a machine learning model to predict Sale_Amount using the provided dataset
- Hosts the trained model inside the Django backend
- Allows users to upload a test.csv file (without Sale_Amount column)
- Generates predictions and provides a downloadable CSV with Predicted_Sale_Amount column

## Dataset Overview

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

### Data Quality Issues (Intentional)

The dataset simulates real-world advertising data with:
- Inconsistent date formats
- Typos in names and keywords
- Duplicate and missing rows
- Inconsistent casing in categorical fields
- Incorrect Conversion Rate values

## Tech Stack

- **Backend**: Django 4.x
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Database**: SQLite (development), PostgreSQL (production-ready)
- **Frontend**: Bootstrap 5, vanilla JavaScript

## Project Structure

```
adastraa-ml-django-challenge/
├── ml_prediction/           # Django project
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── predictor/               # Django app for predictions
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
│   └── train_data.csv
├── requirements.txt
├── manage.py
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip and virtualenv

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aaron-seq/adastraa-ml-django-challenge.git
   cd adastraa-ml-django-challenge
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the training dataset in `data/train_data.csv`

5. Train the model:
   ```bash
   python ml_models/train_model.py
   ```

6. Run database migrations:
   ```bash
   python manage.py migrate
   ```

7. Start the development server:
   ```bash
   python manage.py runserver
   ```

8. Access the application at `http://localhost:8000`

For detailed setup instructions and troubleshooting, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

## Data Cleaning & Preprocessing

### Date Standardization
- Parse multiple date formats (YYYY/MM/DD, DD-MM-YY, etc.) into standard datetime
- Extract temporal features: day_of_week, month, quarter, days_since_start

### Text Normalization
- Lowercase conversion for all text fields
- Fuzzy matching to correct typos in Campaign_Name, Location, and Keyword
- Standardize categorical values (Device: mobile/desktop/tablet)

### Missing Value Handling
- Numeric columns: median imputation
- Categorical columns: mode imputation or 'unknown' category

### Duplicate Removal
- Remove exact duplicates
- Deduplicate based on Ad_ID

### Feature Engineering
- Click-Through Rate (CTR): Clicks / Impressions
- Cost Per Click (CPC): Cost / Clicks
- Cost Per Lead (CPL): Cost / Leads
- Corrected Conversion Rate: Conversions / Clicks
- Lead-to-Conversion Rate: Conversions / Leads
- Cost Per Conversion: Cost / Conversions
- Engagement Score: composite metric combining CTR and conversion rate

### Outlier Treatment
- IQR method for outlier detection
- Capping extreme values to maintain data integrity

### Encoding and Scaling
- Label encoding for categorical variables
- Standard scaling for numeric features

## Machine Learning Approach

### Models Evaluated

1. Linear Regression (baseline)
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. XGBoost
5. LightGBM

### Evaluation Metrics

- R² Score (coefficient of determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

### Validation Strategy

- 5-fold cross-validation for robust performance estimation
- 80-20 train-test split
- Best model selected based on test R² score

### Model Selection Rationale

The training script evaluates all models using cross-validation and selects the best performer based on test set R² score. Tree-based ensemble methods (Random Forest, XGBoost, LightGBM) typically perform well on this type of tabular data with mixed feature types.

## Django Application Features

### Core Functionality

1. **File Upload**: User-friendly interface for CSV upload with drag-and-drop support
2. **Validation**: Checks for required columns and CSV format
3. **Preprocessing**: Applies the same preprocessing pipeline used during training
4. **Prediction**: Generates Sale_Amount predictions using the trained model
5. **Download**: Provides downloadable CSV with all original columns plus Predicted_Sale_Amount
6. **Error Handling**: Comprehensive error messages and validation feedback

### User Interface

- Responsive design with Bootstrap 5
- Clean, modern aesthetics
- Interactive upload area with visual feedback
- Results preview with summary statistics
- About page documenting the technical approach

## Deliverables

- Public GitHub repository with complete source code
- Working Django web application
- Trained ML model with preprocessing pipeline
- Comprehensive documentation (README, SETUP_GUIDE, SUBMISSION)
- requirements.txt with all dependencies
- Detailed explanation of data cleaning and modeling approach

## Future Improvements

### Model Enhancements
- Hyperparameter tuning using GridSearchCV or Bayesian optimization
- Ensemble methods (stacking, blending)
- Deep learning approaches for complex feature interactions
- Automated feature selection

### Production Scaling

**Architecture**:
- Microservices architecture with separate model serving
- Load balancing and auto-scaling
- Asynchronous task processing with Celery

**Database**:
- PostgreSQL for production
- Redis for caching and session management
- Connection pooling for performance

**Model Management**:
- MLflow for experiment tracking and model versioning
- A/B testing framework
- Automated retraining pipeline with drift detection

**Monitoring**:
- Model performance monitoring
- Data drift detection
- Real-time alerting and logging

**Security**:
- API authentication and rate limiting
- Input sanitization and validation
- HTTPS and secure file handling

**CI/CD**:
- Automated testing pipeline
- Docker containerization
- Kubernetes orchestration

## Assumptions & Limitations

### Assumptions

1. Test data follows similar distribution as training data
2. Same data quality issues present in test data
3. Feature relationships remain stable over time
4. Missing values in test data can be imputed using training statistics

### Current Limitations

1. Single model approach (no ensemble in v1)
2. No real-time model retraining
3. Limited handling of extreme outliers or novel categories
4. SQLite database (not suitable for production scale)
5. Synchronous request processing (may be slow for large files)

## Author

**Aaron Sequeira**  
GitHub: [@aaron-seq](https://github.com/aaron-seq)  
Email: aaronsequeira12@gmail.com

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was created for the AdAstraa AI technical assessment. Thanks to AdAstraa AI for the challenging and realistic problem statement.

---

**Submission Details**  
Challenge: AdAstraa AI – 24h ML + Django Challenge  
Role: Machine Learning Engineer (Full Stack)  
Timeline: 24 hours  
Contact: mayur@adastraa.ai
