# Setup Guide - AdAstraa AI ML + Django Challenge

This guide provides step-by-step instructions for setting up and running the Django application locally.

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher - [Download](https://www.python.org/downloads/)
- Git - [Download](https://git-scm.com/downloads/)
- pip (included with Python)
- virtualenv (recommended) - Install with: `pip install virtualenv`

## Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/aaron-seq/adastraa-ml-django-challenge.git
cd adastraa-ml-django-challenge
```

### Step 2: Create and Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Django 4.2.7
- scikit-learn, XGBoost, LightGBM
- pandas, numpy
- Other required packages

### Step 4: Download the Dataset

1. Download the training dataset from the Google Drive link in the challenge email
2. Place the CSV file in the `data/` directory
3. Rename it to `train_data.csv`

Expected directory structure:
```
adastraa-ml-django-challenge/
├── data/
│   ├── .gitkeep
│   └── train_data.csv
├── ml_models/
├── predictor/
...
```

### Step 5: Train the Model

```bash
python ml_models/train_model.py
```

This process:
- Loads and preprocesses the training data
- Trains multiple ML models (Linear Regression, Random Forest, XGBoost, LightGBM, Gradient Boosting)
- Evaluates models using cross-validation
- Selects the best performer
- Saves model and preprocessing artifacts to `ml_models/`

**Output files:**
- `ml_models/model.pkl` - Trained model
- `ml_models/preprocessor.pkl` - Data preprocessor
- `ml_models/label_encoders.pkl` - Categorical encoders
- `ml_models/scaler.pkl` - Feature scaler
- `ml_models/feature_columns.pkl` - Feature column names

**Note:** Training may take 5-15 minutes depending on system specifications and dataset size.

### Step 6: Run Database Migrations

```bash
python manage.py migrate
```

This creates the SQLite database and necessary tables.

### Step 7: Create Media and Static Directories

**Windows:**
```bash
mkdir media uploads results static
```

**macOS/Linux:**
```bash
mkdir -p media/uploads media/results static
```

### Step 8: Collect Static Files (Optional)

```bash
python manage.py collectstatic --noinput
```

### Step 9: Start the Development Server

```bash
python manage.py runserver
```

Expected output:
```
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

### Step 10: Access the Application

Open your browser and navigate to:
```
http://localhost:8000
```

You should see the home page with the file upload interface.

## Usage Instructions

### Uploading Test Data

1. **Navigate to Home Page**: `http://localhost:8000`

2. **Prepare Test CSV**: File should contain these columns (without Sale_Amount):
   - Ad_ID
   - Campaign_Name
   - Clicks
   - Impressions
   - Cost
   - Leads
   - Conversions
   - Conversion Rate
   - Ad_Date
   - Location
   - Device
   - Keyword

3. **Upload File**: 
   - Click "Choose File" or drag and drop your CSV
   - Click "Generate Predictions"

4. **View Results**:
   - Preview first 10 rows with predictions
   - See summary statistics
   - Download complete CSV with `Predicted_Sale_Amount` column

### Creating Test Data

To create test data from the training set:

```python
import pandas as pd

# Load training data
df = pd.read_csv('data/train_data.csv')

# Remove Sale_Amount column
test_df = df.drop(columns=['Sale_Amount'])

# Save first 100 rows as test data
test_df.head(100).to_csv('data/test_data.csv', index=False)
```

## Troubleshooting

### "Module not found" errors

**Solution:**
```bash
# Ensure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### "Model file not found"

**Solution:**
```bash
# Train the model first
python ml_models/train_model.py
```

### Port 8000 already in use

**Solution:**
```bash
# Use a different port
python manage.py runserver 8080
```

### Database errors

**Solution:**
```bash
# Delete database and recreate
del db.sqlite3  # Windows
rm db.sqlite3   # macOS/Linux

# Run migrations again
python manage.py migrate
```

### Static files not loading

**Solution:**
```bash
# Collect static files
python manage.py collectstatic --noinput

# Verify DEBUG=True in settings.py for development
```

## Development Tips

### Debug Mode

The application runs in debug mode by default. Error details appear in:
- Console where you ran `python manage.py runserver`
- Browser (when DEBUG=True)

### Database Exploration

Command line:
```bash
python manage.py dbshell
```

GUI tool: [DB Browser for SQLite](https://sqlitebrowser.org/)

### Admin Access

Create superuser:
```bash
python manage.py createsuperuser
```

Access admin interface: `http://localhost:8000/admin`

### Running Tests

```bash
python manage.py test
```

## Model Performance

After training, performance metrics are displayed:

```
Model: XGBoost
============================================================
Train R² Score: 0.9456
Test R² Score: 0.8923
Train RMSE: 1234.56
Test RMSE: 1456.78
Train MAE: 987.65
Test MAE: 1123.45
Cross-Val R² Score: 0.8845 (+/- 0.0234)
============================================================
```

## Deployment (Optional)

### Heroku

```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create your-app-name

# Add Procfile
echo "web: gunicorn ml_prediction.wsgi" > Procfile

# Deploy
git push heroku main
```

### Railway

1. Push code to GitHub
2. Connect Railway to repository
3. Add environment variables
4. Deploy automatically

### Render

1. Create new Web Service
2. Connect GitHub repository
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn ml_prediction.wsgi:application`

## Environment Variables (Production)

Create a `.env` file:

```env
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
```

Update `settings.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY', 'default-dev-key')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', '*').split(',')
```

## Getting Help

If you encounter issues:

1. Review this setup guide
2. Check error messages in console
3. Verify Django logs
4. Examine code in `predictor/views.py` for debugging

## Next Steps

1. Complete setup following this guide
2. Train the model with your data
3. Test upload and prediction functionality
4. Review code and documentation
5. (Optional) Deploy to cloud platform
6. Submit GitHub repository link

---

For questions or issues, contact: aaronsequeira12@gmail.com
