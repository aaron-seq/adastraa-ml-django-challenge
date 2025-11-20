# Setup Guide - AdAstraa AI ML + Django Challenge

This guide will walk you through setting up and running the Django application locally.

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **Git** - [Download Git](https://git-scm.com/downloads/)
- **pip** (usually comes with Python)
- **virtualenv** (recommended) - Install with: `pip install virtualenv`

## 🚀 Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/aaron-seq/adastraa-ml-django-challenge.git
cd adastraa-ml-django-challenge
```

### Step 2: Create and Activate Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt, indicating the virtual environment is active.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- Django 4.2.7
- scikit-learn, XGBoost, LightGBM
- pandas, numpy
- And other dependencies

### Step 4: Download the Dataset

1. Download the training dataset from the Google Drive link provided in the challenge email
2. Place the CSV file in the `data/` directory
3. Rename it to `train_data.csv`

Your directory structure should look like:
```
adastraa-ml-django-challenge/
├── data/
│   ├── .gitkeep
│   └── train_data.csv  # <-- Place your dataset here
├── ml_models/
├── predictor/
...
```

### Step 5: Train the Model

```bash
python ml_models/train_model.py
```

This will:
- Load and preprocess the training data
- Train multiple ML models (Linear Regression, Random Forest, XGBoost, LightGBM, Gradient Boosting)
- Evaluate each model using cross-validation
- Select the best performing model
- Save the model and preprocessing artifacts to `ml_models/` directory

**Expected output files:**
- `ml_models/model.pkl` - Trained model
- `ml_models/preprocessor.pkl` - Data preprocessor
- `ml_models/label_encoders.pkl` - Categorical encoders
- `ml_models/scaler.pkl` - Feature scaler
- `ml_models/feature_columns.pkl` - Feature column names

**Note:** Training may take 5-15 minutes depending on your system and dataset size.

### Step 6: Run Database Migrations

```bash
python manage.py migrate
```

This creates the SQLite database and necessary tables.

### Step 7: Create Media and Static Directories

```bash
# On Windows
mkdir media uploads results static

# On macOS/Linux
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

You should see output like:
```
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

### Step 10: Access the Application

Open your web browser and navigate to:
```
http://localhost:8000
```

You should see the home page with the file upload interface!

## 📝 Usage Instructions

### Uploading Test Data

1. **Navigate to Home Page**: Go to `http://localhost:8000`

2. **Prepare Test CSV**: Your test CSV file should have these columns (without `Sale_Amount`):
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
   - Preview the first 10 rows with predictions
   - See summary statistics
   - Download the complete CSV with `Predicted_Sale_Amount` column

### Creating Test Data

If you need to create test data from the training set:

```python
import pandas as pd

# Load training data
df = pd.read_csv('data/train_data.csv')

# Remove Sale_Amount column
test_df = df.drop(columns=['Sale_Amount'])

# Save first 100 rows as test data
test_df.head(100).to_csv('data/test_data.csv', index=False)
```

## 🐛 Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
# Make sure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: "Model file not found"

**Solution:**
```bash
# Train the model first
python ml_models/train_model.py
```

### Issue: Port 8000 already in use

**Solution:**
```bash
# Use a different port
python manage.py runserver 8080
```

### Issue: Database errors

**Solution:**
```bash
# Delete database and recreate
del db.sqlite3  # On Windows
rm db.sqlite3   # On macOS/Linux

# Run migrations again
python manage.py migrate
```

### Issue: Static files not loading

**Solution:**
```bash
# Collect static files
python manage.py collectstatic --noinput

# Make sure DEBUG=True in settings.py for development
```

## 🛠️ Development Tips

### Running in Debug Mode

The application runs in debug mode by default. To see detailed error messages:
- Check the console where you ran `python manage.py runserver`
- Error details will be displayed in the browser

### Viewing Database

To explore the database:

```bash
python manage.py dbshell
```

Or use a GUI tool like [DB Browser for SQLite](https://sqlitebrowser.org/)

### Creating Superuser (Admin Access)

```bash
python manage.py createsuperuser
```

Then access admin at: `http://localhost:8000/admin`

### Running Tests

Create and run tests:

```bash
python manage.py test
```

## 📊 Model Performance

After training, you'll see performance metrics like:

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

## 🚀 Deployment (Optional)

For production deployment, consider:

### Heroku

```bash
# Install Heroku CLI
# Login to Heroku
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
2. Connect Railway to your repository
3. Add environment variables
4. Deploy automatically

### Render

1. Create new Web Service
2. Connect GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn ml_prediction.wsgi:application`

## 📝 Environment Variables (Production)

For production, create a `.env` file:

```env
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
```

Update `settings.py` to use these variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY', 'default-dev-key')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', '*').split(',')
```

## ❓ Getting Help

If you encounter issues:

1. Check this setup guide
2. Review error messages carefully
3. Check Django logs in the console
4. Review the code in `predictor/views.py` for debugging

## 🎯 Next Steps

1. ✅ Complete setup following this guide
2. ✅ Train the model with your data
3. ✅ Test the upload and prediction functionality
4. ✅ Review the code and documentation
5. ✅ (Optional) Deploy to a cloud platform
6. ✅ Submit your GitHub repository link

---

**Good luck with the challenge!** 🚀

For questions or issues, contact: aaronsequeira12@gmail.com
