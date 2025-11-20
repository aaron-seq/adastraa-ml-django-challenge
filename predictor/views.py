import os
import pandas as pd
import joblib
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
import traceback
from .models import Prediction
import sys

# Add ml_models to path
sys.path.append(str(settings.BASE_DIR / 'ml_models'))


class PredictionService:
    """
    Service class to load model and make predictions.
    """
    _model = None
    _preprocessor = None
    _label_encoders = None
    _scaler = None
    _feature_columns = None
    
    @classmethod
    def load_model_artifacts(cls):
        """
        Load all model artifacts (model, preprocessor, encoders, scaler).
        """
        if cls._model is None:
            try:
                cls._model = joblib.load(settings.MODEL_PATH)
                cls._preprocessor = joblib.load(settings.PREPROCESSOR_PATH)
                cls._label_encoders = joblib.load(settings.LABEL_ENCODERS_PATH)
                cls._scaler = joblib.load(settings.SCALER_PATH)
                cls._feature_columns = joblib.load(settings.FEATURE_COLUMNS_PATH)
                print("Model artifacts loaded successfully!")
            except Exception as e:
                print(f"Error loading model artifacts: {e}")
                raise
    
    @classmethod
    def preprocess_and_predict(cls, df):
        """
        Preprocess data and generate predictions.
        """
        # Load model artifacts if not already loaded
        cls.load_model_artifacts()
        
        # Store original columns for output
        original_df = df.copy()
        
        # Preprocess the data
        df_processed = cls._preprocessor.transform(df)
        
        # Prepare features (same as training)
        X = df_processed.drop(columns=['Ad_Date', 'Ad_ID'], errors='ignore')
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in cls._label_encoders:
                # Handle unseen categories
                X[col] = X[col].astype(str).apply(
                    lambda x: x if x in cls._label_encoders[col].classes_ else 'unknown'
                )
                # Add 'unknown' to classes if not present
                if 'unknown' not in cls._label_encoders[col].classes_:
                    import numpy as np
                    cls._label_encoders[col].classes_ = np.append(
                        cls._label_encoders[col].classes_, 'unknown'
                    )
                X[col] = cls._label_encoders[col].transform(X[col])
        
        # Ensure columns match training data
        for col in cls._feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        # Select only the columns used during training
        X = X[cls._feature_columns]
        
        # Scale features
        X_scaled = pd.DataFrame(
            cls._scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Make predictions
        predictions = cls._model.predict(X_scaled)
        
        # Add predictions to original dataframe
        original_df['Predicted_Sale_Amount'] = predictions
        
        return original_df


def home(request):
    """
    Home page with upload form.
    """
    return render(request, 'predictor/home.html')


def upload_and_predict(request):
    """
    Handle file upload and prediction generation.
    """
    if request.method == 'POST':
        try:
            # Check if file is uploaded
            if 'file' not in request.FILES:
                messages.error(request, 'No file uploaded. Please select a CSV file.')
                return redirect('home')
            
            uploaded_file = request.FILES['file']
            
            # Validate file extension
            if not uploaded_file.name.endswith('.csv'):
                messages.error(request, 'Invalid file type. Please upload a CSV file.')
                return redirect('home')
            
            # Read CSV file
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                messages.error(request, f'Error reading CSV file: {str(e)}')
                return redirect('home')
            
            # Validate required columns
            required_columns = ['Ad_ID', 'Campaign_Name', 'Clicks', 'Impressions', 
                                'Cost', 'Leads', 'Conversions', 'Conversion Rate', 
                                'Ad_Date', 'Location', 'Device', 'Keyword']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                messages.error(request, f'Missing required columns: {", ".join(missing_columns)}')
                return redirect('home')
            
            # Generate predictions
            try:
                result_df = PredictionService.preprocess_and_predict(df)
            except Exception as e:
                messages.error(request, f'Error generating predictions: {str(e)}')
                print(traceback.format_exc())
                return redirect('home')
            
            # Save prediction record
            prediction = Prediction.objects.create(
                num_predictions=len(result_df)
            )
            
            # Save result to session for download
            request.session['prediction_id'] = prediction.id
            request.session['result_csv'] = result_df.to_csv(index=False)
            
            # Show preview
            context = {
                'prediction': prediction,
                'num_predictions': len(result_df),
                'preview_data': result_df.head(10).to_html(classes='table table-striped', index=False),
                'columns': result_df.columns.tolist(),
            }
            
            messages.success(request, f'Successfully generated predictions for {len(result_df)} rows!')
            return render(request, 'predictor/results.html', context)
            
        except Exception as e:
            messages.error(request, f'An unexpected error occurred: {str(e)}')
            print(traceback.format_exc())
            return redirect('home')
    
    return redirect('home')


def download_predictions(request):
    """
    Download predictions as CSV file.
    """
    if 'result_csv' not in request.session:
        messages.error(request, 'No predictions available for download.')
        return redirect('home')
    
    # Get CSV data from session
    csv_data = request.session['result_csv']
    
    # Create HTTP response with CSV
    response = HttpResponse(csv_data, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
    
    return response


def about(request):
    """
    About page with information about the model.
    """
    context = {
        'model_info': {
            'algorithms': ['Linear Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting'],
            'metrics': ['R² Score', 'RMSE', 'MAE'],
            'features': [
                'Click-Through Rate (CTR)',
                'Cost Per Click (CPC)',
                'Cost Per Lead (CPL)',
                'Conversion Rate',
                'Engagement Score',
                'Temporal Features',
            ]
        }
    }
    return render(request, 'predictor/about.html', context)
