import pandas as pd
import numpy as np
from datetime import datetime
from fuzzywuzzy import process
import re
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for marketing campaign data.
    Handles data cleaning, feature engineering, and transformation.
    """
    
    def __init__(self):
        self.location_mapping = {}
        self.device_mapping = {}
        self.keyword_mapping = {}
        self.campaign_mapping = {}
        self.numeric_imputers = {}
        self.categorical_imputers = {}
        
    def parse_date(self, date_str):
        """
        Parse dates from multiple formats into standardized datetime.
        Handles: YYYY/MM/DD, DD-MM-YY, YYYY-MM-DD, etc.
        """
        if pd.isna(date_str):
            return None
            
        date_str = str(date_str).strip()
        
        # Common date formats
        formats = [
            '%Y/%m/%d', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y',
            '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%y', '%d-%m-%y',
            '%Y%m%d', '%d.%m.%Y', '%d.%m.%y'
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
                
        # Try pandas auto-parsing as last resort
        try:
            return pd.to_datetime(date_str)
        except:
            return None
    
    def clean_text(self, text):
        """
        Clean and normalize text data.
        """
        if pd.isna(text):
            return 'unknown'
        return str(text).lower().strip()
    
    def fuzzy_match_categories(self, series, unique_threshold=10):
        """
        Use fuzzy matching to standardize categorical values with typos.
        """
        if len(series.unique()) <= unique_threshold:
            return series
            
        # Get cleaned values
        cleaned = series.apply(self.clean_text)
        unique_vals = cleaned.unique()
        
        # Create mapping using fuzzy matching
        mapping = {}
        processed = set()
        
        for val in unique_vals:
            if val in processed or val == 'unknown':
                continue
                
            # Find similar values
            matches = process.extract(val, unique_vals, limit=5)
            similar = [m[0] for m in matches if m[1] > 85]  # 85% similarity
            
            # Map all similar values to the most common one
            most_common = max(similar, key=lambda x: (cleaned == x).sum())
            for s in similar:
                mapping[s] = most_common
                processed.add(s)
        
        return cleaned.map(lambda x: mapping.get(x, x))
    
    def clean_cost(self, cost_str):
        """
        Clean cost column with inconsistent formatting.
        """
        if pd.isna(cost_str):
            return np.nan
            
        # Remove currency symbols, commas, spaces
        cost_str = str(cost_str)
        cost_str = re.sub(r'[^0-9.]', '', cost_str)
        
        try:
            return float(cost_str)
        except:
            return np.nan
    
    def extract_temporal_features(self, df, date_col='Ad_Date'):
        """
        Extract temporal features from date column.
        """
        df[date_col] = pd.to_datetime(df[date_col])
        
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['day_of_month'] = df[date_col].dt.day
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        
        # Days since earliest date (for trend)
        min_date = df[date_col].min()
        df['days_since_start'] = (df[date_col] - min_date).dt.days
        
        return df
    
    def engineer_features(self, df):
        """
        Create derived features for better predictions.
        """
        # Click-Through Rate
        df['ctr'] = np.where(df['Impressions'] > 0, 
                             df['Clicks'] / df['Impressions'], 0)
        
        # Cost Per Click
        df['cpc'] = np.where(df['Clicks'] > 0, 
                             df['Cost'] / df['Clicks'], 0)
        
        # Cost Per Lead
        df['cpl'] = np.where(df['Leads'] > 0, 
                             df['Cost'] / df['Leads'], 0)
        
        # Recalculate Conversion Rate (override incorrect values)
        df['conversion_rate_corrected'] = np.where(df['Clicks'] > 0,
                                                    df['Conversions'] / df['Clicks'], 0)
        
        # Lead-to-Conversion Rate
        df['lead_conversion_rate'] = np.where(df['Leads'] > 0,
                                               df['Conversions'] / df['Leads'], 0)
        
        # Cost Per Conversion
        df['cost_per_conversion'] = np.where(df['Conversions'] > 0,
                                              df['Cost'] / df['Conversions'], 0)
        
        # Engagement Score (composite metric)
        df['engagement_score'] = (df['Clicks'] / (df['Impressions'] + 1)) * \
                                 (df['Conversions'] / (df['Clicks'] + 1))
        
        # Replace inf values with 0
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def handle_missing_values(self, df, is_training=True):
        """
        Handle missing values with appropriate imputation strategies.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Numeric columns - use median
        for col in numeric_cols:
            if df[col].isna().any():
                if is_training:
                    self.numeric_imputers[col] = df[col].median()
                
                fill_value = self.numeric_imputers.get(col, 0)
                df[col].fillna(fill_value, inplace=True)
        
        # Categorical columns - use mode or 'unknown'
        for col in categorical_cols:
            if df[col].isna().any():
                if is_training:
                    mode_val = df[col].mode()
                    self.categorical_imputers[col] = mode_val[0] if len(mode_val) > 0 else 'unknown'
                
                fill_value = self.categorical_imputers.get(col, 'unknown')
                df[col].fillna(fill_value, inplace=True)
        
        return df
    
    def remove_duplicates(self, df):
        """
        Remove duplicate rows based on Ad_ID or feature combinations.
        """
        # First, drop exact duplicates
        df = df.drop_duplicates()
        
        # Drop duplicates based on Ad_ID if it exists
        if 'Ad_ID' in df.columns:
            df = df.drop_duplicates(subset=['Ad_ID'], keep='first')
        
        return df
    
    def handle_outliers(self, df, columns=None, method='iqr'):
        """
        Handle outliers using IQR method or Z-score.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in df.columns:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    
        return df
    
    def fit_transform(self, df):
        """
        Complete preprocessing pipeline for training data.
        """
        print("Starting data preprocessing...")
        
        # Make a copy
        df = df.copy()
        
        # 1. Remove duplicates
        print(f"Initial shape: {df.shape}")
        df = self.remove_duplicates(df)
        print(f"After duplicate removal: {df.shape}")
        
        # 2. Parse and clean dates
        print("Parsing dates...")
        df['Ad_Date'] = df['Ad_Date'].apply(self.parse_date)
        df = df.dropna(subset=['Ad_Date'])  # Drop rows with unparseable dates
        
        # 3. Extract temporal features
        print("Extracting temporal features...")
        df = self.extract_temporal_features(df)
        
        # 4. Clean cost column
        print("Cleaning cost data...")
        df['Cost'] = df['Cost'].apply(self.clean_cost)
        
        # 5. Normalize text columns
        print("Normalizing text columns...")
        text_cols = ['Campaign_Name', 'Location', 'Device', 'Keyword']
        for col in text_cols:
            if col in df.columns:
                df[col] = self.fuzzy_match_categories(df[col])
        
        # 6. Handle missing values
        print("Handling missing values...")
        df = self.handle_missing_values(df, is_training=True)
        
        # 7. Engineer features
        print("Engineering features...")
        df = self.engineer_features(df)
        
        # 8. Handle outliers (only for numeric columns, not target)
        print("Handling outliers...")
        outlier_cols = ['Clicks', 'Impressions', 'Cost', 'Leads', 'Conversions']
        df = self.handle_outliers(df, columns=outlier_cols)
        
        print(f"Final shape: {df.shape}")
        print("Preprocessing complete!")
        
        return df
    
    def transform(self, df):
        """
        Apply preprocessing to test data using fitted parameters.
        """
        print("Applying preprocessing to test data...")
        
        # Make a copy
        df = df.copy()
        
        # 1. Parse and clean dates
        df['Ad_Date'] = df['Ad_Date'].apply(self.parse_date)
        df = df.dropna(subset=['Ad_Date'])
        
        # 2. Extract temporal features
        df = self.extract_temporal_features(df)
        
        # 3. Clean cost column
        df['Cost'] = df['Cost'].apply(self.clean_cost)
        
        # 4. Normalize text columns
        text_cols = ['Campaign_Name', 'Location', 'Device', 'Keyword']
        for col in text_cols:
            if col in df.columns:
                df[col] = self.fuzzy_match_categories(df[col])
        
        # 5. Handle missing values (using training statistics)
        df = self.handle_missing_values(df, is_training=False)
        
        # 6. Engineer features
        df = self.engineer_features(df)
        
        print("Test data preprocessing complete!")
        
        return df
