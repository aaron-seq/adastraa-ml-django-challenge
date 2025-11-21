import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor


class ModelTrainer:
    """
    Train and evaluate multiple regression models for Sale_Amount prediction.
    """
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
    
    def clean_sale_amount(self, value):
        """
        Clean Sale_Amount column by removing $ and other non-numeric characters.
        """
        if pd.isna(value):
            return np.nan
        # Remove currency symbols, commas, spaces
        value_str = str(value)
        value_str = re.sub(r'[^0-9.]', '', value_str)
        try:
            return float(value_str)
        except:
            return np.nan
        
    def load_data(self, filepath):
        """
        Load training data from CSV file.
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def prepare_features(self, df, is_training=True):
        """
        Prepare features for model training.
        """
        # Separate features and target
        if 'Sale_Amount' in df.columns:
            # Clean Sale_Amount column
            y = df['Sale_Amount'].apply(self.clean_sale_amount)
            X = df.drop(columns=['Sale_Amount', 'Ad_Date', 'Ad_ID'], errors='ignore')
        else:
            X = df.drop(columns=['Ad_Date', 'Ad_ID'], errors='ignore')
            y = None
        
        # Store original columns for test data
        if is_training:
            self.feature_columns = X.columns.tolist()
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    X[col] = X[col].astype(str).apply(
                        lambda x: x if x in self.label_encoders[col].classes_ else 'unknown'
                    )
                    # Add 'unknown' to classes if not present
                    if 'unknown' not in self.label_encoders[col].classes_:
                        self.label_encoders[col].classes_ = np.append(
                            self.label_encoders[col].classes_, 'unknown'
                        )
                    X[col] = self.label_encoders[col].transform(X[col])
        
        # Scale numerical features
        if is_training:
            X = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X, y
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Evaluate model performance with multiple metrics.
        """
        # Train the model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                     scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(f"Train R² Score: {train_r2:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Train MAE: {train_mae:.2f}")
        print(f"Test MAE: {test_mae:.2f}")
        print(f"Cross-Val R² Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"{'='*60}\n")
        
        return {
            'model': model,
            'model_name': model_name,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
    
    def train_multiple_models(self, X_train, X_test, y_train, y_test):
        """
        Train and compare multiple regression models.
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        results = []
        
        for model_name, model in models.items():
            result = self.evaluate_model(
                model, X_train, X_test, y_train, y_test, model_name
            )
            results.append(result)
        
        # Find best model based on test R² score
        best_result = max(results, key=lambda x: x['test_r2'])
        
        print("\n" + "="*60)
        print(f"BEST MODEL: {best_result['model_name']}")
        print(f"Test R² Score: {best_result['test_r2']:.4f}")
        print(f"Test RMSE: {best_result['test_rmse']:.2f}")
        print("="*60 + "\n")
        
        return best_result
    
    def fine_tune_model(self, X_train, y_train, base_model_name):
        """
        Fine-tune the best model with hyperparameter optimization.
        """
        print(f"Fine-tuning {base_model_name}...")
        
        if 'XGBoost' in base_model_name:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5]
            }
            model = XGBRegressor(random_state=42, n_jobs=-1)
            
        elif 'LightGBM' in base_model_name:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 70]
            }
            model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
            
        elif 'Random Forest' in base_model_name:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        else:
            # Return the model without tuning
            return None
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV R² score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_model_artifacts(self, model, output_dir='ml_models'):
        """
        Save trained model and preprocessing objects.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, 'model.pkl')
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save preprocessor
        preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"Preprocessor saved to {preprocessor_path}")
        
        # Save label encoders
        encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
        joblib.dump(self.label_encoders, encoders_path)
        print(f"Label encoders saved to {encoders_path}")
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
        # Save feature columns
        features_path = os.path.join(output_dir, 'feature_columns.pkl')
        joblib.dump(self.feature_columns, features_path)
        print(f"Feature columns saved to {features_path}")
    
    def train(self, data_filepath, test_size=0.2, fine_tune=False):
        """
        Complete training pipeline.
        """
        # Load data
        df = self.load_data(data_filepath)
        
        # Preprocess data
        df_processed = self.preprocessor.fit_transform(df)
        
        # Prepare features
        X, y = self.prepare_features(df_processed, is_training=True)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}\n")
        
        # Train multiple models
        best_result = self.train_multiple_models(X_train, X_test, y_train, y_test)
        
        # Fine-tune if requested
        if fine_tune:
            tuned_model = self.fine_tune_model(
                X_train, y_train, best_result['model_name']
            )
            if tuned_model:
                best_result['model'] = tuned_model
        
        # Save model and artifacts
        self.save_model_artifacts(best_result['model'])
        
        return best_result


if __name__ == '__main__':
    # Train the model
    trainer = ModelTrainer()
    
    # Update this path to your training data
    data_path = 'data/train_data.csv'
    
    if os.path.exists(data_path):
        result = trainer.train(data_path, fine_tune=False)
        print("\nTraining completed successfully!")
    else:
        print(f"Error: Data file not found at {data_path}")
        print("Please download the dataset and place it in the 'data' directory.")
