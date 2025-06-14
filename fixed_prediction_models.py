# fixed_prediction_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
import mysql.connector
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CommodityPredictor:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None
        self.models = {}
        self.feature_columns = []

    def connect_db(self):
        """Koneksi ke database"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            print("✅ Database connection successful!")
            return True
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return False

    def check_available_tables(self):
        """Check what tables are available"""
        if not self.connection:
            self.connect_db()
        
        cursor = self.connection.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        available_tables = [table[0] for table in tables]
        print(f"📋 Available tables: {available_tables}")
        cursor.close()
        return available_tables

    def load_data(self):
        """Load data from the most appropriate table"""
        if not self.connection:
            self.connect_db()

        available_tables = self.check_available_tables()
        
        # Try to load from processed_gold_data first
        if 'processed_gold_data' in available_tables:
            print("📊 Loading from processed_gold_data table...")
            query = """
            SELECT date, price, volume, market_cap, price_change, 
                   price_ma_7, price_ma_30, volume_ma_7, price_lag_1, 
                   price_lag_7, volatility_7, volatility_30, trend_7
            FROM processed_gold_data 
            ORDER BY date ASC
            """
            
            try:
                df = pd.read_sql(query, self.connection)
                df['date'] = pd.to_datetime(df['date'])
                print(f"✅ Loaded {len(df)} records from processed_gold_data")
                return df
            except Exception as e:
                print(f"❌ Error loading from processed_gold_data: {e}")
                print("🔄 Trying to load from gold_prices and create features...")
        
        # Fallback: Load from gold_prices and create features
        if 'gold_prices' in available_tables:
            print("📊 Loading from gold_prices table and creating features...")
            query = """
            SELECT date, price, volume, market_cap
            FROM gold_prices 
            ORDER BY date ASC
            """
            
            try:
                df = pd.read_sql(query, self.connection)
                df['date'] = pd.to_datetime(df['date'])
                
                if len(df) < 30:
                    print(f"⚠️  Not enough data for feature engineering ({len(df)} records)")
                    return None
                
                # Create features
                df = self.create_features(df)
                print(f"✅ Loaded and processed {len(df)} records from gold_prices")
                return df
                
            except Exception as e:
                print(f"❌ Error loading from gold_prices: {e}")
                return None
        
        print("❌ No suitable data table found")
        return None

    def create_features(self, df):
        """Create features from raw data"""
        print("⚙️  Creating features from raw data...")
        
        # Feature engineering
        df['price_change'] = df['price'].pct_change()
        df['price_ma_7'] = df['price'].rolling(window=7, min_periods=1).mean()
        df['price_ma_30'] = df['price'].rolling(window=30, min_periods=1).mean()
        df['volume_ma_7'] = df['volume'].rolling(window=7, min_periods=1).mean()
        df['price_lag_1'] = df['price'].shift(1)
        df['price_lag_7'] = df['price'].shift(7)
        df['volatility_7'] = df['price'].rolling(window=7, min_periods=1).std()
        df['volatility_30'] = df['price'].rolling(window=30, min_periods=1).std()
        df['trend_7'] = df['price'].rolling(window=7, min_periods=2).apply(
            lambda x: 1 if len(x) > 1 and x.iloc[-1] > x.iloc[0] else 0, raw=False)

        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df

    def prepare_features(self, df):
        """Persiapan fitur untuk machine learning"""
        # Check which columns are available
        potential_features = [
            'volume', 'market_cap', 'price_change', 'price_ma_7',
            'price_ma_30', 'volume_ma_7', 'price_lag_1', 'price_lag_7',
            'volatility_7', 'volatility_30', 'trend_7'
        ]
        
        self.feature_columns = [col for col in potential_features if col in df.columns]
        
        print(f"📊 Using features: {self.feature_columns}")
        
        if len(self.feature_columns) == 0:
            print("❌ No suitable features found")
            return None, None

        X = df[self.feature_columns]
        y = df['price']
        
        # Handle any remaining NaN values
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        return X, y

    def train_linear_regression(self, X, y):
        """Training model Linear Regression"""
        if len(X) < 10:
            print("⚠️  Not enough data for Linear Regression training")
            return None
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.models['linear_regression'] = {
            'model': model,
            'mae': mae,
            'mse': mse,
            'r2': r2
        }

        print(f"✅ Linear Regression - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.4f}")
        return model

    def train_random_forest(self, X, y):
        """Training model Random Forest"""
        if len(X) < 10:
            print("⚠️  Not enough data for Random Forest training")
            return None
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.models['random_forest'] = {
            'model': model,
            'mae': mae,
            'mse': mse,
            'r2': r2
        }

        print(f"✅ Random Forest - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.4f}")
        return model

    def train_prophet(self, df):
        """Training model Prophet"""
        if len(df) < 30:
            print("⚠️  Not enough data for Prophet training (minimum 30 days required)")
            return None
            
        # Persiapan data untuk Prophet
        prophet_df = df[['date', 'price']].copy()
        prophet_df.columns = ['ds', 'y']

        try:
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,  # Not enough data for yearly
                interval_width=0.95
            )
            model.fit(prophet_df)

            # Evaluasi dengan cross-validation sederhana
            train_size = int(len(prophet_df) * 0.8)
            train_data = prophet_df.iloc[:train_size]
            test_data = prophet_df.iloc[train_size:]

            if len(test_data) == 0:
                print("⚠️  Not enough data for Prophet validation")
                r2 = 0.0
                mae = 0.0
                mse = 0.0
            else:
                model_eval = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False
                )
                model_eval.fit(train_data)

                future = model_eval.make_future_dataframe(periods=len(test_data))
                forecast = model_eval.predict(future)

                test_pred = forecast.iloc[train_size:]['yhat'].values
                test_actual = test_data['y'].values
                
                mae = mean_absolute_error(test_actual, test_pred)
                mse = mean_squared_error(test_actual, test_pred)
                r2 = r2_score(test_actual, test_pred)

            self.models['prophet'] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'r2': r2
            }

            print(f"✅ Prophet - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.4f}")
            return model
            
        except Exception as e:
            print(f"❌ Error training Prophet: {e}")
            return None

    def make_predictions(self, days_ahead=7):
        """Membuat prediksi untuk beberapa hari ke depan"""
        predictions = {}

        # Load data terbaru
        df = self.load_data()
        if df is None:
            print("❌ Cannot load data for predictions")
            return None

        # Prediksi dengan Prophet
        if 'prophet' in self.models:
            try:
                prophet_df = df[['date', 'price']].copy()
                prophet_df.columns = ['ds', 'y']

                future = self.models['prophet']['model'].make_future_dataframe(
                    periods=days_ahead)
                forecast = self.models['prophet']['model'].predict(future)

                future_dates = future.tail(days_ahead)['ds'].tolist()
                future_prices = forecast.tail(days_ahead)['yhat'].tolist()

                predictions['prophet'] = {
                    'dates': future_dates,
                    'prices': future_prices,
                    'model_metrics': {
                        'mae': self.models['prophet']['mae'],
                        'r2': self.models['prophet']['r2']
                    }
                }
                print(f"✅ Prophet predictions generated for {days_ahead} days")
            except Exception as e:
                print(f"❌ Error making Prophet predictions: {e}")

        # Prediksi dengan ML models (simplified approach)
        if 'random_forest' in self.models or 'linear_regression' in self.models:
            try:
                X, y = self.prepare_features(df)
                if X is not None:
                    # Use last known values for prediction
                    last_features = X.iloc[-1:].values
                    
                    for model_name in ['random_forest', 'linear_regression']:
                        if model_name in self.models:
                            model = self.models[model_name]['model']
                            pred_price = model.predict(last_features)[0]
                            
                            # Simple projection (same prediction for all days)
                            future_dates = []
                            future_prices = []
                            last_date = df['date'].max()
                            
                            for i in range(1, days_ahead + 1):
                                future_date = last_date + timedelta(days=i)
                                future_dates.append(future_date)
                                future_prices.append(pred_price)
                            
                            predictions[model_name] = {
                                'dates': future_dates,
                                'prices': future_prices,
                                'model_metrics': {
                                    'mae': self.models[model_name]['mae'],
                                    'r2': self.models[model_name]['r2']
                                }
                            }
                            print(f"✅ {model_name} predictions generated")
            except Exception as e:
                print(f"❌ Error making ML predictions: {e}")

        return predictions if predictions else None

    def save_predictions(self, predictions):
        """Menyimpan prediksi ke database"""
        if not predictions or not self.connection:
            print("❌ No predictions to save or no database connection")
            return False

        cursor = self.connection.cursor()

        try:
            # Clear old predictions
            cursor.execute("DELETE FROM predictions WHERE commodity_type = 'gold'")
            
            insert_count = 0
            for model_name, pred_data in predictions.items():
                for date, price in zip(pred_data['dates'], pred_data['prices']):
                    query = """
                    INSERT INTO predictions (commodity_type, prediction_date, predicted_price, 
                                          confidence_level, model_used) 
                    VALUES (%s, %s, %s, %s, %s)
                    """

                    confidence = max(0, min(100, pred_data['model_metrics']['r2'] * 100))
                    cursor.execute(
                        query, ('gold', date, price, confidence, model_name))
                    insert_count += 1

            self.connection.commit()
            print(f"✅ {insert_count} predictions saved to database")
            return True
        except Exception as e:
            print(f"❌ Error saving predictions: {e}")
            self.connection.rollback()
            return False
        finally:
            cursor.close()

    def save_models(self):
        """Menyimpan model yang sudah ditraining"""
        saved_count = 0
        for model_name, model_data in self.models.items():
            try:
                filename = f"{model_name}_model.joblib"
                joblib.dump(model_data['model'], filename)
                print(f"✅ Model {model_name} saved as {filename}")
                saved_count += 1
            except Exception as e:
                print(f"❌ Error saving {model_name}: {e}")
        
        print(f"📁 Total models saved: {saved_count}")

    def close_connection(self):
        if self.connection:
            self.connection.close()
            print("🔒 Database connection closed")


# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'commodity_prediction'
}


def main():
    print("🤖 Starting Commodity Prediction Training...")
    
    predictor = CommodityPredictor(DB_CONFIG)

    try:
        # Load data
        print("\n📊 Loading data...")
        df = predictor.load_data()

        if df is not None:
            print(f"✅ Data loaded: {len(df)} records")
            print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")

            # Prepare features
            X, y = predictor.prepare_features(df)
            
            if X is not None:
                print(f"📊 Features prepared: {X.shape[1]} features, {X.shape[0]} samples")

                # Train models
                print("\n🎯 Training models...")
                predictor.train_linear_regression(X, y)
                predictor.train_random_forest(X, y)
                predictor.train_prophet(df)

                if predictor.models:
                    print(f"\n✅ Successfully trained {len(predictor.models)} models")
                    
                    # Show model comparison
                    print("\n📈 Model Performance Comparison:")
                    for model_name, metrics in predictor.models.items():
                        print(f"  {model_name:15}: R² = {metrics['r2']:.4f}, MAE = {metrics['mae']:.2f}")

                    # Make predictions
                    print("\n🔮 Making predictions...")
                    predictions = predictor.make_predictions(days_ahead=7)

                    if predictions:
                        print(f"✅ Predictions generated for {len(predictions)} models")
                        
                        # Show sample predictions
                        print("\n📅 Sample Predictions:")
                        for model_name, pred_data in predictions.items():
                            print(f"\n{model_name.upper()}:")
                            for i, (date, price) in enumerate(zip(pred_data['dates'][:3], pred_data['prices'][:3])):
                                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                                print(f"  {date_str}: ${price:.2f}")
                            if len(pred_data['dates']) > 3:
                                print(f"  ... and {len(pred_data['dates']) - 3} more")

                        # Save predictions
                        print("\n💾 Saving predictions...")
                        predictor.save_predictions(predictions)
                    else:
                        print("❌ No predictions generated")

                    # Save models
                    print("\n💾 Saving trained models...")
                    predictor.save_models()
                else:
                    print("❌ No models were successfully trained")
            else:
                print("❌ Could not prepare features")
        else:
            print("❌ Could not load data")
            print("💡 Make sure to run improved_data_collection.py first")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        predictor.close_connection()

    print("\n🎉 Prediction training completed!")


if __name__ == "__main__":
    main()