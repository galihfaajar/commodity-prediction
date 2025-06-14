# prediction_models.py
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
            return True
        except Exception as e:
            print(f"Error koneksi database: {e}")
            return False

    def load_data(self):
        """Load data yang sudah diproses"""
        if not self.connection:
            self.connect_db()

        query = """
        SELECT date, price, volume, market_cap, price_change, 
               price_ma_7, price_ma_30, volume_ma_7, price_lag_1, 
               price_lag_7, volatility_7, volatility_30, trend_7
        FROM gold_prices 
        ORDER BY date ASC
        """

        try:
            df = pd.read_sql(query, self.connection)
            df['date'] = pd.to_datetime(df['date'])

            # Feature engineering
            df['price_change'] = df['price'].pct_change()
            df['price_ma_7'] = df['price'].rolling(window=7).mean()
            df['price_ma_30'] = df['price'].rolling(window=30).mean()
            df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
            df['price_lag_1'] = df['price'].shift(1)
            df['price_lag_7'] = df['price'].shift(7)
            df['volatility_7'] = df['price'].rolling(window=7).std()
            df['volatility_30'] = df['price'].rolling(window=30).std()
            df['trend_7'] = df['price'].rolling(window=7).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)

            df = df.dropna().reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def prepare_features(self, df):
        """Persiapan fitur untuk machine learning"""
        self.feature_columns = [
            'volume', 'market_cap', 'price_change', 'price_ma_7',
            'price_ma_30', 'volume_ma_7', 'price_lag_1', 'price_lag_7',
            'volatility_7', 'volatility_30', 'trend_7'
        ]

        X = df[self.feature_columns]
        y = df['price']

        return X, y

    def train_linear_regression(self, X, y):
        """Training model Linear Regression"""
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

        print(
            f"Linear Regression - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.4f}")
        return model

    def train_random_forest(self, X, y):
        """Training model Random Forest"""
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

        print(f"Random Forest - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.4f}")
        return model

    def train_prophet(self, df):
        """Training model Prophet"""
        # Persiapan data untuk Prophet
        prophet_df = df[['date', 'price']].copy()
        prophet_df.columns = ['ds', 'y']

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        model.fit(prophet_df)

        # Evaluasi dengan cross-validation sederhana
        train_size = int(len(prophet_df) * 0.8)
        train_data = prophet_df.iloc[:train_size]
        test_data = prophet_df.iloc[train_size:]

        model_eval = Prophet()
        model_eval.fit(train_data)

        future = model_eval.make_future_dataframe(periods=len(test_data))
        forecast = model_eval.predict(future)

        test_pred = forecast.iloc[train_size:]['yhat']
        mae = mean_absolute_error(test_data['y'], test_pred)
        mse = mean_squared_error(test_data['y'], test_pred)
        r2 = r2_score(test_data['y'], test_pred)

        self.models['prophet'] = {
            'model': model,
            'mae': mae,
            'mse': mse,
            'r2': r2
        }

        print(f"Prophet - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.4f}")
        return model

    def make_predictions(self, days_ahead=7):
        """Membuat prediksi untuk beberapa hari ke depan"""
        predictions = {}

        # Load data terbaru
        df = self.load_data()
        if df is None:
            return None

        # Prediksi dengan Prophet
        if 'prophet' in self.models:
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

        return predictions

    def save_predictions(self, predictions):
        """Menyimpan prediksi ke database"""
        if not predictions or not self.connection:
            return False

        cursor = self.connection.cursor()

        for model_name, pred_data in predictions.items():
            for date, price in zip(pred_data['dates'], pred_data['prices']):
                query = """
                INSERT INTO predictions (commodity_type, prediction_date, predicted_price, 
                                      confidence_level, model_used) 
                VALUES (%s, %s, %s, %s, %s)
                """

                confidence = pred_data['model_metrics']['r2'] * 100
                cursor.execute(
                    query, ('gold', date, price, confidence, model_name))

        try:
            self.connection.commit()
            print("Prediksi berhasil disimpan ke database")
            return True
        except Exception as e:
            print(f"Error menyimpan prediksi: {e}")
            return False

    def save_models(self):
        """Menyimpan model yang sudah ditraining"""
        for model_name, model_data in self.models.items():
            filename = f"{model_name}_model.joblib"
            joblib.dump(model_data['model'], filename)
            print(f"Model {model_name} disimpan sebagai {filename}")

    def close_connection(self):
        if self.connection:
            self.connection.close()


# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'commodity_prediction'
}


def main():
    predictor = CommodityPredictor(DB_CONFIG)

    # Load data
    print("Loading data...")
    df = predictor.load_data()

    if df is not None:
        print(f"Data loaded: {len(df)} records")

        # Prepare features
        X, y = predictor.prepare_features(df)

        # Train models
        print("\nTraining models...")
        predictor.train_linear_regression(X, y)
        predictor.train_random_forest(X, y)
        predictor.train_prophet(df)

        # Make predictions
        print("\nMaking predictions...")
        predictions = predictor.make_predictions(days_ahead=7)

        if predictions:
            print("Predictions made successfully")
            for model_name, pred_data in predictions.items():
                print(f"\n{model_name.upper()} Predictions:")
                for date, price in zip(pred_data['dates'], pred_data['prices']):
                    print(f"  {date.strftime('%Y-%m-%d')}: ${price:.2f}")

            # Save predictions
            predictor.save_predictions(predictions)

        # Save models
        predictor.save_models()

    predictor.close_connection()


if __name__ == "__main__":
    main()
