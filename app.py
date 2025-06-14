# app.py - Flask Backend
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import mysql.connector
import pandas as pd
import json
from datetime import datetime, timedelta
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'commodity_prediction'
}


class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.config)
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False

    def close(self):
        if self.connection:
            self.connection.close()

    def execute_query(self, query, params=None):
        if not self.connection:
            self.connect()

        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params)
            result = cursor.fetchall()
            cursor.close()
            return result
        except Exception as e:
            print(f"Query execution error: {e}")
            return None


db_manager = DatabaseManager(DB_CONFIG)


@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html')


@app.route('/api/historical-data')
def get_historical_data():
    """API untuk mengambil data historis"""
    days = request.args.get('days', 30, type=int)

    query = """
    SELECT date, price, volume, market_cap 
    FROM gold_prices 
    WHERE date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
    ORDER BY date ASC
    """

    data = db_manager.execute_query(query, [days])

    if data:
        # Format data untuk Chart.js
        formatted_data = {
            'labels': [item['date'].strftime('%Y-%m-%d') for item in data],
            'prices': [float(item['price']) for item in data],
            'volumes': [int(item['volume']) for item in data]
        }
        return jsonify(formatted_data)
    else:
        return jsonify({'error': 'Failed to fetch data'}), 500


@app.route('/api/predictions')
def get_predictions():
    """API untuk mengambil prediksi"""
    days = request.args.get('days', 7, type=int)

    query = """
    SELECT prediction_date, predicted_price, confidence_level, model_used 
    FROM predictions 
    WHERE commodity_type = 'gold' 
    AND prediction_date >= CURDATE()
    ORDER BY prediction_date ASC
    LIMIT %s
    """

    data = db_manager.execute_query(query, [days])

    if data:
        formatted_data = {
            'labels': [item['prediction_date'].strftime('%Y-%m-%d') for item in data],
            'prices': [float(item['predicted_price']) for item in data],
            'confidence': [float(item['confidence_level']) for item in data],
            'models': [item['model_used'] for item in data]
        }
        return jsonify(formatted_data)
    else:
        return jsonify({'error': 'No predictions found'}), 404


@app.route('/api/model-performance')
def get_model_performance():
    """API untuk mendapatkan performa model"""
    query = """
    SELECT model_name, parameters, accuracy_score 
    FROM model_configs 
    WHERE is_active = 1
    ORDER BY accuracy_score DESC
    """

    data = db_manager.execute_query(query)

    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'No model data found'}), 404


@app.route('/api/analytics')
def get_analytics():
    """API untuk analytics dashboard"""
    # Data statistik dasar
    stats_query = """
    SELECT 
        COUNT(*) as total_records,
        MIN(price) as min_price,
        MAX(price) as max_price,
        AVG(price) as avg_price,
        MIN(date) as start_date,
        MAX(date) as end_date
    FROM gold_prices
    """

    stats = db_manager.execute_query(stats_query)

    # Trend analysis
    trend_query = """
    SELECT 
        DATE_FORMAT(date, '%Y-%m') as month,
        AVG(price) as avg_price,
        AVG(volume) as avg_volume
    FROM gold_prices
    WHERE date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY DATE_FORMAT(date, '%Y-%m')
    ORDER BY month
    """

    trends = db_manager.execute_query(trend_query)

    if stats and trends:
        analytics_data = {
            'statistics': stats[0],
            'monthly_trends': trends
        }
        return jsonify(analytics_data)
    else:
        return jsonify({'error': 'Analytics data not available'}), 500


@app.route('/api/run-prediction', methods=['POST'])
def run_prediction():
    """API untuk menjalankan prediksi baru"""
    try:
        # Simulate model prediction (dalam implementasi nyata, panggil model ML)
        from prediction_models import CommodityPredictor

        predictor = CommodityPredictor(DB_CONFIG)
        predictions = predictor.make_predictions(days_ahead=7)

        if predictions:
            predictor.save_predictions(predictions)
            predictor.close_connection()
            return jsonify({'message': 'Predictions generated successfully'})
        else:
            return jsonify({'error': 'Failed to generate predictions'}), 500

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/api/latest-price')
def get_latest_price():
    """API untuk mendapatkan harga terbaru"""
    query = """
    SELECT date, price, volume, market_cap
    FROM gold_prices 
    ORDER BY date DESC 
    LIMIT 1
    """

    data = db_manager.execute_query(query)

    if data:
        # Calculate price change
        prev_query = """
        SELECT price 
        FROM gold_prices 
        ORDER BY date DESC 
        LIMIT 1 OFFSET 1
        """
        prev_data = db_manager.execute_query(prev_query)

        latest = data[0]
        price_change = 0
        if prev_data:
            price_change = float(latest['price']) - \
                float(prev_data[0]['price'])

        result = {
            'date': latest['date'].strftime('%Y-%m-%d'),
            'price': float(latest['price']),
            'volume': int(latest['volume']),
            'market_cap': float(latest['market_cap']),
            'price_change': price_change,
            'change_percent': (price_change / float(latest['price'])) * 100 if latest['price'] else 0
        }
        return jsonify(result)
    else:
        return jsonify({'error': 'No data found'}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
