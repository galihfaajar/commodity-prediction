# data_collection.py
import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataCollector:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None

    def connect_db(self):
        """Koneksi ke database MySQL"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            print("Berhasil terhubung ke database")
            return True
        except Exception as e:
            print(f"Error koneksi database: {e}")
            return False

    def get_historical_data(self, days=90):
        """Mengambil data historis dari database"""
        if not self.connection:
            self.connect_db()

        query = """
        SELECT date, price, volume, market_cap 
        FROM gold_prices 
        WHERE date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        ORDER BY date ASC
        """

        try:
            df = pd.read_sql(query, self.connection, params=[days])
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"Error mengambil data: {e}")
            return None

    def preprocess_data(self, df):
        """Preprocessing data untuk model machine learning"""
        if df is None or df.empty:
            return None

        # Sorting berdasarkan tanggal
        df = df.sort_values('date').reset_index(drop=True)

        # Menambah fitur teknikal
        df['price_change'] = df['price'].pct_change()
        df['price_ma_7'] = df['price'].rolling(window=7).mean()
        df['price_ma_30'] = df['price'].rolling(window=30).mean()
        df['volume_ma_7'] = df['volume'].rolling(window=7).mean()

        # Menambah fitur lag
        df['price_lag_1'] = df['price'].shift(1)
        df['price_lag_7'] = df['price'].shift(7)

        # Menambah fitur volatilitas
        df['volatility_7'] = df['price'].rolling(window=7).std()
        df['volatility_30'] = df['price'].rolling(window=30).std()

        # Menambah fitur trend
        df['trend_7'] = df['price'].rolling(window=7).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)

        # Hapus baris dengan NaN
        df = df.dropna().reset_index(drop=True)

        return df

    def save_processed_data(self, df, table_name='processed_data'):
        """Simpan data yang sudah diproses ke database"""
        if df is None or df.empty:
            return False

        try:
            df.to_sql(table_name, self.connection,
                      if_exists='replace', index=False)
            print(f"Data berhasil disimpan ke tabel {table_name}")
            return True
        except Exception as e:
            print(f"Error menyimpan data: {e}")
            return False

    def close_connection(self):
        """Tutup koneksi database"""
        if self.connection:
            self.connection.close()
            print("Koneksi database ditutup")


# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'commodity_prediction'
}


def main():
    # Initialize data collector
    collector = DataCollector(DB_CONFIG)

    # Collect and preprocess data
    print("Mengambil data historis...")
    raw_data = collector.get_historical_data(days=90)

    if raw_data is not None:
        print(f"Data berhasil diambil: {len(raw_data)} record")
        print("\nPreprocessing data...")
        processed_data = collector.preprocess_data(raw_data)

        if processed_data is not None:
            print(f"Data setelah preprocessing: {len(processed_data)} record")
            print("\nSample processed data:")
            print(processed_data.head())

            # Simpan hasil preprocessing
            collector.save_processed_data(processed_data)
        else:
            print("Error dalam preprocessing data")
    else:
        print("Gagal mengambil data dari database")

    collector.close_connection()


if __name__ == "__main__":
    main()
