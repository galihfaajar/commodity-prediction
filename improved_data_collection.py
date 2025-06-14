# improved_data_collection.py
import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ImprovedDataCollector:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None

    def connect_db(self):
        """Koneksi ke database MySQL"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            print("✅ Berhasil terhubung ke database")
            return True
        except Exception as e:
            print(f"❌ Error koneksi database: {e}")
            return False

    def check_data_availability(self):
        """Check if data is available in the database"""
        if not self.connection:
            self.connect_db()

        cursor = self.connection.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM gold_prices")
            count = cursor.fetchone()[0]
            
            if count == 0:
                print("⚠️  Tidak ada data dalam tabel gold_prices")
                print("💡 Jalankan database_diagnostic.py terlebih dahulu untuk membuat data sample")
                return False
            else:
                print(f"📊 Ditemukan {count} record dalam database")
                return True
        except Exception as e:
            print(f"❌ Error checking data: {e}")
            return False
        finally:
            cursor.close()

    def get_historical_data(self, days=90):
        """Mengambil data historis dari database"""
        if not self.connection:
            self.connect_db()

        # First check if data is available
        if not self.check_data_availability():
            return None

        query = """
        SELECT date, price, volume, market_cap 
        FROM gold_prices 
        WHERE date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        ORDER BY date ASC
        """

        try:
            df = pd.read_sql(query, self.connection, params=[days])
            df['date'] = pd.to_datetime(df['date'])
            
            if df.empty:
                print(f"⚠️  Tidak ada data dalam {days} hari terakhir")
                # Try to get any available data
                query_all = "SELECT date, price, volume, market_cap FROM gold_prices ORDER BY date ASC"
                df = pd.read_sql(query_all, self.connection)
                df['date'] = pd.to_datetime(df['date'])
                
                if not df.empty:
                    print(f"📊 Menggunakan semua data yang tersedia: {len(df)} record")
                else:
                    print("❌ Tidak ada data sama sekali dalam database")
                    return None
            
            return df
        except Exception as e:
            print(f"❌ Error mengambil data: {e}")
            return None

    def preprocess_data(self, df):
        """Preprocessing data untuk model machine learning dengan error handling yang lebih baik"""
        if df is None or df.empty:
            print("❌ DataFrame kosong atau None")
            return None

        print(f"📊 Memproses {len(df)} record...")

        # Sorting berdasarkan tanggal
        df = df.sort_values('date').reset_index(drop=True)

        # Check if we have enough data for rolling calculations
        min_data_points = 30
        if len(df) < min_data_points:
            print(f"⚠️  Data terlalu sedikit ({len(df)} record). Minimal diperlukan {min_data_points} record")
            print("💡 Jalankan database_diagnostic.py untuk membuat lebih banyak data sample")
            return None

        try:
            # Menambah fitur teknikal
            df['price_change'] = df['price'].pct_change()
            
            # Moving averages dengan pengecekan ukuran window
            window_7 = min(7, len(df) - 1)
            window_30 = min(30, len(df) - 1)
            
            if window_7 > 0:
                df['price_ma_7'] = df['price'].rolling(window=window_7, min_periods=1).mean()
                df['volume_ma_7'] = df['volume'].rolling(window=window_7, min_periods=1).mean()
                df['volatility_7'] = df['price'].rolling(window=window_7, min_periods=1).std()
            
            if window_30 > 0:
                df['price_ma_30'] = df['price'].rolling(window=window_30, min_periods=1).mean()
                df['volatility_30'] = df['price'].rolling(window=window_30, min_periods=1).std()

            # Menambah fitur lag
            df['price_lag_1'] = df['price'].shift(1)
            
            if len(df) > 7:
                df['price_lag_7'] = df['price'].shift(7)
            else:
                df['price_lag_7'] = df['price'].shift(1)  # Fallback to 1-day lag

            # Menambah fitur trend dengan pengecekan ukuran window
            trend_window = min(7, len(df) - 1)
            if trend_window > 1:
                df['trend_7'] = df['price'].rolling(window=trend_window, min_periods=2).apply(
                    lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=False)
            else:
                df['trend_7'] = 0

            # Handle NaN values more gracefully
            initial_rows = len(df)
            df = df.dropna().reset_index(drop=True)
            final_rows = len(df)
            
            print(f"📊 Rows after preprocessing: {final_rows} (removed {initial_rows - final_rows} rows with NaN)")
            
            if final_rows == 0:
                print("❌ Semua data hilang setelah preprocessing")
                return None
            
            if final_rows < 10:
                print("⚠️  Data tersisa terlalu sedikit setelah preprocessing")
                return None

            # Show sample of processed data
            print("\n📈 Sample processed features:")
            feature_cols = ['date', 'price', 'price_change', 'price_ma_7', 'price_ma_30', 'volatility_7']
            available_cols = [col for col in feature_cols if col in df.columns]
            print(df[available_cols].head())

            return df
            
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")
            return None

    def validate_processed_data(self, df):
        """Validate the processed data"""
        if df is None or df.empty:
            return False

        print("\n🔍 Validating processed data...")
        
        required_columns = ['date', 'price', 'volume', 'market_cap']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False

        # Check for infinite values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        infinite_counts = {}
        for col in numeric_columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                infinite_counts[col] = inf_count

        if infinite_counts:
            print(f"⚠️  Infinite values found: {infinite_counts}")
            # Replace infinite values with NaN and then forward fill
            df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill')

        # Final statistics
        print(f"✅ Data validation complete:")
        print(f"   - Total rows: {len(df)}")
        print(f"   - Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   - Price range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")
        print(f"   - Columns: {list(df.columns)}")

        return True

    def save_processed_data(self, df, table_name='processed_gold_data'):
        """Simpan data yang sudah diproses ke database"""
        if df is None or df.empty:
            print("❌ Tidak ada data untuk disimpan")
            return False

        cursor = self.connection.cursor()
        
        try:
            # Drop table if exists and create new one
            drop_query = f"DROP TABLE IF EXISTS {table_name}"
            cursor.execute(drop_query)
            
            # Create table with proper MySQL syntax
            create_query = f"""
            CREATE TABLE {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date DATE NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                volume BIGINT NOT NULL,
                market_cap DECIMAL(15,2) NOT NULL,
                price_change DECIMAL(8,6),
                price_ma_7 DECIMAL(10,2),
                volume_ma_7 DECIMAL(15,2),
                volatility_7 DECIMAL(10,2),
                price_ma_30 DECIMAL(10,2),
                volatility_30 DECIMAL(10,2),
                price_lag_1 DECIMAL(10,2),
                price_lag_7 DECIMAL(10,2),
                trend_7 DECIMAL(3,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_query)
            
            # Prepare insert query
            columns = ['date', 'price', 'volume', 'market_cap', 'price_change', 
                      'price_ma_7', 'volume_ma_7', 'volatility_7', 'price_ma_30', 
                      'volatility_30', 'price_lag_1', 'price_lag_7', 'trend_7']
            
            placeholders = ', '.join(['%s'] * len(columns))
            insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            # Convert DataFrame to list of tuples
            data_to_insert = []
            for _, row in df.iterrows():
                row_data = []
                for col in columns:
                    if col == 'date':
                        row_data.append(row[col].strftime('%Y-%m-%d'))
                    elif pd.isna(row[col]):
                        row_data.append(None)
                    else:
                        row_data.append(float(row[col]))
                data_to_insert.append(tuple(row_data))
            
            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(data_to_insert), batch_size):
                batch = data_to_insert[i:i + batch_size]
                cursor.executemany(insert_query, batch)
            
            self.connection.commit()
            print(f"✅ Data berhasil disimpan ke tabel {table_name}")
            print(f"   - {len(data_to_insert)} rows inserted")
            
            return True
            
        except Exception as e:
            print(f"❌ Error menyimpan data: {e}")
            self.connection.rollback()
            return False
        finally:
            cursor.close()

    def close_connection(self):
        """Tutup koneksi database"""
        if self.connection:
            self.connection.close()
            print("🔒 Koneksi database ditutup")


# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'commodity_prediction'
}


def main():
    print("🚀 Starting Improved Data Collection...")
    
    # Initialize data collector
    collector = ImprovedDataCollector(DB_CONFIG)

    try:
        # Collect and preprocess data
        print("\n📊 Mengambil data historis...")
        raw_data = collector.get_historical_data(days=90)

        if raw_data is not None:
            print(f"✅ Data berhasil diambil: {len(raw_data)} record")
            
            print("\n⚙️  Preprocessing data...")
            processed_data = collector.preprocess_data(raw_data)

            if processed_data is not None:
                print(f"✅ Data setelah preprocessing: {len(processed_data)} record")
                
                # Validate processed data
                if collector.validate_processed_data(processed_data):
                    # Simpan hasil preprocessing
                    print("\n💾 Menyimpan processed data...")
                    if collector.save_processed_data(processed_data):
                        print("🎉 Proses data collection berhasil!")
                    else:
                        print("❌ Gagal menyimpan data")
                else:
                    print("❌ Data validation gagal")
            else:
                print("❌ Error dalam preprocessing data")
                print("💡 Pastikan database memiliki data yang cukup (minimal 30 record)")
        else:
            print("❌ Gagal mengambil data dari database")
            print("💡 Jalankan 'python database_diagnostic.py' untuk membuat data sample")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        collector.close_connection()

    print("\n🏁 Data collection process completed")


if __name__ == "__main__":
    main()