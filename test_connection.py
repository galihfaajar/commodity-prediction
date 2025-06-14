import mysql.connector

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Ganti dengan password Anda
    'database': 'commodity_prediction'
}

try:
    connection = mysql.connector.connect(**DB_CONFIG)
    print("✅ Koneksi database berhasil!")
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM gold_prices")
    result = cursor.fetchone()
    print(f"Jumlah data dalam tabel: {result[0]}")
    connection.close()
except Exception as e:
    print(f"❌ Error koneksi database: {e}")
