# database_diagnostic.py
import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'commodity_prediction'
}

class DatabaseDiagnostic:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None
    
    def connect(self):
        """Connect to database"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            print("✅ Database connection successful!")
            return True
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return False
    
    def check_tables(self):
        """Check if required tables exist"""
        if not self.connection:
            return False
        
        cursor = self.connection.cursor()
        
        # Check existing tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"\n📋 Available tables: {[table[0] for table in tables]}")
        
        # Check gold_prices table structure
        try:
            cursor.execute("DESCRIBE gold_prices")
            columns = cursor.fetchall()
            print(f"\n📊 gold_prices table structure:")
            for col in columns:
                print(f"  - {col[0]}: {col[1]}")
        except Exception as e:
            print(f"❌ gold_prices table doesn't exist: {e}")
            return False
        
        # Check data count
        cursor.execute("SELECT COUNT(*) FROM gold_prices")
        count = cursor.fetchone()[0]
        print(f"\n📈 Records in gold_prices: {count}")
        
        if count > 0:
            # Show sample data
            cursor.execute("SELECT * FROM gold_prices LIMIT 5")
            sample_data = cursor.fetchall()
            print(f"\n🔍 Sample data:")
            for row in sample_data:
                print(f"  {row}")
        
        cursor.close()
        return True
    
    def create_tables(self):
        """Create required tables if they don't exist"""
        if not self.connection:
            return False
        
        cursor = self.connection.cursor()
        
        # Create gold_prices table
        create_gold_prices = """
        CREATE TABLE IF NOT EXISTS gold_prices (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE NOT NULL,
            price DECIMAL(10,2) NOT NULL,
            volume BIGINT NOT NULL,
            market_cap DECIMAL(15,2) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_date (date)
        )
        """
        
        # Create predictions table
        create_predictions = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            commodity_type VARCHAR(50) NOT NULL,
            prediction_date DATE NOT NULL,
            predicted_price DECIMAL(10,2) NOT NULL,
            confidence_level DECIMAL(5,2) NOT NULL,
            model_used VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Create model_configs table
        create_model_configs = """
        CREATE TABLE IF NOT EXISTS model_configs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            parameters TEXT,
            accuracy_score DECIMAL(5,4),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        try:
            cursor.execute(create_gold_prices)
            cursor.execute(create_predictions)
            cursor.execute(create_model_configs)
            self.connection.commit()
            print("✅ Tables created successfully!")
            return True
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            return False
        finally:
            cursor.close()
    
    def generate_sample_data(self, days=90):
        """Generate sample gold price data"""
        if not self.connection:
            return False
        
        cursor = self.connection.cursor()
        
        # Generate dates from 90 days ago to today
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Base gold price (around $2000)
        base_price = 2000.0
        current_price = base_price
        
        sample_data = []
        current_date = start_date
        
        while current_date <= end_date:
            # Simulate price movement (random walk with slight upward trend)
            price_change = random.uniform(-0.05, 0.06)  # -5% to +6% daily change
            current_price = current_price * (1 + price_change)
            
            # Keep price within reasonable bounds
            current_price = max(1500, min(2500, current_price))
            
            # Generate volume (random between 1M to 10M)
            volume = random.randint(1000000, 10000000)
            
            # Generate market cap (price * approximate supply)
            market_cap = current_price * 200000000  # Approximate gold supply
            
            sample_data.append((
                current_date.strftime('%Y-%m-%d'),
                round(current_price, 2),
                volume,
                round(market_cap, 2)
            ))
            
            current_date += timedelta(days=1)
        
        # Insert data
        insert_query = """
        INSERT IGNORE INTO gold_prices (date, price, volume, market_cap) 
        VALUES (%s, %s, %s, %s)
        """
        
        try:
            cursor.executemany(insert_query, sample_data)
            self.connection.commit()
            print(f"✅ Generated {len(sample_data)} sample records")
            
            # Verify insertion
            cursor.execute("SELECT COUNT(*) FROM gold_prices")
            count = cursor.fetchone()[0]
            print(f"📈 Total records in database: {count}")
            
            return True
        except Exception as e:
            print(f"❌ Error inserting sample data: {e}")
            return False
        finally:
            cursor.close()
    
    def add_model_configs(self):
        """Add sample model configurations"""
        if not self.connection:
            return False
        
        cursor = self.connection.cursor()
        
        model_configs = [
            ('Linear Regression', '{"fit_intercept": true, "normalize": false}', 0.7543, True),
            ('Random Forest', '{"n_estimators": 100, "max_depth": 10, "random_state": 42}', 0.8234, True),
            ('Prophet', '{"daily_seasonality": true, "weekly_seasonality": true, "yearly_seasonality": true}', 0.7891, True)
        ]
        
        insert_query = """
        INSERT IGNORE INTO model_configs (model_name, parameters, accuracy_score, is_active) 
        VALUES (%s, %s, %s, %s)
        """
        
        try:
            cursor.executemany(insert_query, model_configs)
            self.connection.commit()
            print("✅ Model configurations added")
            return True
        except Exception as e:
            print(f"❌ Error adding model configs: {e}")
            return False
        finally:
            cursor.close()
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("🔒 Database connection closed")

def main():
    print("🔍 Starting Database Diagnostic...")
    
    diagnostic = DatabaseDiagnostic(DB_CONFIG)
    
    if not diagnostic.connect():
        return
    
    # Check current state
    print("\n" + "="*50)
    print("CHECKING CURRENT DATABASE STATE")
    print("="*50)
    diagnostic.check_tables()
    
    # Create tables if needed
    print("\n" + "="*50)
    print("CREATING TABLES")
    print("="*50)
    diagnostic.create_tables()
    
    # Generate sample data
    print("\n" + "="*50)
    print("GENERATING SAMPLE DATA")
    print("="*50)
    diagnostic.generate_sample_data(days=90)
    
    # Add model configurations
    print("\n" + "="*50)
    print("ADDING MODEL CONFIGURATIONS")
    print("="*50)
    diagnostic.add_model_configs()
    
    # Final check
    print("\n" + "="*50)
    print("FINAL DATABASE STATE")
    print("="*50)
    diagnostic.check_tables()
    
    diagnostic.close()
    
    print("\n🎉 Database diagnostic complete!")
    print("You can now run your data_collection.py script.")

if __name__ == "__main__":
    main()