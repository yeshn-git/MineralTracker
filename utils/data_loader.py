import pandas as pd
import sqlite3
from pathlib import Path

def create_database():
    """Create SQLite database and tables"""
    
    # Create database connection
    conn = sqlite3.connect('minerals.db')
    cursor = conn.cursor()
    
    # Drop existing tables if they exist
    cursor.execute('DROP TABLE IF EXISTS predictions')
    cursor.execute('DROP TABLE IF EXISTS import_history')
    cursor.execute('DROP TABLE IF EXISTS minerals')
    
    # Create minerals table WITH vulnerability_score column
    cursor.execute('''
        CREATE TABLE minerals (
            mineral_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            china_dependency_pct REAL,
            strategic_importance REAL,
            total_import_2023_usd_millions REAL,
            domestic_production_mt REAL,
            key_applications TEXT,
            alternative_sources TEXT,
            current_reserves_india_mt REAL,
            vulnerability_score REAL DEFAULT 0
        )
    ''')
    
    # Create import history table
    cursor.execute('''
        CREATE TABLE import_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mineral_id INTEGER,
            year INTEGER,
            import_value_usd_millions REAL,
            import_quantity_mt REAL,
            china_share_pct REAL,
            price_usd_per_kg REAL,
            FOREIGN KEY (mineral_id) REFERENCES minerals(mineral_id)
        )
    ''')
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mineral_id INTEGER,
            predicted_risk_level TEXT,
            risk_score REAL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (mineral_id) REFERENCES minerals(mineral_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database created successfully!")

def load_data_to_database():
    """Load CSV data into SQLite database"""
    
    conn = sqlite3.connect('minerals.db')
    cursor = conn.cursor()
    
    # Load minerals data from CSV
    minerals_df = pd.read_csv('data/raw/minerals_data.csv')
    
    # Insert data row by row to preserve schema
    for _, row in minerals_df.iterrows():
        cursor.execute('''
            INSERT OR REPLACE INTO minerals 
            (mineral_id, name, category, china_dependency_pct, strategic_importance, 
             total_import_2023_usd_millions, domestic_production_mt, key_applications, 
             alternative_sources, current_reserves_india_mt, vulnerability_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        ''', (
            row['mineral_id'], row['name'], row['category'], 
            row['china_dependency_pct'], row['strategic_importance'],
            row['total_import_2023_usd_millions'], row['domestic_production_mt'],
            row['key_applications'], row['alternative_sources'], 
            row['current_reserves_india_mt']
        ))
    
    print(f"✅ Loaded {len(minerals_df)} minerals into database")
    
    # Load import history
    history_df = pd.read_csv('data/raw/import_history.csv')
    
    # Clear existing import history
    cursor.execute('DELETE FROM import_history')
    
    # Insert import history
    for _, row in history_df.iterrows():
        cursor.execute('''
            INSERT INTO import_history 
            (mineral_id, year, import_value_usd_millions, import_quantity_mt, 
             china_share_pct, price_usd_per_kg)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            row['mineral_id'], row['year'], row['import_value_usd_millions'],
            row['import_quantity_mt'], row['china_share_pct'], row['price_usd_per_kg']
        ))
    
    print(f"✅ Loaded {len(history_df)} import records into database")
    
    conn.commit()
    conn.close()
def get_all_minerals():
    """Retrieve all minerals from database"""
    conn = sqlite3.connect('minerals.db')
    df = pd.read_sql_query("SELECT * FROM minerals", conn)
    conn.close()
    return df

def get_mineral_by_id(mineral_id):
    """Get specific mineral data"""
    conn = sqlite3.connect('minerals.db')
    query = "SELECT * FROM minerals WHERE mineral_id = ?"
    df = pd.read_sql_query(query, conn, params=(mineral_id,))
    conn.close()
    return df.iloc[0] if len(df) > 0 else None

def get_import_history(mineral_id):
    """Get import history for a mineral"""
    conn = sqlite3.connect('minerals.db')
    query = "SELECT * FROM import_history WHERE mineral_id = ? ORDER BY year"
    df = pd.read_sql_query(query, conn, params=(mineral_id,))
    conn.close()
    return df

if __name__ == "__main__":
    # Run this to set up database
    create_database()
    load_data_to_database()
    
    # Test
    minerals = get_all_minerals()
    print(f"\n📊 Database contains {len(minerals)} minerals:")
    print(minerals[['name', 'china_dependency_pct', 'strategic_importance']])