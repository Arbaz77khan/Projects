# Import libraries
import psycopg2
import logging
import pandas as pd
from dotenv import load_dotenv
import os
import re
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect database
def connect_db():
    try:
        # load_dotenv()
        conn = psycopg2.connect(
            # host=os.getenv('DB_HOST'), 
            # port=os.getenv('DB_PORT'), 
            # database=os.getenv('DB_NAME'), 
            # user=os.getenv('DB_USER'), 
            # password=os.getenv('DB_PASSWORD')
            host=st.secrets["DB_HOST"],
            port=st.secrets["DB_PORT"],
            database=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"]
        )
        conn.autocommit = True
        logging.info("Database connected successfully!")
        return conn
    except Exception as e:
        logging.error(f"Connection failed: {e}")
        return None
        
# Create tables
def create_core_table(conn):
    with conn.cursor() as cur:
        # Table: stock_list
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_list(
                stock_symbol TEXT PRIMARY KEY,
                added_at DATE DEFAULT CURRENT_DATE
            );
        """)
        logging.info("'stock_list' table activated")

        # Table: model_meta
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_meta(
                stock_symbol TEXT PRIMARY KEY,
                avg_mse NUMERIC,
                avg_r2 NUMERIC,
                model_url TEXT,
                day_1 NUMERIC, 
                day_2 NUMERIC, 
                day_3 NUMERIC, 
                day_4 NUMERIC, 
                day_5 NUMERIC, 
                day_6 NUMERIC, 
                day_7 NUMERIC, 
                weekly_trend VARCHAR,
                last_trained_at DATE DEFAULT CURRENT_DATE
            );
        """)
        logging.info("'model_meta' table activated")

# check stock in stock_list
def check_stock_in_list(conn, symbol):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT stock_symbol 
            FROM stock_list 
            WHERE stock_symbol = %s
        """, (symbol,))
        result = cur.fetchone()
        return result is not None
    
# Add stock to stock_list table
def add_stock_to_list(conn, symbol):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO stock_list (stock_symbol) VALUES (%s)
            ON CONFLICT (stock_symbol) DO NOTHING
        """, (symbol,))
        logging.info(f"Added {symbol} to stock_list table.")

# fetch symbol
def fetch_symbols(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT stock_symbol 
            FROM stock_list
        """)
        symbol_tuple = cur.fetchall()
        symbol_list = [symbol[0] for symbol in symbol_tuple]  # Extract symbols from tuples
        logging.info(f"Fetched stock symbol list: {symbol_list}")
        return symbol_list

# create table name
def create_table_name(symbol):
    clean = re.sub(r'[^\w]', '_', symbol.lower())
    return f"{clean}_data"

# Create processed table
def create_processed_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date DATE PRIMARY KEY,
                close NUMERIC,
                sma_14 NUMERIC,
                ema_14 NUMERIC,
                rsi_14 NUMERIC,
                macd NUMERIC,
                close_lag1 NUMERIC,
                close_lag2 NUMERIC,
                close_lag3 NUMERIC,
                close_lag4 NUMERIC,
                close_lag5 NUMERIC,
                close_lag6 NUMERIC,
                close_lag7 NUMERIC,
                target_day_1 NUMERIC,
                target_day_2 NUMERIC,
                target_day_3 NUMERIC,
                target_day_4 NUMERIC,
                target_day_5 NUMERIC,
                target_day_6 NUMERIC,
                target_day_7 NUMERIC,
                avg_next_7_days NUMERIC,
                trend TEXT
            );
        """)
        logging.info(f"Table {table_name} activated.")

# Inserting stock data in processed tables
def insert_processed_data(conn, table_name, df):
    with conn.cursor() as cur:
        insert_query = f"""
            INSERT INTO {table_name} values(
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (date) DO NOTHING;
        """
        data_tuples = [
            (row.Date, row.Close, row.sma_14, row.ema_14, row.rsi_14, row.macd,
            row.close_lag1, row.close_lag2, row.close_lag3, row.close_lag4, row.close_lag5,
            row.close_lag6, row.close_lag7, row.target_day_1, row.target_day_2, row.target_day_3,
            row.target_day_4, row.target_day_5, row.target_day_6, row.target_day_7, row.avg_next_7_days, row.trend)
            for row in df.itertuples(index=False)
        ]

        cur.executemany(insert_query, data_tuples)
        logging.info(f"Inserted {len(data_tuples)} rows into '{table_name}'")

# Fetch latest date
def get_latest_date(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT MAX(date) 
            FROM {table_name};
        """)
        result = cur.fetchone()
        latest_date = result[0]
        logging.info(f"Latest date in {table_name}: {latest_date}")
        return latest_date

# Delete older data
def delete_old_data(conn, table_name, max_table_size):
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cur.fetchone()[0]

        if count > max_table_size:
            offset = count - max_table_size
            cur.execute(f"""
                DELETE FROM {table_name}
                WHERE date IN (
                    SELECT date FROM {table_name}
                    ORDER BY date ASC
                    LIMIT {offset}
                    );
            """)
            logging.info(f"Deleted {offset} older dates from {table_name}")
        else:
            logging.info(f"Table {table_name} has {count} rows; No deletion of older dates required.")

# Call insert/delete data
def update_table(conn, table_name, df, max_table_size):
    latest_date = get_latest_date(conn, table_name)
    
    # Ensure proper datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    latest_date = pd.to_datetime(latest_date)

    if latest_date is not None:
        final_df = df[df['Date'] > latest_date]
    else:
        final_df = df.copy()

    if not final_df.empty:
        insert_processed_data(conn, table_name, final_df)
        delete_old_data(conn, table_name, max_table_size)
    else:
        logging.info("No new data to insert; DB already up-to-date")

# Fetch dataset
def get_entire_processed_data(symbol, conn):
    table_name = create_table_name(symbol)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    logging.info(f"Loaded {len(df)} rows from table {table_name}")
    return df

# Fetch latest feature row
def get_latest_feature_row(conn, symbol):
    table_name = create_table_name(symbol)
    query = f"""
        SELECT sma_14, ema_14, rsi_14, macd, close_lag1, close_lag2, close_lag3, close_lag4, close_lag5, close_lag6, close_lag7
        FROM {table_name}
        ORDER BY date DESC
        LIMIT 1;
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        logging.error(f"No feature data available for {symbol}")
        return 
    else:
        logging.info("Latest feature row fetched")
        return df

# Insert executable model
def update_model_meta(conn, symbol, avg_mse, avg_r2, model_url):
    with conn.cursor() as cur:
        query = """
            INSERT INTO model_meta (stock_symbol, avg_mse, avg_r2, model_url, last_trained_at)
            VALUES (%s, %s, %s, %s, DEFAULT)
            ON CONFLICT (stock_symbol) 
            DO UPDATE SET
                avg_mse = EXCLUDED.avg_mse,
                avg_r2 = EXCLUDED.avg_r2,
                model_url = EXCLUDED.model_url,
                last_trained_at = CURRENT_DATE;
        """
        params = (symbol, avg_mse, avg_r2, model_url)
        cur.execute(query, params)
        logging.info(f"Updated model_meta for {symbol}")

# update inference result
def update_inference_result(conn, symbol, inference_dict, trend):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE model_meta
            SET day_1 = %s,
                day_2 = %s,
                day_3 = %s,
                day_4 = %s,
                day_5 = %s,
                day_6 = %s,
                day_7 = %s,
                weekly_trend = %s
            WHERE stock_symbol = %s
        """, (
            inference_dict.get('Day 1'),
            inference_dict.get('Day 2'),
            inference_dict.get('Day 3'),
            inference_dict.get('Day 4'),
            inference_dict.get('Day 5'),
            inference_dict.get('Day 6'),
            inference_dict.get('Day 7'),
            trend,
            symbol))
        conn.commit()
        logging.info(f"Updated inference result for {symbol}")

# Fetch inference result
def fetch_inference_result(conn, symbol):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT day_1, day_2, day_3, day_4, day_5, day_6, day_7, weekly_trend
            FROM model_meta
            WHERE stock_symbol = %s
        """, (symbol,))
        row = cur.fetchone()
        if row:
            inference = {
                f'Day {i+1}': row[i] for i in range(7)
            }
            trend = row[7]
            return inference, trend
        else:
            return None, None

# Fetch model URL
def get_model_url(conn, symbol):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT model_url
            FROM model_meta
            WHERE stock_symbol = %s
        """, (symbol,))
        result = cur.fetchone()
        model_url = result[0]
        logging.info(f"Model URL: {model_url}")
        return model_url

# Main execution
if __name__ == '__main__':

    df = pd.read_csv('D:/Master_Folder/Data Science Course/Projects/StockTrendPredictor/data/processed/data.csv')
    symbol = 'TSLA'
    max_table_size = 365
    table_name = create_table_name(symbol)

    conn = connect_db()
    if conn is not None:
        create_core_table(conn)
        if not check_stock_in_list(conn, symbol):
            add_stock_to_list(conn, symbol)
            create_processed_table(conn, table_name)
        update_table(conn, table_name, df, max_table_size)

        fetch_symbols(conn)

        



