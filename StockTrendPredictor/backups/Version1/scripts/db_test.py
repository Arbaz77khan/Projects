# Import libraries
import psycopg2

DB_HOST = 'db.vzhqubwxotgwppljcpsy.supabase.co'
DB_PORT = '5432'
DB_NAME = 'postgres'
DB_USER = 'postgres'
DB_PASSWORD = 'SUb$ua$d*Sn6ma+'

def test_connection():
    try:
        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        print("Connected successfully!")
        conn.close()
    except Exception as e:
        print("Connection failed: ", e)

if __name__ == '__main__':
    test_connection()