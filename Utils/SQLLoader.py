import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv


def save_csv_from_database(query: str, file_path: str):
    load_dotenv()

    conn = psycopg2.connect(
        database=os.getenv('DATABASE'),
        user=os.getenv('USER_DB'),
        password=os.getenv('PASSWORD'),
        host=os.getenv('HOST'),
        port=os.getenv('PORT')
    )

    df_psycopg = pd.read_sql(query, conn)
    conn.close()

    df_psycopg.to_csv(file_path, index=False)
