import pandas as pd
from database.db_utils import TimeSeriesDB

if __name__ == "__main__":
    df = pd.read_csv("merged_influenza_data.csv")
    db = TimeSeriesDB()
    db.insert_dataframe(df, table_name="influenza_data")
    db.close()
