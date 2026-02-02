import os
from database.db_utils import TimeSeriesDB

if __name__ == "__main__":
    db = TimeSeriesDB()
    try:
        db.connect()
        print("✅ PostgreSQL 연결 성공!")
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
    finally:
        db.close()
