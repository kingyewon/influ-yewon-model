
"""
PostgreSQL을 사용한 시계열 데이터 관리 유틸리티

PostgreSQL은 강력한 트랜잭션, 확장성, SQL 표준 지원을 제공하는 오픈소스 RDBMS입니다.
대용량 CSV 파일을 효율적으로 처리하고, 외부 연결 및 분석에 적합합니다.

주요 기능:
- CSV를 PostgreSQL 테이블로 임포트
- SQL 쿼리를 통한 유연한 데이터 필터링
- Pandas와의 통합
- 대용량 데이터 처리
- API를 통한 최신 데이터 자동 업데이트
"""

import psycopg2
import psycopg2.extras
import pandas as pd
from pathlib import Path
from typing import Optional, List
import time
import os
import json
import requests
from dotenv import load_dotenv
import warnings

# 환경변수 로드
load_dotenv()

# SSL 경고 무시
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class TimeSeriesDB:
    """시계열 데이터를 PostgreSQL로 관리하는 클래스"""
    def __init__(self, host=None, dbname=None, user=None, password=None, port=5432):
        self.host = host or os.getenv('PG_HOST', 'localhost')
        self.dbname = dbname or os.getenv('PG_DB', 'influenza')
        self.user = user or os.getenv('PG_USER', 'postgres')
        self.password = password or os.getenv('PG_PASSWORD', 'postgres')
        self.port = int(port or os.getenv('PG_PORT', 5432))
        self.conn = None

    def insert_dataframe(self, df: pd.DataFrame, table_name: str = "influenza_data", if_exists: str = "append"):
        """
        DataFrame 데이터를 PostgreSQL 테이블에 적재
        (컬럼명 매핑 및 결측치 None 처리)
        """
        self.connect()
        # 한글→영문 매핑 (테이블 생성과 동일)
        col_map = {
            '연도': 'year',
            '주차': 'week',
            '연령대': 'age_group',
            '의사환자 분율': 'ili',
            '입원환자 수': 'hospitalization',
            '아형': 'subtype',
            '인플루엔자 검출률': 'detection_rate',
            '예방접종률': 'vaccine_rate',
            '응급실 인플루엔자 환자': 'emergency_patients',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        # 결측치 None으로 변환
        df = df.where(pd.notnull(df), None)
        columns = list(df.columns)
        values = df.values.tolist()
        placeholders = ','.join(['%s'] * len(columns))
        col_names = ','.join([f'"{col}"' for col in columns])
        sql = f'INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})'
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, values)
        self.conn.commit()
        print(f"✅ 데이터 {len(df)}건 적재 완료: {table_name}")

    def create_table_from_dataframe(self, df: pd.DataFrame, table_name: str = "influenza_data", if_exists: str = "fail"):
        """
        DataFrame의 컬럼 정보를 기반으로 PostgreSQL 테이블 생성
        (한글 컬럼명은 영문으로 매핑 필요)
        """
        self.connect()
        # 한글→영문 매핑 (기존 col_map 활용)
        col_map = {
            '연도': 'year',
            '주차': 'week',
            '연령대': 'age_group',
            '의사환자 분율': 'ili',
            '입원환자 수': 'hospitalization',
            '아형': 'subtype',
            '인플루엔자 검출률': 'detection_rate',
            '예방접종률': 'vaccine_rate',
            '응급실 인플루엔자 환자': 'emergency_patients',
        }
        columns = []
        for col in df.columns:
            col_eng = col_map.get(col, col)
            # 간단한 타입 추론 (float, int, str)
            sample = df[col].dropna()
            if not sample.empty:
                v = sample.iloc[0]
                if isinstance(v, float):
                    col_type = 'DOUBLE PRECISION'
                elif isinstance(v, int):
                    col_type = 'INTEGER'
                else:
                    col_type = 'TEXT'
            else:
                col_type = 'TEXT'
            columns.append(f'"{col_eng}" {col_type}')
        col_defs = ', '.join(columns)
        sql = f'CREATE TABLE {table_name} ({col_defs})'
        if if_exists == "replace":
            with self.conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.commit()
        elif if_exists == "fail":
            # 이미 존재하면 생성하지 않음
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT to_regclass('{table_name}')")
                exists = cur.fetchone()[0]
            if exists:
                print(f"⚠️ 이미 테이블이 존재합니다: {table_name}")
                return
        with self.conn.cursor() as cur:
            cur.execute(sql)
        self.conn.commit()
        print(f"✅ 테이블 생성 완료: {table_name}")
    """시계열 데이터를 PostgreSQL로 관리하는 클래스"""
    def __init__(self, host=None, dbname=None, user=None, password=None, port=5432):
        self.host = host or os.getenv('PG_HOST', 'localhost')
        self.dbname = dbname or os.getenv('PG_DB', 'influenza')
        self.user = user or os.getenv('PG_USER', 'postgres')
        self.password = password or os.getenv('PG_PASSWORD', 'postgres')
        self.port = int(port or os.getenv('PG_PORT', 5432))
        self.conn = None

    def connect(self):
        if self.conn is None:
            self.conn = psycopg2.connect(
                host=self.host,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                port=self.port
            )
            print(f"✅ PostgreSQL 연결됨: {self.dbname}@{self.host}:{self.port}")
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            print("✅ PostgreSQL 연결 종료됨")
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.close()
    
    def import_csv_to_db(
        self,
        csv_path: str,
        table_name: str = "influenza_data",
        chunk_size: int = 100000,
        show_progress: bool = True
    ):
        """
        대용량 CSV를 PostgreSQL 테이블로 임포트
        (DuckDB의 read_csv_auto와 달리 pandas+copy_from 사용)
        """
        # ...구현은 이후 단계에서 추가...
        pass
    
    def export_to_parquet(self, *args, **kwargs):
        """
        (DuckDB 전용 기능) PostgreSQL에서는 직접 Parquet 내보내기 미지원.
        필요시 pandas DataFrame으로 export 후 pyarrow 등으로 저장 가능.
        """
        print("⚠️ PostgreSQL은 Parquet 직접 내보내기를 지원하지 않습니다.")
        pass
    
    def load_data(
        self,
        table_name: str = "influenza_data",
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        PostgreSQL 테이블에서 데이터를 Pandas DataFrame으로 로드
        """
        self.connect()
        if columns is None:
            select_cols = "*"
        else:
            select_cols = ", ".join([f'"{col}"' for col in columns])
        query = f"SELECT {select_cols} FROM {table_name}"
        if where:
            query += f" WHERE {where}"
        if order_by:
            query += f" ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"
        print(f"\n📊 데이터 로드 중...")
        print(f"쿼리: {query[:100]}{'...' if len(query) > 100 else ''}")
        start_time = time.time()
        df = pd.read_sql_query(query, self.conn)
        elapsed = time.time() - start_time
        print(f"✅ 로드 완료: {df.shape[0]:,} 행 × {df.shape[1]} 열 ({elapsed:.2f}초)\n")
        return df
    
    def get_table_info(self, table_name: str = "influenza_data"):
        """
        테이블 정보 출력 (컬럼명, 데이터 타입, 샘플 데이터)
        """
        self.connect()
        print(f"\n{'='*60}")
        print(f"📋 테이블 정보: {table_name}")
        print(f"{'='*60}")
        # PostgreSQL에서 컬럼 정보 조회
        query = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"
        schema = pd.read_sql_query(query, self.conn)
        print("\n컬럼 정보:")
        print(schema.to_string(index=False))
        # 행 수
        row_count = pd.read_sql_query(f"SELECT COUNT(*) FROM {table_name}", self.conn).iloc[0, 0]
        print(f"\n총 행 수: {row_count:,}")
        # 샘플 데이터
        print("\n샘플 데이터 (처음 5행):")
        sample = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", self.conn)
        print(sample.to_string(index=False))
        print(f"{'='*60}\n")
    
    def create_indices(self, table_name: str = "influenza_data", columns: Optional[list] = None):
        """
        PostgreSQL 인덱스 생성 (명시적으로 지정 필요)
        """
        self.connect()
        if columns:
            for col in columns:
                sql = f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{col} ON {table_name} (\"{col}\");"
                with self.conn.cursor() as cur:
                    cur.execute(sql)
            self.conn.commit()
            print(f"✅ 인덱스 생성 완료: {columns}")
        else:
            print("⚠️ 인덱스 생성할 컬럼을 지정하세요.")
    
    def optimize_database(self):
        """PostgreSQL 데이터베이스 최적화 (ANALYZE)"""
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute("ANALYZE;")
        self.conn.commit()
        print(f"✅ PostgreSQL ANALYZE 완료\n")



def load_from_postgres(
    table_name: str = "influenza_data",
    **kwargs
) -> pd.DataFrame:
    """
    편의 함수: PostgreSQL에서 데이터 로드
    """
    db = TimeSeriesDB()
    try:
        db.connect()
        return db.load_data(table_name, **kwargs)
    finally:
        db.close()


def fetch_latest_data_from_api(api_url: str = None, dataset_ids: List[str] = None) -> pd.DataFrame:
    """
    API를 통해 최신 데이터를 가져옴
    
    Parameters:
    -----------
    api_url : str, optional
        API 서버 URL (기본값: 환경변수 API_URL)
    dataset_ids : List[str], optional
        가져올 데이터셋 ID 리스트
    
    Returns:
    --------
    pd.DataFrame
        API에서 가져온 최신 데이터
    """
    if api_url is None:
        api_url = os.getenv('API_URL', 'http://localhost:3000')
    
    if dataset_ids is None:
        # 기본 데이터셋 ID 리스트
        dataset_ids = [
            'ds_0101', 'ds_0102', 'ds_0103', 'ds_0104', 'ds_0105', 
            'ds_0106', 'ds_0107', 'ds_0108', 'ds_0109', 'ds_0110',
            'ds_0701', 'ds_0801', 'ds_0901'
        ]
    
    print(f"\n{'='*60}")
    print(f"🌐 API에서 최신 데이터 가져오기")
    print(f"{'='*60}")
    print(f"API URL: {api_url}")
    print(f"데이터셋 개수: {len(dataset_ids)}")
    
    all_dataframes = []
    
    for idx, dsid in enumerate(dataset_ids, 1):
        print(f"\n[{idx}/{len(dataset_ids)}] {dsid} 로딩 중...")
        
        try:
            request_url = f"{api_url}/download"
            request_body = {"dsid": dsid, "returnData": True}  # 데이터를 직접 반환 요청
            
            response = requests.post(
                request_url,
                json=request_body,
                timeout=300
            )
            
            if response.status_code != 200:
                print(f"  ⚠️ API 요청 실패: {response.status_code}")
                continue
            
            result = response.json()
            if not result.get('ok'):
                print(f"  ⚠️ API 에러: {result.get('error')}")
                continue
            
            # API 응답에서 직접 데이터 가져오기 (파일 저장하지 않음)
            api_data = result.get('result', {}).get('data', [])
            
            if api_data:
                df = pd.DataFrame(api_data)
                df['dataset_id'] = dsid
                all_dataframes.append(df)
                print(f"  ✅ 완료: {len(api_data)} 레코드 (메모리에서 직접 처리)")
        
        except Exception as e:
            print(f"  ⚠️ 오류: {e}")
            continue
        
        # 서버 부하 방지를 위한 대기
        time.sleep(0.5)
    
    if not all_dataframes:
        print(f"\n⚠️ 가져온 데이터가 없습니다!")
        return pd.DataFrame()
    
    # 모든 데이터 병합
    df_latest = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n✅ 최신 데이터 병합 완료: {df_latest.shape}")
    
    return df_latest


def load_historical_data(before_dir: str = 'data/before') -> pd.DataFrame:
    """
    과거 데이터 CSV 파일들을 로드하고 병합
    
    Parameters:
    -----------
    before_dir : str
        과거 데이터가 저장된 디렉토리
    
    Returns:
    --------
    pd.DataFrame
        병합된 과거 데이터
    """
    print(f"\n{'='*60}")
    print(f"📂 과거 데이터 로딩")
    print(f"{'='*60}")
    print(f"디렉토리: {before_dir}")
    
    before_path = Path(before_dir)
    if not before_path.exists():
        print(f"⚠️ 디렉토리가 없습니다: {before_dir}")
        return pd.DataFrame()
    
    csv_files = list(before_path.glob("*.csv"))
    print(f"발견된 CSV 파일: {len(csv_files)}개")
    
    if not csv_files:
        return pd.DataFrame()
    
    # 데이터셋 ID별로 분류
    data_by_dsid = {}
    
    for filepath in csv_files:
        filename = filepath.name
        # 파일명 파싱: flu-0105-2022.csv -> dsid=0105, year=2022
        parts = filename.replace('.csv', '').split('-')
        if len(parts) != 3:
            continue
        
        dsid = f"ds_{parts[1]}"
        
        try:
            df = pd.read_csv(filepath)
            if dsid not in data_by_dsid:
                data_by_dsid[dsid] = []
            data_by_dsid[dsid].append(df)
        except Exception as e:
            print(f"  ⚠️ 파일 읽기 실패 ({filename}): {e}")
    
    # 데이터셋별로 병합
    all_dataframes = []
    for dsid, df_list in data_by_dsid.items():
        df_combined = pd.concat(df_list, ignore_index=True)
        df_combined['dataset_id'] = dsid
        all_dataframes.append(df_combined)
        print(f"  {dsid}: {len(df_list)}개 파일, {len(df_combined)} 레코드")
    
    if not all_dataframes:
        return pd.DataFrame()
    
    df_historical = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n✅ 과거 데이터 병합 완료: {df_historical.shape}")
    
    return df_historical


def consolidate_by_year_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터셋별 데이터를 올바르게 통합:
    - 연도+주차+연령대를 기본 키로 사용하여 연령대별 데이터 유지
    - 아형 데이터: 우세 아형을 각 연령대 행에 추가
    - 입원환자 수: 같은 키를 가진 여러 데이터셋의 값을 합산
    
    Parameters:
    -----------
    df : pd.DataFrame
        병합할 데이터프레임
    
    Returns:
    --------
    pd.DataFrame
        올바르게 통합된 데이터프레임
    """
    print("\n🔄 데이터 통합 중...")
    print(f"통합 전: {len(df)} 행")
    
    # 연도와 주차 컬럼이 있는지 확인
    if '연도' not in df.columns or '주차' not in df.columns:
        print("⚠️ '연도' 또는 '주차' 컬럼이 없습니다. 통합하지 않고 반환합니다.")
        return df
    
    # 메타데이터 컬럼 제거
    meta_columns = ['dsId', 'origin', 'contentType', 'originalData', 'parsedData', 'collectedAt', 'id']
    columns_to_drop = [col for col in meta_columns if col in df.columns]
    
    if columns_to_drop:
        print(f"메타데이터 컬럼 제거: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    
    # 1단계: 아형 데이터 추출 (연도+주차별 우세 아형)
    dominant_subtypes = pd.DataFrame()
    if '아형' in df.columns and '인플루엔자 검출률' in df.columns:
        print("\n[1단계] 아형 데이터 처리: 연도/주차별 최고 검출률 아형 선택")
        
        # '검출률' 값 제거 및 아형 데이터만 추출
        df_subtype = df[(df['아형'].notna()) & (df['아형'] != '검출률')].copy()
        
        if not df_subtype.empty:
            # 각 연도/주차에서 가장 높은 검출률을 가진 아형 찾기
            idx_max = df_subtype.groupby(['연도', '주차'])['인플루엔자 검출률'].idxmax()
            dominant_subtypes = df_subtype.loc[idx_max, ['연도', '주차', '아형']].copy()
            print(f"  추출된 우세 아형: {len(dominant_subtypes)} 건")
            
            # 아형 행 제거 (연령대 기반 행만 유지)
            df = df[df['아형'].isna() | (df['아형'] == '검출률')].copy()
            if '아형' in df.columns:
                df = df.drop(columns=['아형'])
    
    # 2단계: 연령대 기반 데이터 통합
    print(f"\n[2단계] 연령대별 데이터 통합")
    
    # 그룹화 키: 연도, 주차, 연령대
    if '연령대' not in df.columns:
        print("⚠️ '연령대' 컬럼이 없습니다.")
        groupby_cols = ['연도', '주차']
    else:
        groupby_cols = ['연도', '주차', '연령대']
    
    # 각 컬럼별 집계 방식 정의
    aggregation_dict = {}
    
    for col in df.columns:
        if col in groupby_cols:
            continue
        elif col == 'dataset_id':
            # dataset_id는 나중에 제거
            aggregation_dict[col] = lambda x: ', '.join(sorted(set(str(v) for v in x if pd.notna(v))))
        elif col == '입원환자 수':
            # 입원환자 수는 합산 (ds_0103 + ds_0104)
            def sum_patients(x):
                values = [v for v in x if pd.notna(v)]
                if not values:
                    return None
                # 숫자로 변환 가능한 값만 합산
                numeric_values = []
                for v in values:
                    try:
                        numeric_values.append(float(v))
                    except:
                        pass
                return sum(numeric_values) if numeric_values else None
            
            aggregation_dict[col] = sum_patients
        elif col == '응급실 인플루엔자 환자':
            # 응급실 환자도 합산
            def sum_emergency(x):
                values = [v for v in x if pd.notna(v)]
                if not values:
                    return None
                numeric_values = []
                for v in values:
                    try:
                        numeric_values.append(float(v))
                    except:
                        pass
                return sum(numeric_values) if numeric_values else None
            
            aggregation_dict[col] = sum_emergency
        elif col in ['의사환자 분율', '예방접종률']:
            # 평균값 사용
            aggregation_dict[col] = lambda x: pd.Series([v for v in x if pd.notna(v)]).mean() if any(pd.notna(v) for v in x) else None
        else:
            # 기타: 첫 번째 유효값
            aggregation_dict[col] = lambda x: next((v for v in x if pd.notna(v)), None)
    
    # 그룹화 및 집계
    df_consolidated = df.groupby(groupby_cols, as_index=False).agg(aggregation_dict)
    
    # 3단계: 우세 아형 정보 병합
    if not dominant_subtypes.empty:
        print(f"\n[3단계] 우세 아형 정보 병합")
        df_consolidated = pd.merge(
            df_consolidated, 
            dominant_subtypes, 
            on=['연도', '주차'], 
            how='left'
        )
        print(f"  아형 정보 추가 완료")
    
    # dataset_id 컬럼 제거
    if 'dataset_id' in df_consolidated.columns:
        df_consolidated = df_consolidated.drop(columns=['dataset_id'])
    
    print(f"\n통합 후: {len(df_consolidated)} 행")
    
    # 통합 결과 요약
    if '연령대' in df_consolidated.columns:
        age_groups = df_consolidated['연령대'].unique()
        print(f"고유 연령대: {len(age_groups)}개 - {', '.join(sorted(age_groups)[:10])}")
    
    if '아형' in df_consolidated.columns:
        subtypes = df_consolidated['아형'].value_counts()
        print(f"\n아형 분포:")
        for subtype, count in subtypes.items():
            print(f"  {subtype}: {count}건")
    
    # 한글→영문 컬럼명 매핑 (누락 방지)
    col_map = {
        '연도': 'year',
        '주차': 'week',
        '연령대': 'age_group',
        '의사환자 분율': 'ili',
        '입원환자 수': 'hospitalization',
        '아형': 'subtype',
        '인플루엔자 검출률': 'detection_rate',
        '예방접종률': 'vaccine_rate',
        '응급실 인플루엔자 환자': 'emergency_patients',
    }
    df_consolidated = df_consolidated.rename(columns=col_map)
    return df_consolidated


def merge_and_update_database(
    table_name: str = "influenza_data",
    fetch_latest: bool = True,
    api_url: str = None,
    before_dir: str = 'data/before',
    consolidate: bool = True
):
    """
    API, 과거 데이터, 기존 데이터를 병합하여 PostgreSQL에 업데이트
    
    Parameters
    ----------
    table_name : str
        PostgreSQL 테이블 이름
    fetch_latest : bool
        API에서 최신 데이터 가져올지 여부
    api_url : str
        API URL (None이면 환경변수 사용)
    before_dir : str
        과거 데이터 디렉토리
    consolidate : bool
        같은 연도/주차 데이터를 한 행으로 통합할지 여부 (기본: True)
    """
    print("\n" + "="*60)
    print("🔄 데이터 병합 및 PostgreSQL 업데이트 프로세스")
    print("="*60)
    
    all_data = []
    
    # 1. API에서 최신 데이터 가져오기
    if fetch_latest:
        print("\n[단계 1/4] API에서 최신 데이터 가져오기")
        df_latest = fetch_latest_data_from_api(api_url=api_url)
        if not df_latest.empty:
            all_data.append(df_latest)
            print(f"✅ 최신 데이터: {df_latest.shape}")
        else:
            print("⚠️ 최신 데이터 없음")
    else:
        print("\n[단계 1/4] 최신 데이터 가져오기 건너뜀")
    
    # 2. 과거 데이터 로딩
    print("\n[단계 2/4] 과거 데이터 로딩")
    df_historical = load_historical_data(before_dir=before_dir)
    if not df_historical.empty:
        all_data.append(df_historical)
        print(f"✅ 과거 데이터: {df_historical.shape}")
    else:
        print("⚠️ 과거 데이터 없음")
    
    # 3. 모든 데이터 병합
    print("\n[단계 3/4] 데이터 병합")
    if not all_data:
        print("⚠️ 병합할 데이터가 없습니다!")
        return
    
    df_merged = pd.concat(all_data, ignore_index=True)
    print(f"초기 병합 데이터: {df_merged.shape}")
    
    # 3-1. 데이터 통합 (같은 연도/주차를 한 행으로)
    if consolidate:
        df_merged = consolidate_by_year_week(df_merged)
    
    print(f"\n최종 병합 데이터: {df_merged.shape}")
    
    # PostgreSQL에 저장
    with TimeSeriesDB() as db:
        print(f"\nPostgreSQL에 저장 중...")
        start_time = time.time()
        
        # 기존 테이블 삭제 후 새로 생성
        with db.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db.conn.commit()
        
        # 테이블 생성 및 데이터 삽입
        db.create_table_from_dataframe(df_merged, table_name, if_exists="replace")
        db.insert_dataframe(df_merged, table_name)
        
        elapsed = time.time() - start_time
        with db.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cur.fetchone()[0]
        
        print(f"✅ PostgreSQL 저장 완료!")
        print(f"   • 테이블: {table_name}")
        print(f"   • 행 수: {row_count:,}")
        print(f"   • 소요 시간: {elapsed:.2f}초")
    
    # CSV로도 저장 (백업)
    csv_output = "merged_influenza_data.csv"
    print(f"\nCSV 백업 저장 중: {csv_output}")
    df_merged.to_csv(csv_output, index=False)
    csv_size_mb = Path(csv_output).stat().st_size / (1024 * 1024)
    print(f"✅ CSV 저장 완료: {csv_size_mb:.1f} MB")
    
    print("\n" + "="*60)
    print("✅ 모든 작업 완료!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # 사용 예제
    print("=" * 60)
    print("PostgreSQL 시계열 데이터 관리 유틸리티")
    print("=" * 60)
    
    # 명령행 인자 확인
    if len(sys.argv) > 1 and sys.argv[1] == "--update":
        # 업데이트 모드: API + 과거 데이터 + 기존 데이터 병합
        print("\n🔄 업데이트 모드: API에서 최신 데이터 가져와서 병합")
        merge_and_update_database(
            table_name="influenza_data",
            fetch_latest=True,
            before_dir='data/before'
        )
    else:
        # 기본 모드: 기존 CSV를 PostgreSQL로 변환
        csv_file = "merged_influenza_data.csv"
        if Path(csv_file).exists():
            print(f"\n📄 CSV 파일을 PostgreSQL로 변환 중: {csv_file}")
            
            # CSV 데이터 로드
            df = pd.read_csv(csv_file)
            
            # PostgreSQL에 저장
            with TimeSeriesDB() as db:
                db.create_table_from_dataframe(df, "influenza_data", if_exists="replace")
                db.insert_dataframe(df, "influenza_data")
            
            print("\n" + "=" * 60)
            print("사용 예제:")
            print("=" * 60)
            print("""
# 전체 데이터 로드
df = load_from_postgres()

# 특정 컬럼만 로드
df = load_from_postgres(columns=['year', 'week', 'ili'])

# 조건부 로드
df = load_from_postgres(where="year >= 2020")

# 최근 1000개 데이터만
df = load_from_postgres(limit=1000, order_by="year DESC, week DESC")

# API + 과거 데이터 + 기존 데이터 병합하여 업데이트
python db_utils.py --update
            """)
        else:
            print(f"\n⚠️ CSV 파일을 찾을 수 없습니다: {csv_file}")
            print("\n💡 API에서 데이터를 가져와 PostgreSQL을 생성하려면:")
            print("   python db_utils.py --update")
