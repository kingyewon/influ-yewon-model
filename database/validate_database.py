"""
데이터베이스 병합 검증 스크립트
- 원본 CSV 데이터와 병합된 DuckDB 데이터 비교
- 데이터 손실 및 병합 오류 검증
"""

import pandas as pd
from pathlib import Path
from db_utils import load_from_postgres
from tabulate import tabulate

def load_original_csvs(before_dir='/Volumes/ExternalSSD/Workspace/influenza-prediction-model/data/before'):
    """원본 CSV 파일들을 로드하여 데이터셋별로 분류"""
    print("\n" + "="*100)
    print("📂 원본 CSV 파일 로드 중...")
    print("="*100)
    
    before_path = Path(before_dir)
    csv_files = sorted(before_path.glob("*.csv"))
    
    print(f"발견된 CSV 파일: {len(csv_files)}개\n")
    
    # 데이터셋별로 분류
    data_by_dsid = {}
    
    for filepath in csv_files:  # 모든 파일 로드
        filename = filepath.name
        
        # 파일명 파싱
        parts = filename.replace('.csv', '').split('-')
        if len(parts) != 3:
            continue
        
        dsid = f"ds_{parts[1]}"
        year = parts[2]
        
        try:
            df = pd.read_csv(filepath)
            
            if dsid not in data_by_dsid:
                data_by_dsid[dsid] = []
            data_by_dsid[dsid].append({
                'year': year,
                'filename': filename,
                'data': df,
                'columns': list(df.columns)
            })
        except Exception as e:
            print(f"   ⚠️ {filename} 읽기 오류: {e}\n")
    
    return data_by_dsid


def validate_merge():
    """병합 과정 검증"""
    
    print("\n" + "="*100)
    print("🔍 데이터베이스 병합 검증")
    print("="*100)
    
    # 1. 원본 CSV 데이터 로드
    print("\n[1단계] 원본 CSV 데이터 로드")
    original_data = load_original_csvs()
    
    # 2. PostgreSQL 데이터 로드
    print("\n[2단계] PostgreSQL 데이터 로드")
    from db_utils import load_from_postgres
    db_data = load_from_postgres()
    print(db_data.columns)
    
    print(f"\n병합된 데이터베이스:")
    print(f"  - 행 수: {len(db_data)}")
    print(f"  - 컬럼: {list(db_data.columns)}")
    # 컬럼명 매핑 (한글→영문, 영문→한글 모두 지원)
    col_map = {
        '연도': 'year', '주차': 'week', '연령대': 'age_group', '의사환자 분율': 'ili',
        '입원환자 수': 'hospitalization', '아형': 'subtype', '인플루엔자 검출률': 'detection_rate',
        '예방접종률': 'vaccine_rate', '응급실 인플루엔자 환자': 'emergency_patients'
    }
    # 역방향도 추가
    col_map.update({v: k for k, v in col_map.items()})
    def get_col(df, *candidates):
        for c in candidates:
            if c in df.columns:
                return c
            if col_map.get(c) and col_map[c] in df.columns:
                return col_map[c]
        raise KeyError(f"컬럼 후보 {candidates} 중 해당되는 컬럼이 없습니다: {df.columns}")

    # 연도/주차 컬럼명 동적 접근
    year_col = get_col(db_data, '연도', 'year')
    week_col = get_col(db_data, '주차', 'week')
    print(f"  - 연도 범위: {db_data[year_col].min():.0f} ~ {db_data[year_col].max():.0f}")
    
    # 3. 샘플 비교
    print("\n[3단계] 데이터 샘플 비교")
    print("\n" + "="*100)
    print("🔍 특정 연도/주차의 원본 데이터 vs 병합 데이터 비교")
    print("="*100)
    
    # 2017년 36주 데이터 비교
    test_year = 2017
    test_week = 36
    
    db_sample = db_data[(db_data['year'] == test_year) & (db_data['week'] == test_week)]
    
    print(f"\n병합된 데이터 ({test_year}년 {test_week}주):")
    print(tabulate(db_sample, headers='keys', tablefmt='simple', showindex=False))
    
    # 4. 각 데이터셋별 원본 확인
    print("\n[4단계] 데이터셋별 원본 데이터 확인")
    print("\n" + "="*100)
    print("📊 각 dataset_id의 원본 데이터 구조")
    print("="*100)
    
    for dsid, files in original_data.items():
        print(f"\n🔹 {dsid}:")
        if files:
            sample_df = files[0]['data']
            print(f"   컬럼 목록: {list(sample_df.columns)}")
            print(f"   데이터 샘플 (처음 3행):")
            print(f"   {sample_df.head(3).to_string(index=False, max_colwidth=30)}")
    
    # 5. 컬럼별 결측치 원인 분석
    print("\n[5단계] 컬럼별 데이터 존재 여부 분석")
    print("\n" + "="*100)
    print("📊 각 데이터셋이 가진 컬럼 맵핑")
    print("="*100)
    
    column_mapping = {}
    for dsid, files in original_data.items():
        if files:
            cols = set(files[0]['data'].columns)
            column_mapping[dsid] = cols
            print(f"\n{dsid}:")
            print(f"  {', '.join(sorted(cols))}")
    
    # 6. 병합 후 결측치 분석
    print("\n[6단계] 병합 후 결측치 분석")
    print("\n" + "="*100)
    print("📊 각 컬럼의 결측치 현황")
    print("="*100)
    
    missing_info = []
    for col in db_data.columns:
        missing_count = db_data[col].isna().sum()
        missing_pct = (missing_count / len(db_data) * 100)
        has_data_count = len(db_data) - missing_count
        
        missing_info.append({
            '컬럼명': col,
            '유효 데이터': has_data_count,
            '결측치': missing_count,
            '결측치(%)': f"{missing_pct:.1f}%"
        })
    
    print(tabulate(missing_info, headers='keys', tablefmt='simple', showindex=False))
    
    # 7. 응급실 인플루엔자 환자 데이터 추적
    print("\n[7단계] '응급실 인플루엔자 환자' 데이터 추적")
    print("\n" + "="*100)
    print("🔍 원본 CSV에서 응급실 데이터 검색")
    print("="*100)
    
    for dsid, files in original_data.items():
        for file_info in files[:1]:  # 각 데이터셋의 첫 파일만
            df = file_info['data']
            # 응급실 관련 컬럼 찾기
            emergency_cols = [col for col in df.columns if '응급' in col or 'emergency' in col.lower()]
            if emergency_cols:
                print(f"\n✅ {dsid} ({file_info['filename']}):")
                print(f"   응급실 관련 컬럼: {emergency_cols}")
                print(f"   샘플 데이터:")
                print(f"   {df[emergency_cols].head(5).to_string(index=False)}")
            else:
                print(f"\n❌ {dsid}: 응급실 관련 컬럼 없음")
    
    # 8. 아형 데이터 다양성 확인
    print("\n[8단계] 'subtype' 데이터 다양성 확인")
    print("\n" + "="*100)
    print("🔍 원본 CSV 및 병합 데이터에서 subtype 다양성 확인")
    print("="*100)
    
    # 원본 CSV에서 subtype 데이터 확인
    for dsid, files in original_data.items():
        has_subtype = False
        for file_info in files[:5]:  # 각 데이터셋의 처음 5개 파일
            if 'subtype' in file_info['columns']:
                if not has_subtype:
                    print(f"\n✅ {dsid}:")
                    has_subtype = True
                
                df = file_info['data']
                unique_subtypes = df['subtype'].unique()
                print(f"   {file_info['year']}년: {len(unique_subtypes)}개 subtype - {', '.join(map(str, unique_subtypes[:10]))}")
    
    # 병합된 데이터에서 subtype 확인
    print(f"\n병합된 데이터베이스의 subtype 다양성:")
    unique_db_subtypes = db_data['subtype'].unique()
    print(f"  총 {len(unique_db_subtypes)}개의 고유 subtype:")
    for subtype in unique_db_subtypes[:20]:
        count = (db_data['subtype'] == subtype).sum()
        print(f"    - {subtype}: {count}건")
    
    # 9. 2017년 36주 데이터 상세 비교
    print("\n[9단계] 2017년 36주 데이터 상세 비교")
    print("\n" + "="*100)
    print("🔍 원본 데이터와 병합 데이터 비교")
    print("="*100)
    
    test_year = 2017
    test_week = 36
    
    print(f"\n원본 CSV 데이터 ({test_year}년 {test_week}주):")
    for dsid, files in sorted(original_data.items()):
        for file_info in files:
            if file_info['year'] == str(test_year):
                df = file_info['data']
                week_data = df[df['week'] == test_week] if 'week' in df.columns else pd.DataFrame()
                if not week_data.empty:
                    print(f"\n  {dsid} ({file_info['filename']}):")
                    print(f"  {week_data.to_string(index=False, max_colwidth=30)}")
    
    print(f"\n병합된 데이터 ({test_year}년 {test_week}주):")
    db_sample = db_data[(db_data['year'] == test_year) & (db_data['week'] == test_week)]
    print(tabulate(db_sample, headers='keys', tablefmt='simple', showindex=False, maxcolwidths=30))
    
    # 10. 문제점 검증
    print("\n[10단계] 데이터 무결성 검증")
    print("\n" + "="*100)
    print("🔍 병합 전후 데이터 비교 및 문제점 검증")
    print("="*100)
    
    issues = []
    
    # 문제 1: 연령대 손실 확인
    age_group_loss = False
    if 'age_group' in db_data.columns:
        unique_ages = db_data['age_group'].nunique()
        if unique_ages < 7:  # 최소 7개 연령대는 있어야 함
            age_group_loss = True
            issues.append(f"연령대 데이터 손실: {unique_ages}개만 존재 (예상: 7개 이상)")
    
    # 문제 2: subtype 다양성 확인
    subtype_loss = False
    if 'subtype' in db_data.columns:
        unique_subtypes = db_data['subtype'].nunique()
        if unique_subtypes < 3:  # 최소 3개 subtype (A(H1N1)pdm09, A(H3N2), B)
            subtype_loss = True
            issues.append(f"subtype 데이터 손실: {unique_subtypes}개만 존재 (예상: 3개 이상)")
    
    # 문제 3: 입원환자 수 합산 확인 (2017년 36주 예시)
    test_sample = db_data[(db_data['year'] == 2017) & (db_data['week'] == 36) & (db_data['age_group'] == '65세이상')]
    if not test_sample.empty and 'hospitalization' in test_sample.columns:
        merged_patients = test_sample['hospitalization'].iloc[0]
        # 원본: ds_0103=8, ds_0104=1 -> 합계 9
        if merged_patients < 9:
            issues.append(f"입원환자 수 합산 오류: 2017년 36주 65세이상 {merged_patients}명 (예상: 9명)")
    
    # 문제 4: 데이터 과도한 축소 확인
    expected_min_rows = 3000  # 최소 3000행 이상은 있어야 함
    if len(db_data) < expected_min_rows:
        issues.append(f"데이터 과도한 축소: {len(db_data)}행 (예상: {expected_min_rows}행 이상)")
    
    # 문제 5: 필수 컬럼 누락 확인
    required_columns = ['year', 'week', 'age_group', 'ili', 'hospitalization', 'subtype', 'detection_rate']
    missing_columns = [col for col in required_columns if col not in db_data.columns]
    if missing_columns:
        issues.append(f"필수 컬럼 누락: {', '.join(missing_columns)}")
    
    # 문제 6: 결측치 비율이 너무 높은 컬럼 확인 (80% 이상)
    high_missing_cols = []
    for col in db_data.columns:
        if col in ['year', 'week', 'age_group', 'subtype']:  # 필수 키 컬럼은 제외
            continue
        missing_rate = db_data[col].isna().sum() / len(db_data) * 100
        if missing_rate > 80:
            high_missing_cols.append(f"{col} ({missing_rate:.1f}%)")
    
    if high_missing_cols:
        issues.append(f"과도한 결측치 발견 (>80%): {', '.join(high_missing_cols)}")
    
    # 결과 출력
    print("\n" + "="*100)
    if issues:
        print("⚠️ 문제점 발견!")
        print("="*100)
        for i, issue in enumerate(issues, 1):
            print(f"\n문제점 {i}: {issue}")
    else:
        print("✅ 문제점 없음!")
        print("="*100)
        print("\n모든 데이터가 올바르게 병합되었습니다:")
        print(f"  • 총 행 수: {len(db_data):,}행")
        print(f"  • 고유 연령대: {db_data['age_group'].nunique()}개")
        print(f"  • 고유 subtype: {db_data['subtype'].nunique()}개")
        if not test_sample.empty:
            print(f"  • 입원환자 수 합산: 정상 (2017년 36주 65세이상 {merged_patients}명)")
    
    print("\n" + "="*100)
    print("✅ 검증 완료!")
    print("="*100)


if __name__ == "__main__":
    validate_merge()
