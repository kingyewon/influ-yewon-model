
"""
PostgreSQL 업데이트 프로세스 테스트

이 스크립트는 다음 작업을 수행합니다:
1. API에서 최신 데이터 가져오기
2. data/before 폴더의 과거 데이터 로딩
3. 모든 데이터 병합
4. PostgreSQL에 저장
"""

try:
    # 모듈로서 import 될 때
    from .db_utils import merge_and_update_database
except ImportError:
    # 직접 실행될 때
    from db_utils import merge_and_update_database

import os

if __name__ == "__main__":
    print("\n" + "="*60)
    print("📊 인플루엔자 데이터 병합 및 PostgreSQL 업데이트")
    print("="*60)
    
    # 환경 확인
    print("\n환경 확인:")
    print(f"  • API_URL: {os.getenv('API_URL', 'http://localhost:3000')}")
    print(f"  • data/before 폴더 존재: {os.path.exists('data/before')}")
    print(f"  • merged_influenza_data.csv 존재: {os.path.exists('merged_influenza_data.csv')}")
    
    # 사용자 확인
    print("\n다음 작업을 수행합니다:")
    print("  1. API에서 최신 데이터 가져오기")
    print("  2. data/before 폴더의 과거 데이터 로딩")
    print("  3. 모든 데이터 병합 및 중복 제거")
    print("  4. PostgreSQL에 저장")
    print("  5. CSV로 백업")
    
    response = input("\n계속하시겠습니까? (y/n): ").lower()
    
    if response == 'y':
        merge_and_update_database(
            table_name="influenza_data",
            fetch_latest=True,  # API에서 최신 데이터 가져오기
            before_dir='data/before',
            consolidate=True  # 수정된 병합 로직 사용
        )
        print("\n" + "="*60)
        print("✅ 작업 완료!")
        print("="*60)
        print("\n다음 명령어로 데이터를 사용할 수 있습니다:")
        print("  python patchTST.py")
    else:
        print("\n취소되었습니다.")
