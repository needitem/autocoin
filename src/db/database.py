"""
SQLite 기반 데이터베이스 매니저
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from src.utils.logger import get_logger, DEBUG, INFO

class DatabaseManager:
    def __init__(self, db_path: str = 'market_data.db', verbose: bool = False):
        """데이터베이스 매니저 초기화
        
        Args:
            db_path (str): SQLite 데이터베이스 파일 경로
            verbose (bool): 상세 로깅 여부
        """
        self.db_path = db_path
        self.verbose = verbose
        self.logger = get_logger('database')
        
        # 로깅 레벨 설정
        if verbose:
            self.logger.setLevel(DEBUG)
        else:
            self.logger.setLevel(INFO)
            
        self.db_conn = None
        self._initialize_db()

    def _initialize_db(self):
        """데이터베이스 초기화"""
        try:
            self.db_conn = sqlite3.connect(self.db_path)
            cursor = self.db_conn.cursor()
            
            # market_data 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    timestamp DATETIME,
                    market TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (timestamp, market)
                )
            """)
            
            # cache 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expires_at DATETIME
                )
            """)
            
            self.db_conn.commit()
            self.logger.info("데이터베이스 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
            raise

    def close(self):
        """데이터베이스 연결 종료"""
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None

    def invalidate_cache(self, key: str) -> bool:
        """캐시 무효화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"캐시 무효화 중 오류 발생: {str(e)}")
            return False

    def check_data_consistency(self, data: pd.DataFrame) -> bool:
        """데이터 일관성 검사"""
        try:
            # 필수 컬럼 확인
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.warning("필수 컬럼이 누락되었습니다")
                return False
            
            # 데이터 비어있는지 확인
            if data.empty:
                self.logger.warning("데이터가 비어있습니다")
                return False
            
            # 음수 값 확인
            for col in required_columns:
                if (data[col] < 0).any():
                    self.logger.warning(f"{col} 컬럼에 음수 값이 있습니다")
                    return False
            
            # 가격 관계 확인 (high >= low, high >= open, high >= close)
            if not (
                (data['high'] >= data['low']).all() and
                (data['high'] >= data['open']).all() and
                (data['high'] >= data['close']).all()
            ):
                self.logger.warning("가격 데이터 관계가 올바르지 않습니다")
                return False
            
            # 거래량이 음수가 아닌지 확인
            if (data['volume'] < 0).any():
                self.logger.warning("거래량에 음수 값이 있습니다")
                return False
            
            # NaN 값 확인
            if data[required_columns].isna().any().any():
                self.logger.warning("NaN 값이 있습니다")
                return False
            
            # 무한대 값 확인
            if data[required_columns].isin([float('inf'), float('-inf')]).any().any():
                self.logger.warning("무한대 값이 있습니다")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"데이터 일관성 검사 중 오류 발생: {str(e)}")
            return False

    def save_market_data(self, market: str, data: pd.DataFrame) -> bool:
        """시장 데이터 저장"""
        try:
            if not self.check_data_consistency(data):
                return False

            with sqlite3.connect(self.db_path) as conn:
                # timestamp를 인덱스에서 컬럼으로 이동
                df_to_save = data.reset_index()
                
                # market 컬럼 추가
                df_to_save['market'] = market
                
                # 데이터 저장
                df_to_save.to_sql('market_data', conn, if_exists='append', index=False)
                
                self.logger.info(f"{len(data)}개의 데이터 포인트를 저장했습니다")
                return True
                
        except Exception as e:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(e)}")
            return False

    def load_market_data(self, market: str, start_time: str = None, end_time: str = None) -> pd.DataFrame:
        """시장 데이터 로드"""
        try:
            # 쿼리 생성
            query = "SELECT * FROM market_data WHERE market = ?"
            params = [market]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
                
            query += " ORDER BY timestamp"
                
            with sqlite3.connect(self.db_path) as conn:
                # 데이터 로드
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    self.logger.warning(f"데이터가 없습니다: {market}")
                    return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # timestamp를 datetime으로 변환
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # market 컬럼 제거
                if 'market' in df.columns:
                    df = df.drop('market', axis=1)
                
                # 데이터 타입 변환
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
                
                self.logger.info(f"{len(df)}개의 데이터 포인트를 로드했습니다")
                return df
                
        except Exception as e:
            self.logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def set_cache(self, key: str, value: any, expires_in: int = 3600):
        """캐시 데이터 저장"""
        try:
            expires_at = datetime.now().timestamp() + expires_in
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
                    (key, json.dumps(value), str(expires_at))
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error setting cache: {str(e)}")

    def get_cache(self, key: str) -> any:
        """캐시 데이터 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT value, expires_at FROM cache WHERE key = ?",
                    (key,)
                )
                result = cursor.fetchone()
                
                if result:
                    value, expires_at = result
                    if float(expires_at) > datetime.now().timestamp():
                        return json.loads(value)
                    else:
                        # 만료된 캐시 삭제
                        cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                        conn.commit()
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting cache: {str(e)}")
            return None

    def delete_cache(self, key: str = None):
        """캐시 삭제"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if key:
                    cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                else:
                    cursor.execute("DELETE FROM cache")
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")

    def save_data(self, market: str, interval: str, data: str) -> bool:
        """데이터 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                key = f"{market}:{interval}"
                cursor.execute(
                    "INSERT OR REPLACE INTO market_data (market, data) VALUES (?, ?)",
                    (key, data)
                )
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            return False

    def get_data(self, market: str, interval: str) -> str:
        """데이터 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                key = f"{market}:{interval}"
                cursor.execute(
                    "SELECT data FROM market_data WHERE market = ?",
                    (key,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            self.logger.error(f"Error getting data: {str(e)}")
            return None

    def get_total_chunks(self, market: str) -> int:
        """총 청크 수 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM chunk_data WHERE market = ?",
                    (market,)
                )
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error getting total chunks: {str(e)}")
            return 0

    def get_chunk(self, market: str, chunk_id: int) -> str:
        """청크 데이터 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data FROM chunk_data WHERE market = ? AND chunk_id = ?",
                    (market, chunk_id)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            self.logger.error(f"Error getting chunk: {str(e)}")
            return None 