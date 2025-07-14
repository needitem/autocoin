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
        """데이터베이스 초기화 (성능 최적화)"""
        try:
            # WAL 모드와 성능 최적화 설정
            self.db_conn = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                timeout=30.0
            )
            
            # 성능 최적화 설정
            cursor = self.db_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            cursor.execute("PRAGMA synchronous=NORMAL")  # 빠른 쓰기
            cursor.execute("PRAGMA cache_size=10000")  # 캐시 크기 증가
            cursor.execute("PRAGMA temp_store=MEMORY")  # 메모리 임시 저장
            cursor.execute("PRAGMA mmap_size=268435456")  # 256MB 메모리 맵
            
            # market_data 테이블 생성 (인덱스 포함)
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
            
            # 성능 최적화 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_timestamp 
                ON market_data(market, timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON market_data(timestamp DESC)
            """)
            
            # cache 테이블 생성 (인덱스 포함)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expires_at REAL
                )
            """)
            
            # 캐시 만료 시간 인덱스
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expires 
                ON cache(expires_at)
            """)
            
            self.db_conn.commit()
            self.logger.info("데이터베이스 초기화 완료 (최적화 적용)")
            
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
        """시장 데이터 저장 (성능 최적화)"""
        try:
            if not self.check_data_consistency(data):
                return False

            with sqlite3.connect(self.db_path) as conn:
                # 성능 최적화 설정
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                
                # timestamp를 인덱스에서 컬럼으로 이동
                df_to_save = data.reset_index()
                
                # market 컬럼 추가
                df_to_save['market'] = market
                
                # 배치 삽입으로 성능 향상
                cursor = conn.cursor()
                
                # 기존 데이터 중복 체크 (최근 1시간만)
                cursor.execute("""
                    DELETE FROM market_data 
                    WHERE market = ? AND timestamp >= datetime('now', '-1 hour')
                """, (market,))
                
                # 배치 삽입
                insert_data = []
                for _, row in df_to_save.iterrows():
                    insert_data.append((
                        row['timestamp'], market, 
                        row['open'], row['high'], row['low'], 
                        row['close'], row['volume']
                    ))
                
                cursor.executemany("""
                    INSERT OR REPLACE INTO market_data 
                    (timestamp, market, open, high, low, close, volume) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, insert_data)
                
                conn.commit()
                self.logger.info(f"{len(data)}개의 데이터 포인트를 저장했습니다 (최적화)")
                return True
                
        except Exception as e:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(e)}")
            return False

    def load_market_data(self, market: str, start_time: str = None, end_time: str = None, limit: int = None) -> pd.DataFrame:
        """시장 데이터 로드 (성능 최적화)"""
        try:
            # 캐시 키 생성
            cache_key = f"market_data_{market}_{start_time}_{end_time}_{limit}"
            
            # 메모리 캐시 확인 (1분 캐시)
            from src.utils.performance_cache import _global_cache
            cached_df = _global_cache.get(cache_key)
            if cached_df is not None:
                self.logger.debug(f"캐시에서 데이터 로드: {market}")
                return cached_df
            
            # 최적화된 쿼리 생성
            query = """
                SELECT timestamp, open, high, low, close, volume 
                FROM market_data 
                WHERE market = ?
            """
            params = [market]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
                
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
                
            with sqlite3.connect(self.db_path) as conn:
                # 성능 최적화 설정
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA cache_size=10000")
                
                # 데이터 로드
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    self.logger.debug(f"데이터가 없습니다: {market}")
                    empty_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    return empty_df
                
                # 효율적인 데이터 처리
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # 데이터 타입 최적화
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                
                # 결과 캐싱 (1분)
                _global_cache.set(cache_key, df)
                
                self.logger.debug(f"{len(df)}개의 데이터 포인트를 로드했습니다 (최적화)")
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