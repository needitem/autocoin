import sqlite3
from datetime import datetime
import json
from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path

class Database:
    def __init__(self, db_path: str = "crypto_data.db"):
        """데이터베이스 초기화"""
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """데이터베이스 테이블 생성"""
        with sqlite3.connect(self.db_path) as conn:
            # 뉴스 데이터 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_symbol TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT,
                    source TEXT,
                    url TEXT,
                    published_at TIMESTAMP,
                    sentiment_score REAL,
                    importance_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 키워드 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_symbol TEXT NOT NULL,
                    keyword TEXT NOT NULL,
                    category TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(coin_symbol, keyword, category)
                )
            """)

            # 시장 데이터 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 분석 결과 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_symbol TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    result_data TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 인덱스 생성
            conn.execute("CREATE INDEX IF NOT EXISTS idx_news_coin ON news(coin_symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_news_date ON news(published_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_keywords_coin ON keywords(coin_symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_coin ON market_data(coin_symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data(timestamp)")

    def save_news(self, news_data: Dict) -> bool:
        """뉴스 데이터 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO news (
                        coin_symbol, title, content, source, url, 
                        published_at, sentiment_score, importance_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    news_data['coin_symbol'],
                    news_data['title'],
                    news_data.get('content'),
                    news_data.get('source'),
                    news_data.get('url'),
                    news_data.get('published_at'),
                    news_data.get('sentiment_score'),
                    news_data.get('importance_score')
                ))
                return True
        except Exception as e:
            print(f"뉴스 저장 실패: {str(e)}")
            return False

    def get_recent_news(self, coin_symbol: str, hours: int = 24) -> List[Dict]:
        """최근 뉴스 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM news 
                    WHERE coin_symbol = ? 
                    AND published_at >= datetime('now', ?) 
                    ORDER BY published_at DESC
                """, (coin_symbol, f'-{hours} hours'))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"뉴스 조회 실패: {str(e)}")
            return []

    def save_market_data(self, market_data: Dict) -> bool:
        """시장 데이터 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO market_data (
                        coin_symbol, price, volume, timestamp,
                        open_price, high_price, low_price, close_price
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market_data['coin_symbol'],
                    market_data['price'],
                    market_data['volume'],
                    market_data['timestamp'],
                    market_data.get('open_price'),
                    market_data.get('high_price'),
                    market_data.get('low_price'),
                    market_data.get('close_price')
                ))
                return True
        except Exception as e:
            print(f"시장 데이터 저장 실패: {str(e)}")
            return False

    def get_market_data(self, coin_symbol: str, days: int = 30) -> pd.DataFrame:
        """시장 데이터 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = f"""
                    SELECT * FROM market_data 
                    WHERE coin_symbol = ? 
                    AND timestamp >= datetime('now', '-{days} days')
                    ORDER BY timestamp ASC
                """
                return pd.read_sql_query(query, conn, params=(coin_symbol,))
        except Exception as e:
            print(f"시장 데이터 조회 실패: {str(e)}")
            return pd.DataFrame()

    def save_analysis_result(self, analysis_data: Dict) -> bool:
        """분석 결과 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO analysis_results (
                        coin_symbol, analysis_type, result_data, timestamp
                    ) VALUES (?, ?, ?, ?)
                """, (
                    analysis_data['coin_symbol'],
                    analysis_data['analysis_type'],
                    json.dumps(analysis_data['result_data']),
                    analysis_data.get('timestamp', datetime.now().isoformat())
                ))
                return True
        except Exception as e:
            print(f"분석 결과 저장 실패: {str(e)}")
            return False

    def get_latest_analysis(self, coin_symbol: str, analysis_type: str) -> Optional[Dict]:
        """최신 분석 결과 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM analysis_results 
                    WHERE coin_symbol = ? AND analysis_type = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (coin_symbol, analysis_type))
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    result['result_data'] = json.loads(result['result_data'])
                    return result
                return None
        except Exception as e:
            print(f"분석 결과 조회 실패: {str(e)}")
            return None

    def update_keyword_weight(self, coin_symbol: str, keyword: str, 
                            category: str, weight_change: float) -> bool:
        """키워드 가중치 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE keywords 
                    SET weight = weight + ?,
                        last_used = CURRENT_TIMESTAMP
                    WHERE coin_symbol = ? AND keyword = ? AND category = ?
                """, (weight_change, coin_symbol, keyword, category))
                return True
        except Exception as e:
            print(f"키워드 가중치 업데이트 실패: {str(e)}")
            return False

    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """데이터베이스 백업"""
        if backup_path is None:
            backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

        try:
            with sqlite3.connect(self.db_path) as source:
                backup = sqlite3.connect(backup_path)
                source.backup(backup)
                backup.close()
                return True
        except Exception as e:
            print(f"데이터베이스 백업 실패: {str(e)}")
            return False

    def cleanup_old_data(self, days: int = 90) -> bool:
        """오래된 데이터 정리"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 오래된 뉴스 삭제
                cursor.execute("""
                    DELETE FROM news 
                    WHERE published_at < datetime('now', ?)
                """, (f'-{days} days',))
                
                # 오래된 시장 데이터 삭제
                cursor.execute("""
                    DELETE FROM market_data 
                    WHERE timestamp < datetime('now', ?)
                """, (f'-{days} days',))
                
                # 오래된 분석 결과 삭제
                cursor.execute("""
                    DELETE FROM analysis_results 
                    WHERE timestamp < datetime('now', ?)
                """, (f'-{days} days',))
                
                return True
        except Exception as e:
            print(f"데이터 정리 실패: {str(e)}")
            return False 