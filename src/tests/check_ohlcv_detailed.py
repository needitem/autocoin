import pyupbit
import pandas as pd
import time
from datetime import datetime, timedelta
pd.set_option('display.max_columns', None)

def check_ohlcv_detailed():
    market = "KRW-BTC"
    
    print("\n=== 현재 시간 ===")
    print(datetime.now())
    
    print("\n=== 일봉 데이터 조회 (기본) ===")
    day_candle = pyupbit.get_ohlcv(market, interval="day", count=1)
    print(day_candle)
    
    time.sleep(1)  # Rate limit 준수
    
    print("\n=== 일봉 데이터 조회 (to_KST=False) ===")
    try:
        day_candle = pyupbit.get_ohlcv(market, interval="day", count=1, to_KST=False)
        print(day_candle)
    except Exception as e:
        print(f"Error: {str(e)}")
    
    time.sleep(1)  # Rate limit 준수
    
    print("\n=== 일봉 데이터 조회 (count=2) ===")
    day_candle = pyupbit.get_ohlcv(market, interval="day", count=2)
    print(day_candle)
    
    if day_candle is not None and not day_candle.empty:
        print("\n=== 데이터 상세 정보 ===")
        print(f"Index type: {type(day_candle.index)}")
        print(f"Timezone info: {day_candle.index.tz}")
        print("\nColumns info:")
        for col in day_candle.columns:
            print(f"{col}: {day_candle[col].dtype}")

if __name__ == "__main__":
    check_ohlcv_detailed() 