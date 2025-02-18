import pyupbit
import pandas as pd
pd.set_option('display.max_columns', None)

def check_ohlcv_format():
    # KRW-BTC 마켓의 일봉 데이터 조회
    print("\n=== 일봉 데이터 조회 ===")
    day_candle = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=1)
    
    if day_candle is not None and not day_candle.empty:
        print("\n=== 데이터 형식 ===")
        print(type(day_candle))
        print("\n=== 컬럼 목록 ===")
        print(day_candle.columns.tolist())
        print("\n=== 데이터 내용 ===")
        print(day_candle)
    else:
        print("일봉 데이터 조회 실패")

if __name__ == "__main__":
    check_ohlcv_format() 