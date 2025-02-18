import pyupbit
import json

def check_orderbook_format():
    # KRW-BTC 마켓의 호가 데이터 조회
    orderbook = pyupbit.get_orderbook("KRW-BTC")
    
    # 데이터 구조 출력
    print("\n=== 전체 데이터 구조 ===")
    print(json.dumps(orderbook, indent=2, ensure_ascii=False))
    
    if orderbook and len(orderbook) > 0:
        first_item = orderbook[0]
        print("\n=== 첫 번째 항목의 키 ===")
        print(list(first_item.keys()))
        
        if 'orderbook_units' in first_item:
            print("\n=== orderbook_units의 첫 번째 항목 ===")
            print(json.dumps(first_item['orderbook_units'][0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    check_orderbook_format() 