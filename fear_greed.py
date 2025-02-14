import numpy as np
from typing import List, Dict, Optional
import math
import requests
from datetime import datetime, timedelta
import time

class FearGreedIndex:
    def __init__(self):
        # Constants
        self.LAMBDA = 0.94  # λ for volatility calculation
        self.LAMBDA_SHORT = 1 - 1/7  # λ for short-term momentum
        self.LAMBDA_LONG = 1 - 1/30  # λ for long-term momentum
        self.LAMBDA_2 = 1 - 1/2  # λ for 2-day EMA
        self.LAMBDA_7 = 1 - 1/7  # λ for 7-day EMA
        self.C = 16.387308  # Constant for momentum score calculation
        
    def calculate_ema(self, values: List[float], lambda_param: float) -> float:
        """
        Calculate EMA using the exact formula (Eq. 2-2)
        Uλ,i = Σ(j=0 to i) (1-λ)λʲ⁻ⁱuⱼ
        """
        n = len(values)
        result = 0
        sum_weights = 0
        
        for j in range(n):
            weight = (1 - lambda_param) * (lambda_param ** j)
            result += weight * values[-(j+1)]  # 최신 데이터부터 시작
            sum_weights += weight
            
        return result / sum_weights if sum_weights != 0 else 0
    
    def calculate_volatility(self, returns: List[float]) -> List[float]:
        """
        Calculate volatility using EWMA (Eq. 1-1)
        σᵢ² = λ(σᵢ₋₁²) + (1-λ)(xᵢ)²
        """
        volatilities = []
        volatility_squared = returns[0] ** 2
        
        for ret in returns:
            volatility_squared = self.LAMBDA * volatility_squared + (1 - self.LAMBDA) * ret * ret
            volatilities.append(math.sqrt(volatility_squared))
            
        return volatilities
    
    def calculate_momentum_scores(self, prices: List[float], s1: float, days: int = 7) -> List[float]:
        """
        Calculate momentum scores for the past n days
        """
        scores = []
        for i in range(days):
            if i == 0:
                temp_prices = prices
            else:
                temp_prices = prices[:-i]
            
            # Calculate weights (Eq. 2-1)
            alpha = (9 * s1) + 1
            l_short = alpha
            l_long = 10 - alpha
            
            # Calculate EMAs (Eq. 2-2)
            current_price = temp_prices[-1]
            u_short = self.calculate_ema(temp_prices, self.LAMBDA_SHORT)
            u_long = self.calculate_ema(temp_prices, self.LAMBDA_LONG)
            
            # Calculate deviation ratios
            x_short = (current_price - u_short) / u_short
            x_long = (current_price - u_long) / u_long
            
            # Calculate momentum score (Eq. 2-3)
            score = self.C * (l_long * x_long + l_short * x_short) / 10
            scores.append(score)
            
        return scores
    
    def calculate_momentum_adjustment(self, momentum_scores: List[float]) -> float:
        """
        Calculate momentum adjustment (Eq. 3)
        """
        # Calculate short and long-term weighted averages
        w_short = self.calculate_ema(momentum_scores, self.LAMBDA_2)
        w_long = self.calculate_ema(momentum_scores, self.LAMBDA_7)
        
        # Calculate W
        w = (w_short + w_long) / 2
        abs_w = abs(w)
        
        # Calculate beta
        beta = 2 + abs_w - 4 / (1 + math.exp(-abs_w))
        return beta if w >= 0 else -beta
    
    def calculate(self, prices: List[float], volumes: List[float]) -> Dict[str, float]:
        """
        Calculate complete Fear and Greed Index
        """
        # Calculate returns
        returns = [(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]
        
        # Calculate volatility score
        volatilities = self.calculate_volatility(returns)
        log_volatilities = np.log(volatilities[-365:])
        mu = np.mean(log_volatilities)
        sigma = np.std(log_volatilities)
        current_volatility = volatilities[-1]
        volatility_score = np.clip((np.log(current_volatility) - mu) / sigma, -4, 4)
        
        # Calculate volume score
        current_volume = volumes[-1]
        lambda_20 = 1 - 1/20
        lambda_60 = 1 - 1/60
        v20 = self.calculate_ema(volumes[-20:], lambda_20)
        v60 = self.calculate_ema(volumes[-60:], lambda_60)
        v20_ratio = np.log(current_volume / v20)
        v60_ratio = np.log(current_volume / v60)
        volume_score = np.clip((v20_ratio + v60_ratio) / 2, -4, 4)
        
        # Calculate S1 (Eq. 1-5)
        s1 = (np.clip(volume_score + volatility_score, -4, 4) / 8) + 0.5
        
        # Calculate momentum scores and adjustment
        momentum_scores = self.calculate_momentum_scores(prices, s1)
        s2 = momentum_scores[0]
        beta = self.calculate_momentum_adjustment(momentum_scores)
        s2_adjusted = s2 - beta
        
        # Calculate final index (Eq. 3)
        fear_greed_index = 100 / (1 + math.exp(-(s1 * s2_adjusted)))
        
        return {
            'volatility_score': volatility_score,
            'volume_score': volume_score,
            's1_score': s1,
            's2_score': s2,
            's2_adjusted': s2_adjusted,
            'beta': beta,
            'fear_greed_index': fear_greed_index / 100
        }

def get_candles(symbol: str = "KRW-BTC", count: int = 365) -> Optional[List[Dict]]:
    """
    업비트 API에서 최근 캔들 데이터를 가져옵니다.
    
    Args:
        symbol: 마켓 코드 (예: KRW-BTC, KRW-ETH, BTC-ETH 등)
        count: 가져올 데이터 수 (기본값: 365일)
    
    Returns:
        List[Dict]: 캔들 데이터 리스트
    """
    try:
        all_data = []
        remaining = count
        last_date = None
        
        while remaining > 0:
            url = f"https://api.upbit.com/v1/candles/days"
            params = {
                "market": symbol,
                "count": min(200, remaining)
            }
            
            if last_date:
                params["to"] = last_date.strftime("%Y-%m-%d %H:%M:%S")
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"캔들 데이터를 가져오는데 실패했습니다. 상태 코드: {response.status_code}")
                return None
                
            data = response.json()
            if not data:
                break
                
            all_data.extend(data)
            
            last_date = datetime.strptime(data[-1]['candle_date_time_kst'], "%Y-%m-%dT%H:%M:%S")
            remaining -= len(data)
            
            time.sleep(0.1)  # API 호출 간격 조절
        
        if len(all_data) < count:
            print(f"충분한 데이터가 없습니다. 필요: {count}일, 현재: {len(all_data)}일")
            return None
            
        return all_data
        
    except Exception as e:
        print(f"데이터 가져오기 실패: {str(e)}")
        return None

def print_fear_greed_analysis(data: Dict[str, float], symbol: str):
    """
    공포탐욕지수 분석 결과를 출력합니다.
    
    Args:
        data: 공포탐욕지수 계산 결과
        symbol: 분석 대상 심볼
    """
    if not data:
        print("\n데이터를 가져올 수 없습니다.")
        return
        
    print(f"\n=== {symbol} 공포탐욕지수 분석 ===")
    print(f"날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"지수: {data['fear_greed_index'] * 100:.2f}")
    
    # 시장 상태 판단
    index = data['fear_greed_index'] * 100
    if index >= 80:
        state = "극단적 탐욕"
    elif index >= 65:
        state = "탐욕"
    elif index >= 45:
        state = "중립"
    elif index >= 30:
        state = "공포"
    else:
        state = "극단적 공포"
    
    print(f"상태: {state}")
    
    print("\n[세부 지표]")
    print(f"변동성 점수: {data['volatility_score']:.2f}")
    print(f"거래량 점수: {data['volume_score']:.2f}")
    print(f"모멘텀 점수: {data['s2_score']:.2f}")
    print(f"모멘텀 보정 점수: {data['s2_adjusted']:.2f}")
    print(f"종합 점수: {data['s1_score']:.2f}")
    
    print("\n[투자 제안]")
    if index <= 20:
        print("💡 매수 기회일 수 있습니다. 단, 추가 하락에 대비가 필요합니다.")
    elif index <= 35:
        print("💡 조심스러운 매수를 고려해볼 수 있습니다.")
    elif index <= 55:
        print("💡 추세를 지켜보며 신중한 접근이 필요합니다.")
    elif index <= 75:
        print("💡 위험 관리에 신경 쓰고, 일부 이익실현을 고려해볼 수 있습니다.")
    else:
        print("💡 이익실현을 고려하고, 신규 진입은 신중해야 합니다.")
    
    print("\n⚠️ 주의: 이 지표는 참고용이며, 반드시 다른 지표들과 함께 종합적으로 판단해야 합니다.")

def calculate_returns_and_volatility(candles):
    """
    캔들 데이터로부터 가격과 거래량 데이터를 추출합니다.
    데이터는 과거->현재 순서로 정렬됩니다.
    """
    # 데이터를 날짜 순서대로 정렬 (과거->현재)
    sorted_candles = sorted(candles, key=lambda x: x['candle_date_time_kst'])
    
    prices = [float(candle['trade_price']) for candle in sorted_candles]
    volumes = [float(candle['candle_acc_trade_price']) for candle in sorted_candles]
    return prices, volumes

def get_available_markets() -> List[str]:
    """
    업비트에서 사용 가능한 마켓 목록을 가져옵니다.
    """
    try:
        url = "https://api.upbit.com/v1/market/all"
        response = requests.get(url)
        
        if response.status_code == 200:
            markets = response.json()
            return [market['market'] for market in markets]
        else:
            print(f"마켓 목록을 가져오는데 실패했습니다. 상태 코드: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"마켓 목록 조회 실패: {str(e)}")
        return []

def calculate_fear_greed_for_symbol(symbol: str) -> None:
    """
    지정된 심볼에 대한 공포탐욕지수를 계산하고 출력합니다.
    """
    try:
        # 데이터 가져오기
        print(f"{symbol} 데이터를 가져오는 중...")
        candles = get_candles(symbol, count=365)
        
        if not candles:
            print(f"{symbol} 데이터를 가져오는데 실패했습니다.")
            return
            
        # 데이터 전처리
        prices, volumes = calculate_returns_and_volatility(candles)
        
        # Fear & Greed Index 계산
        fg_index = FearGreedIndex()
        result = fg_index.calculate(prices, volumes)
        
        # 결과 출력
        print_fear_greed_analysis(result, symbol)
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        print(f"상세 에러: {traceback.format_exc()}")

def search_market(query: str, markets: List[str]) -> List[str]:
    """
    주어진 검색어로 마켓을 검색합니다.
    
    Args:
        query: 검색어
        markets: 마켓 목록
    
    Returns:
        검색된 마켓 목록
    """
    query = query.upper()
    return [market for market in markets if query in market]

if __name__ == "__main__":
    # 사용 가능한 마켓 목록 조회
    markets = get_available_markets()
    if not markets:
        print("마켓 목록을 가져올 수 없습니다.")
        exit(1)
    
    # KRW 마켓만 필터링
    krw_markets = [market for market in markets if market.startswith('KRW-')]
    
    while True:
        print("\n=== 공포탐욕지수 계산기 ===")
        print("1. 마켓 목록 보기")
        print("2. 심볼 검색")
        print("q. 종료")
        
        choice = input("\n선택하세요: ")
        
        if choice.lower() == 'q':
            break
            
        elif choice == '1':
            print("\n=== 사용 가능한 KRW 마켓 목록 ===")
            for i, market in enumerate(krw_markets, 1):
                print(f"{i}. {market}")
                
            while True:
                try:
                    idx_choice = input("\n분석할 마켓 번호를 선택하세요 (뒤로가기: b): ")
                    if idx_choice.lower() == 'b':
                        break
                        
                    idx = int(idx_choice) - 1
                    if 0 <= idx < len(krw_markets):
                        selected_market = krw_markets[idx]
                        calculate_fear_greed_for_symbol(selected_market)
                    else:
                        print("올바른 번호를 입력하세요.")
                except ValueError:
                    print("올바른 번호를 입력하세요.")
                    
        elif choice == '2':
            while True:
                search_query = input("\n검색어를 입력하세요 (뒤로가기: b): ")
                if search_query.lower() == 'b':
                    break
                    
                search_results = search_market(search_query, krw_markets)
                
                if not search_results:
                    print("검색 결과가 없습니다.")
                    continue
                    
                print("\n=== 검색 결과 ===")
                for i, market in enumerate(search_results, 1):
                    print(f"{i}. {market}")
                    
                while True:
                    try:
                        idx_choice = input("\n분석할 마켓 번호를 선택하세요 (뒤로가기: b): ")
                        if idx_choice.lower() == 'b':
                            break
                            
                        idx = int(idx_choice) - 1
                        if 0 <= idx < len(search_results):
                            selected_market = search_results[idx]
                            calculate_fear_greed_for_symbol(selected_market)
                        else:
                            print("올바른 번호를 입력하세요.")
                    except ValueError:
                        print("올바른 번호를 입력하세요.")
