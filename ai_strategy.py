import openai
import json
import os

# ... load your OPENAI_API_KEY ...
openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_market_with_ai(candles_data: list, fear_greed: float, news_list: list) -> str:
    prompt_text = f"""
    아래의 종합 정보 참고하여 시장을 분석하고 
    '매수', '매도', '보류' 중 하나를 한국어로 추천해주세요.

    1) 캔들 데이터:
    {candles_data}

    2) 공포 탐욕 지수: {fear_greed}

    3) 최신 뉴스:
    {news_list}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 투자 보조 AI입니다."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7,
            max_tokens=100
        )
        ai_response = response.choices[0].message.content.strip()
        return ai_response
    except Exception as e:
        print("AI 분석 에러:", e)
        return "보류"


def ai_self_reflection(current_strategy: str, fear_greed: float) -> str:
    """
    GPT 모델에게 현재 전략과 Fear & Greed Index를 기반으로 개선점을 물어보고, 
    새로운 전략 코드를 자동으로 생성하는 등 작업을 합니다.
    """
    # 간단 예시로 공포탐욕지수를 출력
    print(f"AI 회고: 현재 전략은 '{current_strategy}', 공포탐욕지수는 {fear_greed}입니다.")
    # 이후 실제 GPT 호출 등을 통해 전략을 수정할 수 있음
    # 예: if fear_greed < 30: "전략이 지나치게 공격적이니 조정이 필요합니다."
    # ...
    return f"{current_strategy} [with Fear & Greed consideration]"


def apply_custom_strategy(candles: list) -> str:
    """
    단순 '워뇨띠' 롱런 전략 예시
    """
    def calculate_indicator(c_list):
        if not c_list:
            return 0.0
        closes = [item["trade_price"] for item in c_list]
        return sum(closes) / len(closes)
    
    ma = calculate_indicator(candles)
    current_price = candles[0]["trade_price"] if candles else 0.0
    if current_price > ma:
        return "매수"
    else:
        return "보류" 


def analyze_price_target_with_ai(upbit_data: dict, binance_data: dict, fear_greed: float, news_list: list) -> str:
    """
    업비트/바이낸스 시세, 공포탐욕지수, 최신 뉴스 정보를 바탕으로
    구체적인 매수, 매도가를 제안하도록 OpenAI에게 질의하는 예시 함수.
    """
    # 환경변수에서 OPENAI_API_KEY 불러오기
    openai.api_key = os.getenv("OPENAI_API_KEY")

    upbit_price   = upbit_data.get("last_price")     # 업비트 현재가
    binance_price = binance_data.get("last_price")   # 바이낸스 현재가

    prompt_text = f"""
    현재 업비트 BTC 가격: {upbit_price} KRW
    현재 바이낸스 BTC 가격: {binance_price} USD (또는 환산 가정)
    공포 탐욕 지수(Fear & Greed Index): {fear_greed}
    최신 뉴스: {news_list}

    위 정보를 토대로, 한국어로:
    1) 지금 BTC를 얼마에 매수하면 좋을지
    2) 어느 구간에서 매도하면 좋을지
    3) 매수/매도 시점에 대한 의견
    을 간단히 제안해주세요.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 투자 보조 AI입니다."},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        # ChatCompletion과 동일하게 응답 파싱
        ai_response = response.choices[0].message["content"].strip()
        return ai_response

    except Exception as e:
        print("OpenAI API 호출 에러:", e)
        return "AI 응답 실패. 보류."


# 예시 사용 방법
if __name__ == "__main__":
    # 아래 예시 데이터들은 실제로는 services.py나 trading.py 등에서 불러최다.
    dummy_upbit_data = {
        "last_price": 37000000,
        "volume_base": 1234.5,
    }
    dummy_binance_data = {
        "last_price": 28000,
        "volume_base": 9876.5,
    }
    dummy_fear_greed = 40
    dummy_news = ["Bitcoin surges above $28k again", "Market uncertainty remains high"]

    result = analyze_price_target_with_ai(
        upbit_data=dummy_upbit_data,
        binance_data=dummy_binance_data,
        fear_greed=dummy_fear_greed,
        news_list=dummy_news
    )
    print("AI 판단 결과:\n", result) 