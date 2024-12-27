import requests

def get_fear_greed_data():
    """
    Example function to fetch Fear & Greed Index.
    실사용 시엔 적절한 API URL, 파라미터 등을 수정하세요.
    """
    try:
        response = requests.get("https://api.alternative.me/fng/")
        data = response.json()
        # 예시: data["data"][0]["value"] 가 실제 지수 값
        index_value = data["data"][0].get("value")
        return int(index_value)
    except Exception as e:
        print("Failed to fetch Fear & Greed data:", e)
        return None 