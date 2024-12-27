import requests
from dotenv import load_dotenv
import os

load_dotenv()

def get_latest_news():
    """
    Fetch the latest 10 Bitcoin-related news headlines.
    실사용 시엔 유효한 NewsAPI 키를 사용하고,
    필요에 따라 파라미터나 인증 토큰 등을 조정하세요.
    """
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "bitcoin",               # 검색 키워드
            "pageSize": 10,              # 원하는 기사 개수
            "apiKey": os.getenv("NEWS_API_KEY")  # NewsAPI 키
        }
        response = requests.get(url, params=params)
        data = response.json()
        # "articles" 속성에서 상위 10개 기사 타이틀만 추출
        headlines = [
            f"{article['source']['name']}: {article['title']}"
            for article in data.get("articles", [])[:10]
        ]
        return headlines
    except Exception as e:
        print("Failed to fetch latest news:", e)
        return [] 