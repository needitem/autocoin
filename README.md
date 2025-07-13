# AutoCoin Trading Bot

가상화폐 자동매매 봇 프로젝트입니다. 업비트(Upbit)와 빗썸(Bithumb) 거래소를 모두 지원합니다.

## 기능

- 멀티 거래소 지원 (Upbit, Bithumb)
- 실시간 시장 분석
- 기술적 분석 기반 매매 신호 생성
- 가상 매매 시뮬레이션
- 웹 기반 대시보드 (Streamlit)
- 거래소 실시간 전환 기능

## 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
cd autocoin
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 실행 방법

### 1. 기본 기능 테스트 (Streamlit 없이)
```bash
python run.py
```

### 2. Streamlit UI 실행
```bash
streamlit run app.py
```

또는

```bash
python -m streamlit run app.py
```

## 환경 설정

1. `.env` 파일 생성 (`.env.example` 참고)
2. 필요한 API 키 설정:

### Upbit API
```
UPBIT_ACCESS_KEY=your_upbit_access_key
UPBIT_SECRET_KEY=your_upbit_secret_key
```

### Bithumb API
```
BITHUMB_API_KEY=your_bithumb_api_key
BITHUMB_SECRET_KEY=your_bithumb_secret_key
```

### 기타 API (선택사항)
```
OPENAI_API_KEY=your_openai_key
NEWS_API_KEY=your_news_key
GEMINI_API_KEY=your_gemini_key
```

## 주요 기능

### 시장 분석
- 기술적 분석 (RSI, MACD, 볼린저 밴드)
- 패턴 인식
- 거래량 분석
- 시장 심리 분석

### 매매 전략
- RSI 기반 매매
- 이동평균선 전략
- 볼린저 밴드 전략
- MACD 전략

### 위험 관리
- 포지션 크기 조절
- 손절/익절 설정
- 변동성 기반 리스크 관리

## 프로젝트 구조

```
autocoin/
├── src/
│   ├── api/            # 거래소 API
│   │   ├── base.py     # 거래소 API 추상 클래스
│   │   ├── upbit.py    # 업비트 API 구현
│   │   └── bithumb.py  # 빗썸 API 구현
│   ├── core/           # 핵심 비즈니스 로직
│   │   ├── trading.py  # 거래 관리자
│   │   └── strategy.py # 매매 전략
│   ├── db/             # 데이터베이스
│   │   └── database.py # SQLite 데이터베이스 매니저
│   ├── ui/             # 사용자 인터페이스
│   │   └── app.py      # Streamlit UI
│   └── utils/          # 유틸리티 함수
│       └── logger.py   # 로깅 설정
├── tests/              # 테스트 코드
├── app.py             # 메인 실행 파일
├── run.py             # CLI 테스트 실행 파일
└── requirements.txt   # 의존성 목록
```

## 라이선스

MIT License