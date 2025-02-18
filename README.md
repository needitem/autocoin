# AutoCoin Trading Bot

가상화폐 자동매매 봇 프로젝트입니다.

## 기능

- 실시간 시장 분석
- 기술적 분석 기반 매매 신호 생성
- 가상 매매 시뮬레이션
- 웹 기반 대시보드 (Streamlit)

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

```bash
streamlit run src/core/app.py
```

## 환경 설정

1. `.env` 파일 생성
2. 필요한 API 키 설정:
```
UPBIT_ACCESS_KEY=your_access_key
UPBIT_SECRET_KEY=your_secret_key
```

## 주요 기능

- 시장 분석
  - 기술적 분석
  - 패턴 인식
  - 거래량 분석
  - 시장 심리 분석

- 매매 전략
  - RSI 기반 매매
  - 이동평균선 전략
  - 볼린저 밴드 전략
  - MACD 전략

- 위험 관리
  - 포지션 크기 조절
  - 손절/익절 설정
  - 변동성 기반 리스크 관리

## 프로젝트 구조

```
autocoin/
├── src/
│   ├── analysis/        # 시장 분석 모듈
│   ├── core/           # 핵심 애플리케이션 로직
│   ├── data/           # 데이터 처리
│   ├── exchange/       # 거래소 API
│   ├── news/           # 뉴스 분석
│   ├── strategies/     # 매매 전략
│   ├── trading/        # 매매 실행
│   ├── utils/          # 유틸리티 함수
│   └── visualization/  # 차트 시각화
├── tests/              # 테스트 코드
├── docs/              # 문서
└── requirements.txt   # 의존성 목록
```

## 라이선스

MIT License