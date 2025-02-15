# AutoCoin - AI-Powered Cryptocurrency Trading Assistant 🚀

## Overview

AutoCoin은 실시간 시장 분석과 AI 기반 투자 전략을 제공하는 암호화폐 트레이딩 어시스턴트입니다. 기술적 분석, 호가 분석, 시장 심리 분석을 통합하여 투자자의 의사결정을 돕습니다.

## 주요 기능 🌟

### 1. 실시간 시장 분석
- 실시간 가격 및 거래량 모니터링
- 기술적 지표 분석 (RSI, 이동평균선, 볼린저 밴드)
- 호가창 분석 및 매수/매도 세력 측정
- 지지/저항 레버 자동 탐지

### 2. 투자 전략 추천
- 4가지 트레이딩 전략 제공:
  - 스캘핑 (초단타)
  - 데이트레이딩 (단타)
  - 스윙 트레이딩 (중기)
  - 포지션 트레이딩 (장기)
- 시장 상황별 맞춤 전략 추천
- 분할 매수/매도 전략
- 손절가 설정 가이드

### 3. 시장 심리 분석
- 공포탐욕지수 실시간 계산
- RSI 기반 과매수/과매도 분석
- 매수/매도 세력 균형 분석
- 변동성 모니터링

### 4. 차트 분석
- 캔들스틱 차트
- 이동평균선 (5일, 20일, 60일)
- 거래량 분석
- 추세 강도 측정

## 설치 방법 🔧

1. 저장소 클론:
```bash
git clone https://github.com/yourusername/autocoin.git
cd autocoin
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정 (.env 파일):
```env
UPBIT_ACCESS_KEY=your_upbit_access_key
UPBIT_SECRET_KEY=your_upbit_secret_key
```

## 사용 방법 📚

1. 프로그램 실행:
```bash
streamlit run app.py
```

2. 웹 인터페이스에서:
   - 코인 선택 (BTC, ETH, XRP 등)
   - 트레이딩 전략 선택
   - 업데이트 주기 설정
   - 분석 시작/중지

## 주요 화면 📊

### 시장 분석 화면
- 실시간 가격 정보
- 기술적 지표 현황
- 호가 분석 결과
- 차트 분석

### 투자 전략 화면
- 전략 추천 및 근거
- 매수/매도 가격 제안
- 투자 비중 추천
- 손절가 정보

## 프로젝트 구조 📁

```
autocoin/
├── app.py              # 메인 애플리케이션
├── market_analysis.py  # 시장 분석 모듈
├── fear_greed.py      # 공포탐욕지수 계산
├── investment_strategy.py  # 투자 전략 모듈
└── requirements.txt    # 프로젝트 의존성
```

## 의존성 📦

- Python 3.8+
- 주요 패키지:
  - streamlit
  - pandas
  - numpy
  - plotly
  - python-dotenv
  - requests

## 주의사항 ⚠️

- 이 프로그램은 투자 참고용으로만 사용하세요
- 모든 투자는 본인 책임하에 진행하세요
- 과도한 레버리지나 투기성 거래는 피하세요
- 항상 손절가를 준수하세요

## 기여하기 🤝

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 라이선스 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## 지원 💬

문제가 있으시면 GitHub 이슈를 생성해주세요.

---
Made with ❤️ by Needitem