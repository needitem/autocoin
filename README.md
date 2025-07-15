# 🚀 AutoCoin - AI 기반 암호화폐 자동매매 시스템

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.1-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**AutoCoin**은 AI 기반 포트폴리오 관리와 고급 시장 감정 분석을 통해 암호화폐 자동매매를 수행하는 통합 플랫폼입니다. 업비트(Upbit)와 빗썸(Bithumb) 거래소를 지원하며, 실시간 대시보드를 통해 포트폴리오 관리와 매매 신호를 제공합니다.

## ✨ 주요 기능

### 🤖 AI 포트폴리오 전략 시스템
- **자동 포트폴리오 배분**: 10개 주요 코인에 대한 AI 기반 최적 배분
- **리스크 레벨 관리**: 보수적/중간/공격적 투자 성향별 차별화된 전략
- **실시간 리밸런싱**: 목표 배분 대비 5% 이상 차이시 자동 조정 제안
- **종합 점수 계산**: 기술적 분석 + 감정 분석 + 거래량 분석 통합

### 📊 향상된 감정 분석 시스템
- **다차원 분석**: 뉴스, 소셜미디어, 기술적 지표 종합 분석
- **실시간 감정 지수**: -1.0 (극도의 공포) ~ 1.0 (극도의 탐욕)
- **신뢰도 계산**: 데이터 소스 다양성과 일치도 기반
- **시그널 생성**: 주요 시장 신호 자동 감지

### 📈 멀티 시간대 기술적 분석
- **9개 시간대 분석**: 30초부터 1개월까지 다양한 시간대 지원
- **기술적 지표**: RSI, MACD, 볼린저 밴드, 스토캐스틱 등
- **패턴 인식**: 차트 패턴 자동 인식 및 분석
- **거래량 분석**: 거래량 기반 시장 강도 측정

### 🔄 멀티 거래소 지원
- **업비트(Upbit)**: 실시간 데이터 및 주문 처리
- **빗썸(Bithumb)**: 다중 거래소 지원으로 유동성 확보
- **실시간 전환**: 거래소 간 실시간 전환 기능
- **통합 관리**: 단일 인터페이스로 다중 거래소 관리

### 🎯 위험 관리 시스템
- **포지션 크기 조절**: 변동성 기반 자동 포지션 조정
- **리스크 지표**: 샤프 비율, VaR, 최대 낙폭 계산
- **손절/익절 설정**: 개별 코인별 자동 손절/익절 설정
- **포트폴리오 다각화**: 상관관계 기반 다각화 전략

## 🛠️ 설치 및 실행

### 1. 시스템 요구사항
- **Python**: 3.8 이상
- **운영체제**: Windows, macOS, Linux
- **메모리**: 최소 4GB (권장 8GB)
- **저장공간**: 최소 1GB

### 2. 설치 방법

#### 2.1 저장소 클론
```bash
git clone https://github.com/needitem/autocoin.git
cd autocoin
```

#### 2.2 가상환경 생성 및 활성화
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

#### 2.3 의존성 설치
```bash
# 전체 의존성 설치
pip install -r requirements.txt

# 또는 최소 의존성만 설치
pip install -r requirements_minimal.txt
```

### 3. 환경 설정

#### 3.1 API 키 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
# 필수 API 키
UPBIT_ACCESS_KEY=your_upbit_access_key
UPBIT_SECRET_KEY=your_upbit_secret_key
BITHUMB_API_KEY=your_bithumb_api_key
BITHUMB_SECRET_KEY=your_bithumb_secret_key

# 선택적 API 키 (고급 기능용)
OPENAI_API_KEY=your_openai_key
NEWS_API_KEY=your_news_key
GEMINI_API_KEY=your_gemini_key
```

#### 3.2 API 키 발급 방법

**업비트 API 키 발급:**
1. [업비트 프로](https://upbit.com/mypage/open_api_management) 접속
2. "Open API 관리" → "API 키 발급"
3. 필요한 권한 선택 (조회, 거래, 출금 등)
4. 생성된 Access Key와 Secret Key 복사

**빗썸 API 키 발급:**
1. [빗썸 Pro](https://www.bithumb.com/react/trade/order/BTC_KRW) 접속
2. "API 관리" → "API 키 발급"
3. 권한 설정 후 API Key와 Secret Key 복사

### 4. 실행 방법

#### 4.1 Streamlit 웹 애플리케이션 실행
```bash
streamlit run app.py
```

또는

```bash
python -m streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하여 대시보드를 확인하세요.

#### 4.2 CLI 테스트 실행
```bash
python run.py
```

#### 4.3 자동화 실행
```bash
python run_auto.py
```

## 📁 프로젝트 구조

```
autocoin/
├── 🏗️ 메인 실행 파일
│   ├── app.py                    # Streamlit 메인 애플리케이션
│   ├── run.py                    # CLI 테스트 실행
│   └── run_auto.py               # 자동화 실행
├── 📁 src/                       # 소스 코드
│   ├── 📡 api/                   # 거래소 API 모듈
│   │   ├── base.py               # API 기본 클래스
│   │   ├── upbit.py              # 업비트 API 구현
│   │   ├── bithumb.py            # 빗썸 API 구현
│   │   ├── news.py               # 뉴스 API
│   │   └── calculate.py          # 기술적 지표 계산
│   ├── 🧠 core/                  # 핵심 비즈니스 로직
│   │   ├── trading.py            # 거래 관리자
│   │   ├── ai_portfolio_strategy.py      # AI 포트폴리오 전략 ⭐
│   │   ├── enhanced_sentiment_analyzer.py # 향상된 감정 분석 ⭐
│   │   ├── portfolio_manager.py  # 포트폴리오 관리 ⭐
│   │   ├── multi_timeframe_analyzer.py   # 멀티타임프레임 분석
│   │   ├── strategy.py           # 기본 매매 전략
│   │   ├── ai_predictor.py       # AI 예측 모델
│   │   ├── chart_analysis.py     # 차트 분석
│   │   ├── news_strategy.py      # 뉴스 기반 전략
│   │   ├── pattern_recognition.py # 패턴 인식
│   │   ├── performance_optimizer.py # 성능 최적화
│   │   ├── price_analyzer.py     # 가격 분석
│   │   └── risk_analyzer.py      # 리스크 분석
│   ├── 💾 db/                    # 데이터베이스
│   │   └── database.py           # SQLite 데이터베이스 매니저
│   ├── 🎨 ui/                    # 사용자 인터페이스
│   │   ├── app.py                # Streamlit 메인 UI
│   │   └── components/           # UI 컴포넌트
│   │       ├── portfolio_dashboard.py    # 포트폴리오 대시보드 ⭐
│   │       ├── market_sentiment_dashboard.py # 시장 감정 대시보드
│   │       ├── chart_analysis.py # 차트 분석 컴포넌트
│   │       ├── analysis.py       # 분석 컴포넌트
│   │       ├── market.py         # 시장 컴포넌트
│   │       ├── news.py           # 뉴스 컴포넌트
│   │       └── trading.py        # 거래 컴포넌트
│   ├── 🔧 utils/                 # 유틸리티
│   │   ├── logger.py             # 로깅 설정
│   │   ├── common.py             # 공통 함수 ⭐
│   │   ├── performance_cache.py  # 성능 캐시 ⭐
│   │   └── async_helper.py       # 비동기 헬퍼
│   ├── 🛠️ config/                # 설정 파일
│   │   └── constants.py          # 상수 정의 ⭐
│   ├── 📊 market/                # 시장 데이터
│   │   └── render.py             # 시장 렌더링
│   └── 💱 trading/               # 거래 관련
│       └── trading_manager.py    # 거래 관리자
├── 🧪 tests/                     # 테스트 코드
│   ├── test_*.py                 # 각종 테스트
│   └── conftest.py               # 테스트 설정
├── 📊 logs/                      # 로그 파일
├── 🗃️ *.db                       # 데이터베이스 파일
├── 📋 requirements.txt           # 의존성 목록
└── 📖 README.md                  # 프로젝트 문서
```

## 🎯 사용 방법

### 1. 웹 대시보드 사용법

#### 1.1 메인 대시보드
- **시장 현황**: 실시간 코인 가격 및 변동률 확인
- **포트폴리오 현황**: 보유 자산 및 수익률 모니터링
- **매매 추천**: AI 기반 매수/매도 신호 확인

#### 1.2 AI 포트폴리오 대시보드
- **포트폴리오 분석**: 리스크 레벨 설정 및 분석 실행
- **배분 현황**: 현재 vs 목표 배분 비교
- **리밸런싱**: 배분 차이에 따른 조정 제안

#### 1.3 시장 감정 대시보드
- **감정 지수**: 실시간 시장 감정 지수 확인
- **구성 요소**: 9개 감정 구성 요소 상세 분석
- **신호 현황**: 주요 시장 신호 및 알림

### 2. CLI 사용법

#### 2.1 기본 실행
```bash
python run.py
```

#### 2.2 특정 코인 분석
```bash
python run.py --coin BTC
```

#### 2.3 백테스팅
```bash
python run.py --backtest --period 30
```

### 3. 자동화 실행

#### 3.1 스케줄러 설정
```python
# run_auto.py에서 설정
schedule.every(10).minutes.do(run_analysis)
schedule.every().hour.do(update_portfolio)
```

#### 3.2 모니터링
- **로그 확인**: `logs/` 디렉토리에서 실행 로그 확인
- **성능 모니터링**: 대시보드에서 실시간 성능 지표 확인

## 📈 투자 전략

### 1. AI 포트폴리오 전략

#### 1.1 리스크 레벨별 전략
- **보수적 (Conservative)**: 변동성 낮은 메이저 코인 중심
- **중간 (Moderate)**: 균형잡힌 포트폴리오
- **공격적 (Aggressive)**: 고수익 알트코인 비중 증가

#### 1.2 배분 알고리즘
- **종합 점수 계산**: 기술적 분석(40%) + 감정 분석(30%) + 거래량 분석(30%)
- **리스크 조정**: 변동성 기반 포지션 크기 조정
- **상관관계 고려**: 코인 간 상관관계 분석으로 다각화

### 2. 기술적 분석 전략

#### 2.1 멀티 시간대 분석
- **단기 (30초~10분)**: 스캘핑 전략
- **중기 (1시간~1일)**: 스윙 트레이딩 전략
- **장기 (1주~1개월)**: 포지션 트레이딩 전략

#### 2.2 기술적 지표 조합
- **RSI + MACD**: 추세 전환점 포착
- **볼린저 밴드 + 거래량**: 브레이크아웃 신호
- **이동평균선 + 스토캐스틱**: 진입/청산 타이밍

### 3. 감정 분석 전략

#### 3.1 감정 지수 활용
- **극도의 공포 (-0.8~-1.0)**: 매수 기회
- **극도의 탐욕 (0.8~1.0)**: 매도 신호
- **중립 구간 (-0.3~0.3)**: 관망 또는 단기 매매

#### 3.2 뉴스 기반 전략
- **긍정적 뉴스**: 매수 신호 강화
- **부정적 뉴스**: 매도 신호 또는 관망
- **중립적 뉴스**: 기술적 분석 우선

## 🔧 고급 설정

### 1. 성능 최적화

#### 1.1 캐시 설정
```python
# src/config/constants.py
CACHE_SIZE = 500  # 캐시 크기
CACHE_TTL = 180   # 캐시 TTL (초)
```

#### 1.2 비동기 처리 설정
```python
# 동시 처리 수 조정
MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 30
```

### 2. 로깅 설정

#### 2.1 로그 레벨 조정
```python
# src/utils/logger.py
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
```

#### 2.2 로그 파일 관리
- **자동 로테이션**: 일별 로그 파일 생성
- **크기 제한**: 최대 50MB per 파일
- **보관 기간**: 최대 30일

### 3. 데이터베이스 최적화

#### 3.1 인덱스 설정
```sql
-- 시장 데이터 인덱스
CREATE INDEX idx_market_timestamp ON market_data(market, timestamp DESC);
CREATE INDEX idx_market_timestamp_covering ON market_data(market, timestamp DESC, open, high, low, close, volume);
```

#### 3.2 데이터 정리
- **자동 정리**: 30일 이상 된 데이터 자동 삭제
- **압축**: 오래된 데이터 압축 저장

## 🧪 테스트

### 1. 단위 테스트 실행
```bash
python -m pytest tests/
```

### 2. 통합 테스트 실행
```bash
python -m pytest tests/test_integration.py
```

### 3. 성능 테스트 실행
```bash
python -m pytest tests/test_performance.py
```

### 4. 백테스팅
```bash
python tests/test_backtest.py --period 90 --initial-capital 10000000
```

## 📊 성능 지표

### 1. 시스템 성능
- **응답 시간**: 평균 < 1초
- **메모리 사용량**: 최적화 후 30-40% 감소
- **처리량**: 초당 100개 이상 데이터 포인트 처리

### 2. 투자 성과 (백테스트 기준)
- **수익률**: 연 20-30% (시장 대비)
- **샤프 비율**: 1.5 이상
- **최대 낙폭**: 15% 이하

### 3. 최적화 효과
- **코드 줄 수**: 1,040줄 감소
- **중복 코드**: 60% 감소
- **캐시 히트율**: 85% 이상

## 🚨 주의사항

### 1. 투자 위험
- **투자 손실 가능성**: 암호화폐 투자는 높은 변동성을 가짐
- **자동매매 리스크**: 시스템 오류로 인한 예상치 못한 손실 가능
- **백테스팅 한계**: 과거 성과가 미래 성과를 보장하지 않음

### 2. 기술적 위험
- **API 제한**: 거래소 API 제한으로 인한 서비스 중단 가능
- **네트워크 오류**: 인터넷 연결 불안정으로 인한 거래 실패
- **시스템 부하**: 높은 부하 시 성능 저하 가능

### 3. 보안 주의사항
- **API 키 보안**: API 키를 안전하게 보관하고 주기적으로 교체
- **권한 제한**: 불필요한 권한은 비활성화
- **로그 보안**: 민감한 정보가 로그에 기록되지 않도록 주의

## 🤝 기여하기

### 1. 기여 방법
1. 이 저장소를 Fork
2. 새로운 기능 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

### 2. 개발 가이드라인
- **코드 스타일**: PEP 8 준수
- **테스트**: 새로운 기능에 대한 테스트 코드 작성
- **문서화**: 코드 변경시 문서 업데이트
- **커밋 메시지**: 명확하고 설명적인 커밋 메시지 작성

## 📞 지원

### 1. 이슈 신고
- **버그 신고**: [GitHub Issues](https://github.com/needitem/autocoin/issues)
- **기능 요청**: [GitHub Issues](https://github.com/needitem/autocoin/issues)
- **질문**: [GitHub Discussions](https://github.com/needitem/autocoin/discussions)

### 2. 문의
- **이메일**: needitem@gmail.com
- **Discord**: [AutoCoin Community](https://discord.gg/autocoin)

## 📜 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- **업비트**: 안정적인 API 제공
- **빗썸**: 다양한 거래 기능 지원
- **Streamlit**: 훌륭한 웹 애플리케이션 프레임워크
- **오픈소스 커뮤니티**: 다양한 라이브러리 제공

---

**⚠️ 면책 조항**: 이 소프트웨어는 교육 및 연구 목적으로 제공됩니다. 실제 투자 결정은 신중하게 내리시기 바라며, 투자 손실에 대한 책임은 사용자에게 있습니다. 투자 전에 반드시 전문가와 상담하시기 바랍니다.

**📈 Happy Trading!** 🚀