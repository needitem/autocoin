"""
Strategy Analysis Component

This component displays trading strategy analysis and recommendations.
"""

import streamlit as st
from typing import Dict, Any

class StrategyAnalysisComponent:
    def render(self, analysis: Dict[str, Any]):
        """Render strategy analysis interface."""
        if not analysis:
            st.warning("전략 분석 데이터를 불러올 수 없습니다.")
            return
        
        st.subheader("전략 분석")
        
        # Price Analysis
        price_data = analysis.get('price_analysis', {})
        if price_data:
            st.write("가격 분석")
            
            col1, col2 = st.columns(2)
            with col1:
                current_price = price_data.get('current_price', 0)
                ma20_price = price_data.get('ma20', 0)
                price_diff = ((current_price - ma20_price) / ma20_price * 100) if ma20_price else 0
                
                price_status = "적정" if abs(price_diff) <= 3 else ("고평가" if price_diff > 3 else "저평가")
                price_color = "🟡" if abs(price_diff) <= 3 else ("🔴" if price_diff > 3 else "🟢")
                
                st.metric(
                    "현재가 상태",
                    f"{price_status} {price_color}",
                    f"20일 평균 대비 {price_diff:.1f}%"
                )
            
            with col2:
                bb_data = price_data.get('bollinger_bands', {})
                if bb_data:
                    upper = bb_data.get('upper', 0)
                    lower = bb_data.get('lower', 0)
                    
                    if current_price > upper:
                        bb_status = "매도 고려"
                        bb_color = "🔴"
                    elif current_price < lower:
                        bb_status = "매수 고려"
                        bb_color = "🟢"
                    else:
                        bb_status = "중립 구간"
                        bb_color = "🟡"
                    
                    st.metric(
                        "볼린저 밴드 위치",
                        f"{bb_status} {bb_color}",
                        f"상단: {((upper - current_price) / current_price * 100):.1f}% / 하단: {((current_price - lower) / current_price * 100):.1f}%"
                    )
            
            # 가격 매매 구간 설명
            st.markdown("""
            **가격 매매 구간**:
            - 🟢 **매수 고려**: 현재가가 20일 평균 대비 3% 이상 하락 또는 볼린저 밴드 하단 도달
            - 🟡 **중립 구간**: 현재가가 20일 평균의 ±3% 이내
            - 🔴 **매도 고려**: 현재가가 20일 평균 대비 3% 이상 상승 또는 볼린저 밴드 상단 도달
            """)
        
        # Strategy Signals
        signals = analysis.get('signals', {})
        if signals:
            st.write("---")
            st.write("매매 신호")
            
            # 신호 설명 추가
            signal_descriptions = {
                'ma': {
                    'BULLISH': '상승 추세 (단기 이평선이 장기 이평선 위)',
                    'BEARISH': '하락 추세 (단기 이평선이 장기 이평선 아래)',
                    'NEUTRAL': '중립 (추세 불분명)'
                },
                'rsi': {
                    'BULLISH': '과매도 구간 (RSI < 30)',
                    'BEARISH': '과매수 구간 (RSI > 70)',
                    'NEUTRAL': '중립 구간 (30 < RSI < 70)'
                },
                'macd': {
                    'BULLISH': 'MACD가 시그널선 위',
                    'BEARISH': 'MACD가 시그널선 아래',
                    'NEUTRAL': 'MACD와 시그널선 교차'
                }
            }
            
            # 매매 신호 표시
            for signal_type, signal_data in signals.items():
                signal = signal_data.get('signal', 'NEUTRAL')
                value = signal_data.get('value', 'N/A')
                description = signal_descriptions.get(signal_type, {}).get(signal, '')
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(
                        signal_type.upper(),
                        value,
                        signal
                    )
                with col2:
                    st.markdown(f"**설명**: {description}")
        
        # 매매 조건 표시
        st.write("---")
        st.write("매매 조건")
        
        # 매수 조건
        st.markdown("""
        **매수 조건** (다음 중 2개 이상 만족 및 가격 조건 충족):
        - MA: 단기 이동평균이 장기 이동평균 위로 교차
        - RSI: 30 이하 (과매도 구간)
        - MACD: MACD선이 시그널선 위로 교차
        - 가격: 20일 평균 대비 3% 이상 하락 또는 볼린저 밴드 하단
        """)
        
        # 매도 조건
        st.markdown("""
        **매도 조건** (다음 중 2개 이상 만족 및 가격 조건 충족):
        - MA: 단기 이동평균이 장기 이동평균 아래로 교차
        - RSI: 70 이상 (과매수 구간)
        - MACD: MACD선이 시그널선 아래로 교차
        - 가격: 20일 평균 대비 3% 이상 상승 또는 볼린저 밴드 상단
        """)
        
        # Risk Analysis
        risk = analysis.get('risk', {})
        if risk:
            st.write("---")
            st.write("리스크 분석")
            col1, col2 = st.columns(2)
            
            with col1:
                volatility = risk.get('volatility', 0)
                st.metric(
                    "변동성",
                    f"{volatility:.2f}%",
                    "높음" if volatility > 5 else ("중간" if volatility > 3 else "낮음")
                )
            
            with col2:
                risk_score = risk.get('risk_score', 0)
                st.metric(
                    "리스크 점수",
                    f"{risk_score:.1f}/10",
                    "매매 제한" if risk_score > 7 else "주의" if risk_score > 5 else "정상"
                )
        
        # Performance Metrics
        performance = analysis.get('performance', {})
        if performance:
            st.write("---")
            st.write("성과 지표")
            
            # 첫 번째 행: 승률과 거래일
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "승률",
                    f"{performance.get('win_rate', 0):.1f}%",
                    f"거래일 {performance.get('trading_days', 0)}일"
                )
            
            with col2:
                st.metric(
                    "연간 변동성",
                    f"{performance.get('volatility', 0):.1f}%"
                )
            
            # 두 번째 행: 수익률과 샤프 비율
            col3, col4 = st.columns(2)
            with col3:
                st.metric(
                    "보유 수익률",
                    f"{performance.get('return_rate', 0):.2f}%",
                    "시작 시점 대비"
                )
            
            with col4:
                sharpe = performance.get('sharpe_ratio', 0)
                st.metric(
                    "샤프 비율",
                    f"{sharpe:.2f}",
                    "위험조정수익률"
                )
            
            # 설명 추가
            with st.expander("성과 지표 설명"):
                st.markdown("""
                * **승률**: 0.5% 이상의 가격 변동이 있는 거래일 중 상승한 날의 비율
                * **연간 변동성**: 일일 수익률의 표준편차를 연율화한 값
                * **보유 수익률**: 시작 시점부터 현재까지의 누적 수익률
                * **샤프 비율**: 무위험 수익률 대비 초과수익률의 변동성 조정 성과
                    * 2 이상: 매우 우수
                    * 1~2: 우수
                    * 0~1: 보통
                    * 0 미만: 저조
                """)
            
            # 매매 실행 조건 설명
            st.write("---")
            st.write("매매 실행 조건")
            
            # 실행 조건 표시
            st.markdown("""
            **매매 실행을 위한 필수 조건**:
            1. 리스크 점수가 7 이하일 것
            2. 최소 2개 이상의 지표가 같은 방향의 신호를 보일 것
            3. 변동성이 정상 범위 내일 것 (5% 이하)
            4. 가격이 매매 적정 구간에 있을 것
            
            **매매 제한 조건**:
            1. 리스크 점수가 7 초과
            2. 지표들의 신호가 불일치
            3. 변동성이 매우 높음 (5% 초과)
            4. 가격이 매매 부적정 구간
            """) 