"""
Comprehensive market analysis component
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

def calculate_signal_strength(indicators: Dict) -> Tuple[float, List[str]]:
    """Calculate overall signal strength and generate reasons."""
    try:
        if not indicators or not isinstance(indicators, dict):
            return 0, []

        buy_signals = []
        sell_signals = []
        signal_details = []
        
        # 이동평균선 크로스 분석
        ma_data = indicators.get('moving_averages', {})
        if isinstance(ma_data, dict) and 'MA5' in ma_data and 'MA20' in ma_data:
            ma5 = ma_data['MA5']
            ma20 = ma_data['MA20']
            if len(ma5) > 0 and len(ma20) > 0:
                ma5_last = ma5.iloc[-1]
                ma20_last = ma20.iloc[-1]
                if not (pd.isna(ma5_last) or pd.isna(ma20_last)):
                    if ma5_last > ma20_last:
                        buy_signals.append(0.6)
                        signal_details.append("단기 이동평균선이 장기 이동평균선을 상향 돌파")
                    else:
                        sell_signals.append(0.6)
                        signal_details.append("단기 이동평균선이 장기 이동평균선을 하향 돌파")

        # RSI 분석
        rsi = indicators.get('rsi', pd.Series())
        if len(rsi) > 0:
            rsi_value = rsi.iloc[-1]
            if not pd.isna(rsi_value):
                if rsi_value < 30:
                    buy_signals.append(0.8)
                    signal_details.append(f"RSI 과매도 구간 (RSI: {rsi_value:.1f})")
                elif rsi_value > 70:
                    sell_signals.append(0.8)
                    signal_details.append(f"RSI 과매수 구간 (RSI: {rsi_value:.1f})")

        # 볼린저 밴드 분석
        bb = indicators.get('bollinger_bands', {})
        if isinstance(bb, dict) and all(k in bb for k in ['lower', 'upper']):
            current_price = indicators.get('current_price', 0)
            if len(bb['lower']) > 0 and len(bb['upper']) > 0:
                lower_band = bb['lower'].iloc[-1]
                upper_band = bb['upper'].iloc[-1]
                if not (pd.isna(lower_band) or pd.isna(upper_band) or pd.isna(current_price)):
                    if current_price < lower_band:
                        buy_signals.append(0.7)
                        signal_details.append("볼린저 밴드 하단 지지")
                    elif current_price > upper_band:
                        sell_signals.append(0.7)
                        signal_details.append("볼린저 밴드 상단 돌파")

        # MACD 분석
        macd = indicators.get('macd', {})
        if isinstance(macd, dict) and all(k in macd for k in ['macd', 'signal']):
            if len(macd['macd']) > 0 and len(macd['signal']) > 0:
                macd_value = macd['macd'].iloc[-1]
                signal_value = macd['signal'].iloc[-1]
                if not (pd.isna(macd_value) or pd.isna(signal_value)):
                    if macd_value > signal_value:
                        buy_signals.append(0.65)
                        signal_details.append("MACD 골든크로스")
                    else:
                        sell_signals.append(0.65)
                        signal_details.append("MACD 데드크로스")

        # Stochastic 분석
        stoch = indicators.get('stochastic', {}).get('slow', {})
        if isinstance(stoch, dict) and all(k in stoch for k in ['k', 'd']):
            if len(stoch['k']) > 0 and len(stoch['d']) > 0:
                k = stoch['k'].iloc[-1]
                d = stoch['d'].iloc[-1]
                if not (pd.isna(k) or pd.isna(d)):
                    if k < 20 and d < 20:
                        buy_signals.append(0.75)
                        signal_details.append("스토캐스틱 과매도")
                    elif k > 80 and d > 80:
                        sell_signals.append(0.75)
                        signal_details.append("스토캐스틱 과매수")

        # 종합 신호 강도 계산
        if len(buy_signals) > len(sell_signals):
            signal_strength = sum(buy_signals) / len(buy_signals)
        elif len(sell_signals) > len(buy_signals):
            signal_strength = -sum(sell_signals) / len(sell_signals)
        else:
            if buy_signals and sell_signals:
                if sum(buy_signals) > sum(sell_signals):
                    signal_strength = sum(buy_signals) / len(buy_signals)
                else:
                    signal_strength = -sum(sell_signals) / len(sell_signals)
            else:
                signal_strength = 0

        return signal_strength, signal_details

    except Exception as e:
        return 0, [f"신호 강도 계산 중 오류 발생: {str(e)}"]

def predict_optimal_timing(indicators: Dict) -> str:
    """예상 최적 매매 시점을 분석."""
    try:
        if not indicators or not isinstance(indicators, dict):
            return "충분한 데이터가 없습니다."

        # 트렌드 분석
        ma_data = indicators.get('moving_averages', {})
        if not isinstance(ma_data, dict) or 'MA5' not in ma_data or 'MA20' not in ma_data:
            return "이동평균선 데이터가 부족합니다."

        ma5 = ma_data['MA5']
        ma20 = ma_data['MA20']
        
        if len(ma5) < 5 or len(ma20) < 5:
            return "충분한 데이터가 없습니다."

        if pd.isna(ma5.iloc[-1]) or pd.isna(ma20.iloc[-1]):
            return "유효하지 않은 이동평균선 데이터입니다."

        trend = "상승" if ma5.iloc[-1] > ma20.iloc[-1] else "하락"
        
        # RSI 분석
        rsi = indicators.get('rsi', pd.Series())
        if len(rsi) == 0 or pd.isna(rsi.iloc[-1]):
            return "RSI 데이터가 부족합니다."
        
        current_rsi = rsi.iloc[-1]
        rsi_momentum = "상승" if current_rsi > 50 else "하락"
        
        # MACD 방향
        macd = indicators.get('macd', {})
        if not isinstance(macd, dict) or 'macd' not in macd or 'signal' not in macd:
            return "MACD 데이터가 부족합니다."

        macd_line = macd['macd']
        signal_line = macd['signal']
        
        if len(macd_line) == 0 or len(signal_line) == 0:
            return "MACD 데이터가 부족합니다."

        if pd.isna(macd_line.iloc[-1]) or pd.isna(signal_line.iloc[-1]):
            return "유효하지 않은 MACD 데이터입니다."

        macd_direction = "상승" if macd_line.iloc[-1] > signal_line.iloc[-1] else "하락"
        
        # 변동성 분석
        bb = indicators.get('bollinger_bands', {})
        if isinstance(bb, dict) and 'volatility' in bb:
            volatility = bb['volatility']
            if volatility > 30:
                return "고변동성 구간입니다. 매매 시 주의가 필요합니다."
            elif volatility < 10:
                if trend == "상승" and rsi_momentum == "상승":
                    return "안정적인 상승 추세. 매수 시점으로 적합"
                elif trend == "하락" and rsi_momentum == "하락":
                    return "안정적인 하락 추세. 매도 시점으로 적합"

        # 분석 결과 해석
        if trend == "상승":
            if rsi_momentum == "상승" and macd_direction == "상승":
                return "현재 강한 상승 추세. 단기 매수 시점으로 적합"
            elif rsi_momentum == "하락":
                return "상승 추세이나 모멘텀 약화. 1-2일 조정 후 매수 고려"
        else:  # 하락 추세
            if rsi_momentum == "하락" and macd_direction == "하락":
                return "현재 하락 추세. 추가 하락 가능성 높음. 1주일 관망 추천"
            elif rsi_momentum == "상승":
                return "하락 추세이나 반등 신호 감지. 2-3일 후 매수 기회 있을 수 있음"

        return "뚜렷한 방향성 없음. 추가 관찰 필요"

    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        return "분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

def render_market_analysis(market: str, market_data: Dict):
    """시장 분석 탭 렌더링"""
    try:
        st.markdown("## 📊 종합 시장 분석")
        
        if not market_data or not isinstance(market_data, dict):
            st.warning("시장 데이터를 불러올 수 없습니다.")
            return

        if 'current_price' not in market_data or not market_data['current_price']:
            st.warning("현재가 데이터를 불러올 수 없습니다.")
            return

        if 'indicators' not in market_data or not isinstance(market_data['indicators'], dict):
            st.warning("기술적 지표 데이터를 불러올 수 없습니다.")
            return

        indicators = market_data['indicators']
        current_price = market_data.get('current_price', 0)

        # 신호 강도 계산
        signal_strength, signal_details = calculate_signal_strength(indicators)
        
        # 신호 강도 표시
        col1, col2 = st.columns(2)
        with col1:
            signal_color = "🟢" if signal_strength > 0 else "🔴" if signal_strength < 0 else "⚪"
            signal_text = "매수" if signal_strength > 0 else "매도" if signal_strength < 0 else "관망"
            confidence = abs(signal_strength) * 100
            st.metric(
                "현재 매매 신호",
                f"{signal_color} {signal_text}",
                f"신뢰도: {confidence:.1f}%"
            )
        
        with col2:
            st.metric(
                "현재가",
                f"₩{current_price:,.0f}",
                f"변동성: {market_data.get('volatility', 0):.2f}%"
            )

        # 세부 분석 표시
        with st.expander("📈 상세 분석 보기", expanded=True):
            if signal_details:
                st.markdown("### 주요 지표 분석")
                for detail in signal_details:
                    st.write(f"• {detail}")
            
            st.markdown("### 🎯 최적 매매 시점")
            timing_prediction = predict_optimal_timing(indicators)
            st.info(timing_prediction)

            # 기술적 지표 요약
            st.markdown("### 📊 기술적 지표 요약")
            
            # RSI
            rsi = indicators.get('rsi', pd.Series())
            if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]):
                rsi_value = rsi.iloc[-1]
                rsi_status = "과매수" if rsi_value > 70 else "과매도" if rsi_value < 30 else "중립"
                st.write(f"• RSI: {rsi_value:.1f} ({rsi_status})")
            
            # 볼린저 밴드
            bb = indicators.get('bollinger_bands', {})
            if isinstance(bb, dict) and all(k in bb for k in ['lower', 'upper']):
                if len(bb['lower']) > 0 and len(bb['upper']) > 0:
                    lower_band = bb['lower'].iloc[-1]
                    upper_band = bb['upper'].iloc[-1]
                    if not (pd.isna(lower_band) or pd.isna(upper_band)):
                        bb_position = (current_price - lower_band) / (upper_band - lower_band) * 100
                        st.write(f"• 볼린저 밴드 위치: {bb_position:.1f}%")
            
            # MACD
            macd = indicators.get('macd', {})
            if isinstance(macd, dict) and all(k in macd for k in ['macd', 'signal']):
                if len(macd['macd']) > 0 and len(macd['signal']) > 0:
                    macd_value = macd['macd'].iloc[-1]
                    signal_value = macd['signal'].iloc[-1]
                    if not (pd.isna(macd_value) or pd.isna(signal_value)):
                        st.write(f"• MACD: {macd_value:.1f} (Signal: {signal_value:.1f})")
            
            # Stochastic
            stoch = indicators.get('stochastic', {}).get('slow', {})
            if isinstance(stoch, dict) and all(k in stoch for k in ['k', 'd']):
                if len(stoch['k']) > 0 and len(stoch['d']) > 0:
                    k = stoch['k'].iloc[-1]
                    d = stoch['d'].iloc[-1]
                    if not (pd.isna(k) or pd.isna(d)):
                        st.write(f"• Stochastic: K({k:.1f}) D({d:.1f})")

        # 거래량 분석
        with st.expander("📊 거래량 분석", expanded=False):
            volume = market_data.get('volume', [])
            if isinstance(volume, list) and len(volume) >= 5:
                avg_volume = np.mean(volume[-5:])
                current_volume = volume[-1]
                if avg_volume > 0:
                    volume_ratio = (current_volume / avg_volume) * 100
                    st.write(f"• 거래량 비율: {volume_ratio:.1f}% (5일 평균 대비)")
                    
                    if volume_ratio > 150:
                        st.write("🔥 거래량 급증! 강한 추세 전환 가능성")
                    elif volume_ratio < 50:
                        st.write("❄️ 거래량 저조. 관망 구간")

        # 투자 위험도
        with st.expander("⚠️ 투자 위험도", expanded=False):
            volatility = market_data.get('volatility', 0)
            if volatility > 5:
                st.warning("현재 변동성이 매우 높습니다. 신중한 투자가 필요합니다.")
            elif volatility < 1:
                st.info("현재 변동성이 낮습니다. 안정적인 거래가 가능합니다.")
            
            # 손절가/목표가 제안
            stop_loss = current_price * 0.97  # 3% 손절
            take_profit = current_price * 1.05  # 5% 익절
            st.write(f"• 제안 손절가: ₩{stop_loss:,.0f} (-3%)")
            st.write(f"• 제안 목표가: ₩{take_profit:,.0f} (+5%)")

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {str(e)}") 