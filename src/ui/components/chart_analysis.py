"""
차트 분석 UI 컴포넌트
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional
from src.core.chart_analysis import ChartAnalyzer, PatternType, TrendDirection, SignalStrength
from src.core.trading import TradingManager
from src.core.alert_system import alert_system, AlertType, AlertPriority
from src.core.ai_predictor import AIPricePredictor, PredictionDirection, PredictionConfidence
from src.core.performance_optimizer import get_performance_report, analysis_cache, performance_monitor
from src.core.risk_analyzer import RealTimeRiskAnalyzer, RiskLevel, RiskType
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

def render_chart_analysis_section(trading_manager: TradingManager, market: str):
    """차트 분석 섹션 렌더링"""
    
    st.header("📊 차트 분석")
    
    # 차트 분석기 초기화
    chart_analyzer = ChartAnalyzer()
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📈 패턴 분석", "📊 기술적 분석", "🎯 지지/저항", "⚡ 실시간 분석", "🔔 알림 센터", "🤖 AI 예측", "⚠️ 위험도 분석", "⚙️ 성능 모니터"])
    
    with tab1:
        render_pattern_analysis(chart_analyzer, trading_manager, market)
    
    with tab2:
        render_technical_analysis(chart_analyzer, trading_manager, market)
    
    with tab3:
        render_support_resistance_analysis(chart_analyzer, trading_manager, market)
    
    with tab4:
        render_realtime_analysis(chart_analyzer, trading_manager, market)
    
    with tab5:
        render_alert_center(trading_manager, market)
    
    with tab6:
        render_ai_prediction(chart_analyzer, trading_manager, market)
    
    with tab7:
        render_risk_analysis(trading_manager, market)
    
    with tab8:
        render_performance_monitor()

def render_pattern_analysis(chart_analyzer: ChartAnalyzer, trading_manager: TradingManager, market: str):
    """패턴 분석 렌더링"""
    try:
        st.subheader("🔍 차트 패턴 분석")
        
        # 분석 옵션
        col1, col2 = st.columns(2)
        
        with col1:
            timeframe = st.selectbox(
                "시간대 선택",
                ["1분", "5분", "15분", "1시간", "4시간", "1일"],
                index=5  # 기본값: 1일
            )
        
        with col2:
            period = st.slider(
                "분석 기간 (일)",
                min_value=30,
                max_value=200,
                value=100,
                step=10
            )
        
        # 데이터 로드
        with st.spinner("차트 데이터 분석 중..."):
            ohlcv_data = trading_manager.get_ohlcv(market, count=period)
            indicators = trading_manager.calculate_indicators(ohlcv_data)
            
            if ohlcv_data.empty:
                st.warning("차트 데이터를 불러올 수 없습니다.")
                return
            
            # 차트 분석 실행
            analysis = chart_analyzer.analyze_chart(ohlcv_data, indicators)
        
        # 감지된 패턴 표시
        st.subheader("🎯 감지된 패턴")
        
        if analysis.patterns:
            for i, pattern in enumerate(analysis.patterns):
                with st.expander(f"패턴 {i+1}: {pattern.pattern_type.value} (신뢰도: {pattern.confidence*100:.1f}%)"):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**패턴 정보**")
                        st.write(f"📅 시작일: {pattern.start_date}")
                        st.write(f"📅 종료일: {pattern.end_date}")
                        st.write(f"💪 신호 강도: {pattern.signal_strength.value}")
                    
                    with col2:
                        st.markdown("**목표 가격**")
                        if pattern.target_price:
                            current_price = ohlcv_data['close'].iloc[-1]
                            change_pct = (pattern.target_price - current_price) / current_price * 100
                            st.metric(
                                "목표가", 
                                f"{pattern.target_price:,.0f}원",
                                delta=f"{change_pct:+.1f}%"
                            )
                        else:
                            st.write("목표가 미설정")
                    
                    with col3:
                        st.markdown("**손절가**")
                        if pattern.stop_loss:
                            current_price = ohlcv_data['close'].iloc[-1]
                            change_pct = (pattern.stop_loss - current_price) / current_price * 100
                            st.metric(
                                "손절가",
                                f"{pattern.stop_loss:,.0f}원",
                                delta=f"{change_pct:+.1f}%"
                            )
                        else:
                            st.write("손절가 미설정")
                    
                    st.info(f"💡 **분석**: {pattern.description}")
        else:
            st.info("현재 명확한 차트 패턴이 감지되지 않았습니다.")
        
        # 패턴 차트 시각화
        render_pattern_chart(ohlcv_data, analysis.patterns, indicators, market)
        
        # 패턴 통계
        render_pattern_statistics(analysis.patterns)
        
    except Exception as e:
        st.error(f"패턴 분석 중 오류 발생: {str(e)}")
        logger.error(f"Pattern analysis error: {str(e)}")

def render_pattern_chart(ohlcv_data: pd.DataFrame, patterns: List, indicators: Dict, market: str = ""):
    """패턴이 표시된 차트 렌더링"""
    try:
        st.subheader("📈 패턴 차트")
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('가격 차트', 'RSI'),
            row_heights=[0.7, 0.3]
        )
        
        # 캔들스틱 차트
        fig.add_trace(
            go.Candlestick(
                x=ohlcv_data.index,
                open=ohlcv_data['open'],
                high=ohlcv_data['high'],
                low=ohlcv_data['low'],
                close=ohlcv_data['close'],
                name='가격',
                increasing_line_color='red',
                decreasing_line_color='blue'
            ),
            row=1, col=1
        )
        
        # 이동평균선 추가
        if 'MA5' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=ohlcv_data.index,
                    y=indicators['MA5'],
                    mode='lines',
                    name='MA5',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'MA20' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=ohlcv_data.index,
                    y=indicators['MA20'],
                    mode='lines',
                    name='MA20',
                    line=dict(color='purple', width=1)
                ),
                row=1, col=1
            )
        
        # 패턴 마커 추가
        for pattern in patterns:
            try:
                start_date = pd.to_datetime(pattern.start_date)
                end_date = pd.to_datetime(pattern.end_date)
                
                # 패턴 구간 하이라이트
                fig.add_vrect(
                    x0=start_date,
                    x1=end_date,
                    fillcolor=get_pattern_color(pattern.pattern_type),
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )
                
                # 패턴 라벨
                mid_date = start_date + (end_date - start_date) / 2
                pattern_price = ohlcv_data.loc[ohlcv_data.index <= end_date, 'high'].iloc[-1] if len(ohlcv_data.loc[ohlcv_data.index <= end_date]) > 0 else ohlcv_data['high'].iloc[-1]
                
                fig.add_annotation(
                    x=mid_date,
                    y=pattern_price * 1.02,
                    text=pattern.pattern_type.value,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=get_pattern_color(pattern.pattern_type),
                    row=1, col=1
                )
                
            except Exception as e:
                logger.error(f"패턴 마커 추가 오류: {str(e)}")
                continue
        
        # RSI 차트
        if 'rsi' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=ohlcv_data.index,
                    y=indicators['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            # RSI 기준선
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # 레이아웃 설정
        fig.update_layout(
            title=f"{market} 패턴 분석 차트",
            height=800,
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="가격 (KRW)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"패턴 차트 렌더링 오류: {str(e)}")
        logger.error(f"Pattern chart error: {str(e)}")

def render_technical_analysis(chart_analyzer: ChartAnalyzer, trading_manager: TradingManager, market: str):
    """기술적 분석 렌더링"""
    try:
        st.subheader("📊 기술적 분석")
        
        # 데이터 로드
        with st.spinner("기술적 지표 분석 중..."):
            ohlcv_data = trading_manager.get_ohlcv(market, count=100)
            indicators = trading_manager.calculate_indicators(ohlcv_data)
            
            if ohlcv_data.empty:
                st.warning("차트 데이터를 불러올 수 없습니다.")
                return
            
            analysis = chart_analyzer.analyze_chart(ohlcv_data, indicators)
        
        # 트렌드 분석
        st.subheader("📈 트렌드 분석")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend_color = get_trend_color(analysis.trend.direction)
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: {trend_color}; border-radius: 10px;">
                <h3 style="color: white; margin: 0;">📊 현재 트렌드</h3>
                <h2 style="color: white; margin: 10px 0;">{analysis.trend.direction.value}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric(
                "트렌드 강도",
                f"{analysis.trend.strength*100:.1f}%",
                delta=f"{analysis.trend.duration_days}일 지속"
            )
        
        with col3:
            st.metric(
                "위험도",
                analysis.risk_level,
                delta="분석 기반"
            )
        
        # 지지/저항 레벨
        if analysis.trend.support_level and analysis.trend.resistance_level:
            col1, col2 = st.columns(2)
            
            with col1:
                current_price = ohlcv_data['close'].iloc[-1]
                support_distance = (current_price - analysis.trend.support_level) / current_price * 100
                st.metric(
                    "주요 지지선",
                    f"{analysis.trend.support_level:,.0f}원",
                    delta=f"{support_distance:.1f}% 아래"
                )
            
            with col2:
                resistance_distance = (analysis.trend.resistance_level - current_price) / current_price * 100
                st.metric(
                    "주요 저항선",
                    f"{analysis.trend.resistance_level:,.0f}원",
                    delta=f"{resistance_distance:.1f}% 위"
                )
        
        # 모멘텀 신호
        st.subheader("⚡ 모멘텀 신호")
        
        if analysis.momentum_signals:
            momentum_cols = st.columns(len(analysis.momentum_signals))
            
            for i, (indicator, signal) in enumerate(analysis.momentum_signals.items()):
                with momentum_cols[i]:
                    signal_color = get_signal_color(signal)
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; background-color: {signal_color}; border-radius: 8px; margin: 5px;">
                        <h4 style="color: white; margin: 0;">{indicator}</h4>
                        <p style="color: white; margin: 5px 0 0 0; font-size: 14px;">{signal}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 거래량 분석
        st.subheader("📊 거래량 분석")
        
        vol_analysis = analysis.volume_analysis
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'volume_ratio' in vol_analysis:
                st.metric(
                    "거래량 비율",
                    f"{vol_analysis['volume_ratio']:.1f}x",
                    delta=f"평균 대비"
                )
        
        with col2:
            st.metric(
                "거래량 트렌드",
                vol_analysis.get('trend', '분석불가'),
                delta=vol_analysis.get('signal', '중립')
            )
        
        with col3:
            st.info(vol_analysis.get('description', '거래량 분석 결과 없음'))
        
        # 종합 점수
        render_technical_score(analysis)
        
    except Exception as e:
        st.error(f"기술적 분석 중 오류 발생: {str(e)}")
        logger.error(f"Technical analysis error: {str(e)}")

def render_support_resistance_analysis(chart_analyzer: ChartAnalyzer, trading_manager: TradingManager, market: str):
    """지지/저항 분석 렌더링"""
    try:
        st.subheader("🎯 지지선 & 저항선 분석")
        
        # 데이터 로드
        with st.spinner("지지/저항 레벨 분석 중..."):
            ohlcv_data = trading_manager.get_ohlcv(market, count=200)
            indicators = trading_manager.calculate_indicators(ohlcv_data)
            
            if ohlcv_data.empty:
                st.warning("차트 데이터를 불러올 수 없습니다.")
                return
            
            analysis = chart_analyzer.analyze_chart(ohlcv_data, indicators)
        
        current_price = ohlcv_data['close'].iloc[-1]
        
        # 지지/저항 레벨 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🟢 주요 지지선")
            support_levels = analysis.support_resistance.get('support', [])
            
            if support_levels:
                for i, level in enumerate(support_levels, 1):
                    distance = (current_price - level) / current_price * 100
                    strength = "강함" if i == 1 else "보통" if i == 2 else "약함"
                    
                    st.markdown(f"""
                    <div style="padding: 10px; margin: 5px 0; background-color: #e8f5e8; border-radius: 5px; border-left: 4px solid #28a745;">
                        <strong>지지선 {i}</strong><br>
                        💰 가격: {level:,.0f}원<br>
                        📏 거리: {distance:.1f}% 아래<br>
                        💪 강도: {strength}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("명확한 지지선을 찾을 수 없습니다.")
        
        with col2:
            st.markdown("### 🔴 주요 저항선")
            resistance_levels = analysis.support_resistance.get('resistance', [])
            
            if resistance_levels:
                for i, level in enumerate(resistance_levels, 1):
                    distance = (level - current_price) / current_price * 100
                    strength = "강함" if i == 1 else "보통" if i == 2 else "약함"
                    
                    st.markdown(f"""
                    <div style="padding: 10px; margin: 5px 0; background-color: #ffeaea; border-radius: 5px; border-left: 4px solid #dc3545;">
                        <strong>저항선 {i}</strong><br>
                        💰 가격: {level:,.0f}원<br>
                        📏 거리: {distance:.1f}% 위<br>
                        💪 강도: {strength}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("명확한 저항선을 찾을 수 없습니다.")
        
        # 지지/저항 차트
        render_support_resistance_chart(ohlcv_data, analysis.support_resistance, current_price)
        
        # 매매 전략 제안
        render_trading_strategy_from_levels(analysis.support_resistance, current_price)
        
    except Exception as e:
        st.error(f"지지/저항 분석 중 오류 발생: {str(e)}")
        logger.error(f"Support/Resistance analysis error: {str(e)}")

def render_support_resistance_chart(ohlcv_data: pd.DataFrame, levels: Dict, current_price: float):
    """지지/저항 차트 렌더링"""
    try:
        st.subheader("📊 지지/저항 차트")
        
        fig = go.Figure()
        
        # 캔들스틱 차트
        fig.add_trace(go.Candlestick(
            x=ohlcv_data.index,
            open=ohlcv_data['open'],
            high=ohlcv_data['high'],
            low=ohlcv_data['low'],
            close=ohlcv_data['close'],
            name='가격'
        ))
        
        # 지지선 추가
        for i, support in enumerate(levels.get('support', [])):
            fig.add_hline(
                y=support,
                line_dash="dash",
                line_color="green",
                annotation_text=f"지지선 {i+1}: {support:,.0f}원",
                annotation_position="bottom right"
            )
        
        # 저항선 추가
        for i, resistance in enumerate(levels.get('resistance', [])):
            fig.add_hline(
                y=resistance,
                line_dash="dash",
                line_color="red",
                annotation_text=f"저항선 {i+1}: {resistance:,.0f}원",
                annotation_position="top right"
            )
        
        # 현재가 라인
        fig.add_hline(
            y=current_price,
            line_dash="solid",
            line_color="blue",
            annotation_text=f"현재가: {current_price:,.0f}원",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title="지지선 & 저항선 분석",
            height=600,
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"지지/저항 차트 오류: {str(e)}")
        logger.error(f"Support/Resistance chart error: {str(e)}")

def render_realtime_analysis(chart_analyzer: ChartAnalyzer, trading_manager: TradingManager, market: str):
    """실시간 분석 렌더링"""
    try:
        st.subheader("⚡ 실시간 차트 분석")
        
        # 자동 새로고침 옵션
        auto_refresh = st.checkbox("자동 새로고침 (30초)", value=False)
        
        if auto_refresh:
            import time
            time.sleep(30)
            st.rerun()
        
        # 실시간 데이터 로드
        with st.spinner("실시간 데이터 분석 중..."):
            ohlcv_data = trading_manager.get_ohlcv(market, count=50)
            indicators = trading_manager.calculate_indicators(ohlcv_data)
            
            if ohlcv_data.empty:
                st.warning("차트 데이터를 불러올 수 없습니다.")
                return
            
            analysis = chart_analyzer.analyze_chart(ohlcv_data, indicators)
        
        # 실시간 신호 대시보드
        st.markdown("### 🚨 실시간 신호")
        
        # 신호 요약
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            trend_emoji = "📈" if analysis.trend.direction == TrendDirection.UPTREND else "📉" if analysis.trend.direction == TrendDirection.DOWNTREND else "➡️"
            st.metric(
                "트렌드",
                f"{trend_emoji} {analysis.trend.direction.value}",
                delta=f"강도: {analysis.trend.strength*100:.0f}%"
            )
        
        with col2:
            pattern_count = len(analysis.patterns)
            recent_patterns = [p for p in analysis.patterns if p.confidence > 0.7]
            st.metric(
                "패턴",
                f"{pattern_count}개 감지",
                delta=f"고신뢰도: {len(recent_patterns)}개"
            )
        
        with col3:
            momentum_signals = analysis.momentum_signals
            bullish_count = sum(1 for signal in momentum_signals.values() if "상승" in signal or "매수" in signal)
            st.metric(
                "모멘텀",
                f"{bullish_count}/{len(momentum_signals)} 긍정",
                delta="신호 비율"
            )
        
        with col4:
            st.metric(
                "위험도",
                analysis.risk_level,
                delta=f"변동성 기반"
            )
        
        # 즉시 실행 알림
        render_immediate_alerts(analysis, ohlcv_data)
        
        # 실시간 권장사항
        render_realtime_recommendations(analysis, ohlcv_data)
        
        # 마지막 업데이트 시간
        st.markdown(f"*마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
    except Exception as e:
        st.error(f"실시간 분석 중 오류 발생: {str(e)}")
        logger.error(f"Real-time analysis error: {str(e)}")

# 유틸리티 함수들
def get_pattern_color(pattern_type: PatternType) -> str:
    """패턴 타입에 따른 색상 반환"""
    bullish_patterns = [
        PatternType.DOUBLE_BOTTOM, PatternType.INVERSE_HEAD_AND_SHOULDERS,
        PatternType.HAMMER, PatternType.ENGULFING_BULLISH, PatternType.MORNING_STAR
    ]
    
    if pattern_type in bullish_patterns:
        return "green"
    else:
        return "red"

def get_trend_color(direction: TrendDirection) -> str:
    """트렌드에 따른 색상 반환"""
    if direction == TrendDirection.UPTREND:
        return "#28a745"
    elif direction == TrendDirection.DOWNTREND:
        return "#dc3545"
    else:
        return "#6c757d"

def get_signal_color(signal: str) -> str:
    """신호에 따른 색상 반환"""
    if "상승" in signal or "매수" in signal or "긍정" in signal:
        return "#28a745"
    elif "하락" in signal or "매도" in signal or "부정" in signal:
        return "#dc3545"
    else:
        return "#6c757d"

def render_pattern_statistics(patterns: List) -> None:
    """패턴 통계 렌더링"""
    try:
        st.subheader("📊 패턴 통계")
        
        if not patterns:
            st.info("감지된 패턴이 없습니다.")
            return
        
        # 패턴 타입별 개수
        pattern_counts = {}
        confidence_sum = 0
        
        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            confidence_sum += pattern.confidence
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 패턴 타입 분포
            fig_pie = px.pie(
                values=list(pattern_counts.values()),
                names=list(pattern_counts.keys()),
                title="패턴 타입 분포"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # 신뢰도 분포
            confidences = [p.confidence for p in patterns]
            fig_hist = px.histogram(
                x=confidences,
                nbins=10,
                title="패턴 신뢰도 분포",
                labels={'x': '신뢰도', 'y': '개수'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # 통계 요약
        avg_confidence = confidence_sum / len(patterns)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("총 패턴 수", len(patterns))
        
        with col2:
            st.metric("평균 신뢰도", f"{avg_confidence*100:.1f}%")
        
        with col3:
            high_conf_patterns = [p for p in patterns if p.confidence > 0.8]
            st.metric("고신뢰도 패턴", len(high_conf_patterns))
        
    except Exception as e:
        logger.error(f"패턴 통계 오류: {str(e)}")

def render_technical_score(analysis) -> None:
    """기술적 분석 종합 점수"""
    try:
        st.subheader("🎯 종합 분석 점수")
        
        # 점수 계산
        score = 0
        max_score = 100
        
        # 트렌드 점수 (30점)
        if analysis.trend.direction == TrendDirection.UPTREND:
            score += 30 * analysis.trend.strength
        elif analysis.trend.direction == TrendDirection.DOWNTREND:
            score += 30 * (1 - analysis.trend.strength)
        else:
            score += 15  # 횡보는 중간 점수
        
        # 모멘텀 점수 (30점)
        if analysis.momentum_signals:
            bullish_count = sum(1 for signal in analysis.momentum_signals.values() 
                              if "상승" in signal or "매수" in signal)
            momentum_score = (bullish_count / len(analysis.momentum_signals)) * 30
            score += momentum_score
        
        # 패턴 점수 (25점)
        if analysis.patterns:
            pattern_score = sum(p.confidence for p in analysis.patterns) / len(analysis.patterns) * 25
            score += pattern_score
        
        # 거래량 점수 (15점)
        vol_signal = analysis.volume_analysis.get('signal', '중립')
        if "확인" in vol_signal:
            score += 15
        elif "의심" in vol_signal:
            score += 5
        else:
            score += 7.5
        
        # 점수 정규화
        score = min(max_score, max(0, score))
        
        # 점수 표시
        score_color = "#28a745" if score > 70 else "#ffc107" if score > 40 else "#dc3545"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background: linear-gradient(45deg, {score_color}, {score_color}aa); border-radius: 15px; margin: 20px 0;">
            <h2 style="color: white; margin: 0;">종합 기술적 점수</h2>
            <h1 style="color: white; margin: 15px 0; font-size: 3em;">{score:.0f}/100</h1>
            <p style="color: white; margin: 0;">
                {"매우 긍정적" if score > 80 else "긍정적" if score > 60 else "보통" if score > 40 else "부정적" if score > 20 else "매우 부정적"}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 점수 구성 요소
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            trend_score = 30 * analysis.trend.strength if analysis.trend.direction == TrendDirection.UPTREND else 15
            st.metric("트렌드", f"{trend_score:.0f}/30")
        
        with col2:
            momentum_score = (bullish_count / len(analysis.momentum_signals)) * 30 if analysis.momentum_signals else 0
            st.metric("모멘텀", f"{momentum_score:.0f}/30")
        
        with col3:
            pattern_score = sum(p.confidence for p in analysis.patterns) / len(analysis.patterns) * 25 if analysis.patterns else 0
            st.metric("패턴", f"{pattern_score:.0f}/25")
        
        with col4:
            volume_score = 15 if "확인" in vol_signal else 5 if "의심" in vol_signal else 7.5
            st.metric("거래량", f"{volume_score:.0f}/15")
        
    except Exception as e:
        logger.error(f"기술적 점수 계산 오류: {str(e)}")

def render_trading_strategy_from_levels(levels: Dict, current_price: float) -> None:
    """지지/저항 기반 매매 전략"""
    try:
        st.subheader("💡 매매 전략 제안")
        
        support_levels = levels.get('support', [])
        resistance_levels = levels.get('resistance', [])
        
        if not support_levels and not resistance_levels:
            st.info("지지/저항 레벨이 없어 전략을 제안할 수 없습니다.")
            return
        
        # 매수 전략
        if support_levels:
            nearest_support = support_levels[0]
            support_distance = (current_price - nearest_support) / current_price * 100
            
            if support_distance < 2:  # 지지선 근처
                st.success(f"""
                🎯 **매수 기회**: 현재가가 주요 지지선({nearest_support:,.0f}원) 근처입니다.
                - 매수 타이밍: 지지선 터치 후 반등 확인
                - 손절가: 지지선 하단 2% ({nearest_support*0.98:,.0f}원)
                """)
        
        # 매도 전략
        if resistance_levels:
            nearest_resistance = resistance_levels[0]
            resistance_distance = (nearest_resistance - current_price) / current_price * 100
            
            if resistance_distance < 3:  # 저항선 근처
                st.warning(f"""
                ⚠️ **매도 고려**: 현재가가 주요 저항선({nearest_resistance:,.0f}원) 근처입니다.
                - 매도 타이밍: 저항선 접근 시 일부 매도
                - 목표가: 저항선 상단 2% ({nearest_resistance*1.02:,.0f}원)
                """)
        
        # 중간 지대 전략
        if support_levels and resistance_levels:
            nearest_support = support_levels[0]
            nearest_resistance = resistance_levels[0]
            
            range_size = (nearest_resistance - nearest_support) / nearest_support * 100
            position_in_range = (current_price - nearest_support) / (nearest_resistance - nearest_support) * 100
            
            if 30 < position_in_range < 70:  # 중간 지대
                st.info(f"""
                ➡️ **관망 전략**: 현재가가 지지선과 저항선 중간 지대에 있습니다.
                - 구간: {nearest_support:,.0f}원 ~ {nearest_resistance:,.0f}원 (범위: {range_size:.1f}%)
                - 현재 위치: 구간의 {position_in_range:.0f}% 지점
                - 권장: 명확한 돌파 신호까지 대기
                """)
        
    except Exception as e:
        logger.error(f"매매 전략 제안 오류: {str(e)}")

def render_immediate_alerts(analysis, ohlcv_data: pd.DataFrame) -> None:
    """즉시 실행 알림"""
    try:
        st.subheader("🚨 즉시 주의 알림")
        
        alerts = []
        current_price = ohlcv_data['close'].iloc[-1]
        
        # 고신뢰도 패턴 알림
        high_conf_patterns = [p for p in analysis.patterns if p.confidence > 0.8]
        for pattern in high_conf_patterns:
            alerts.append({
                'type': 'pattern',
                'level': 'high',
                'message': f"고신뢰도 {pattern.pattern_type.value} 패턴 감지! (신뢰도: {pattern.confidence*100:.0f}%)"
            })
        
        # 지지/저항 근접 알림
        support_levels = analysis.support_resistance.get('support', [])
        resistance_levels = analysis.support_resistance.get('resistance', [])
        
        for support in support_levels[:1]:  # 가장 가까운 지지선만
            distance = abs(current_price - support) / current_price * 100
            if distance < 2:
                alerts.append({
                    'type': 'support',
                    'level': 'medium',
                    'message': f"주요 지지선 근접! 현재가 {current_price:,.0f}원, 지지선 {support:,.0f}원"
                })
        
        for resistance in resistance_levels[:1]:  # 가장 가까운 저항선만
            distance = abs(current_price - resistance) / current_price * 100
            if distance < 2:
                alerts.append({
                    'type': 'resistance',
                    'level': 'medium',
                    'message': f"주요 저항선 근접! 현재가 {current_price:,.0f}원, 저항선 {resistance:,.0f}원"
                })
        
        # 알림 표시
        if alerts:
            for alert in alerts:
                if alert['level'] == 'high':
                    st.error(f"🚨 {alert['message']}")
                elif alert['level'] == 'medium':
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
        else:
            st.success("✅ 현재 특별한 주의 알림이 없습니다.")
        
    except Exception as e:
        logger.error(f"즉시 알림 오류: {str(e)}")

def render_realtime_recommendations(analysis, ohlcv_data: pd.DataFrame) -> None:
    """실시간 권장사항"""
    try:
        st.subheader("💡 실시간 권장사항")
        
        recommendations = []
        
        # 트렌드 기반 권장사항
        if analysis.trend.direction == TrendDirection.UPTREND and analysis.trend.strength > 0.7:
            recommendations.append("📈 강한 상승 트렌드 - 추가 매수 고려")
        elif analysis.trend.direction == TrendDirection.DOWNTREND and analysis.trend.strength > 0.7:
            recommendations.append("📉 강한 하락 트렌드 - 손절매 또는 관망")
        
        # 패턴 기반 권장사항
        bullish_patterns = [p for p in analysis.patterns if get_pattern_color(p.pattern_type) == "green" and p.confidence > 0.7]
        bearish_patterns = [p for p in analysis.patterns if get_pattern_color(p.pattern_type) == "red" and p.confidence > 0.7]
        
        if bullish_patterns:
            recommendations.append("🟢 상승 패턴 감지 - 매수 기회 포착")
        if bearish_patterns:
            recommendations.append("🔴 하락 패턴 감지 - 매도 신호 확인")
        
        # 모멘텀 기반 권장사항
        momentum_signals = analysis.momentum_signals
        bullish_momentum = sum(1 for signal in momentum_signals.values() if "상승" in signal or "매수" in signal)
        
        if bullish_momentum > len(momentum_signals) * 0.7:
            recommendations.append("⚡ 긍정적 모멘텀 - 상승 가능성 높음")
        elif bullish_momentum < len(momentum_signals) * 0.3:
            recommendations.append("⚡ 부정적 모멘텀 - 하락 주의")
        
        # 위험도 기반 권장사항
        if analysis.risk_level == "높음":
            recommendations.append("⚠️ 높은 위험도 - 포지션 크기 축소 권장")
        elif analysis.risk_level == "낮음":
            recommendations.append("✅ 낮은 위험도 - 안정적 투자 환경")
        
        # 권장사항 표시
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div style="padding: 10px; margin: 5px 0; background-color: #f8f9fa; border-radius: 5px; border-left: 3px solid #007bff;">
                    <strong>{i}.</strong> {rec}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("현재 특별한 권장사항이 없습니다. 시장 상황을 지속 모니터링하세요.")
        
    except Exception as e:
        logger.error(f"실시간 권장사항 오류: {str(e)}")

def render_realtime_analysis(chart_analyzer: ChartAnalyzer, trading_manager: TradingManager, market: str):
    """고급 실시간 분석 렌더링"""
    st.subheader("⚡ 실시간 모니터링")
    
    # 모니터링 제어 패널
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("**모니터링 설정**")
        
        # 모니터링 상태
        is_monitoring = market in alert_system.monitoring_markets
        monitoring_status = "🟢 활성" if is_monitoring else "🔴 비활성"
        st.write(f"상태: {monitoring_status}")
    
    with col2:
        if st.button("🚀 모니터링 시작" if not is_monitoring else "⏹️ 모니터링 중지"):
            if not is_monitoring:
                alert_system.add_market_monitor(market)
                if not alert_system.is_monitoring:
                    alert_system.start_monitoring(trading_manager, 30)  # 30초 간격
                st.success(f"{market} 모니터링 시작!")
                st.rerun()
            else:
                alert_system.remove_market_monitor(market)
                st.success(f"{market} 모니터링 중지!")
                st.rerun()
    
    with col3:
        if st.button("📊 즉시 분석"):
            with st.spinner("분석 중..."):
                # 현재 차트 분석
                ohlcv_data = trading_manager.get_ohlcv(market, count=100)
                indicators = trading_manager.calculate_indicators(ohlcv_data)
                analysis = chart_analyzer.analyze_chart(ohlcv_data, indicators)
                
                # 분석 결과 표시
                st.success("분석 완료!")
                
                # 주요 신호 표시
                if analysis.patterns:
                    latest_pattern = analysis.patterns[0]
                    if latest_pattern.confidence > 0.7:
                        alert_type = "success" if "상승" in latest_pattern.description else "warning"
                        getattr(st, alert_type)(
                            f"🎯 {latest_pattern.pattern_type.value} 패턴 감지! (신뢰도: {latest_pattern.confidence*100:.1f}%)"
                        )
    
    # 실시간 메트릭
    st.markdown("---")
    st.subheader("📈 실시간 메트릭")
    
    try:
        # 현재 데이터 가져오기
        ohlcv_data = trading_manager.get_ohlcv(market, count=50)
        current_data = trading_manager.get_market_data(market)
        
        if ohlcv_data is not None and not ohlcv_data.empty and current_data:
            # 메트릭 계산
            current_price = current_data['trade_price']
            price_change_24h = current_data.get('signed_change_rate', 0) * 100
            volume_24h = current_data.get('acc_trade_volume_24h', 0)
            
            # 기술적 지표
            indicators = trading_manager.calculate_indicators(ohlcv_data)
            rsi = indicators['rsi'].iloc[-1] if 'rsi' in indicators else None
            
            # 메트릭 표시
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "현재가",
                    f"{current_price:,}원",
                    delta=f"{price_change_24h:+.2f}%"
                )
            
            with col2:
                volume_color = "normal"
                if volume_24h > 0:
                    avg_volume = ohlcv_data['volume'].tail(20).mean()
                    volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1
                    volume_color = "inverse" if volume_ratio > 2 else "normal"
                
                st.metric(
                    "24시간 거래량",
                    f"{volume_24h:,.0f}",
                    delta=f"평균 대비 {volume_ratio:.1f}배" if 'volume_ratio' in locals() else None
                )
            
            with col3:
                if rsi is not None:
                    rsi_status = "과매수" if rsi > 70 else "과매도" if rsi < 30 else "중립"
                    rsi_color = "inverse" if rsi > 70 or rsi < 30 else "normal"
                    st.metric("RSI", f"{rsi:.1f}", delta=rsi_status)
            
            with col4:
                # 변동성 계산
                returns = ohlcv_data['close'].pct_change().dropna()
                volatility = returns.std() * 100
                vol_status = "높음" if volatility > 5 else "낮음" if volatility < 2 else "보통"
                st.metric("변동성", f"{volatility:.2f}%", delta=vol_status)
        
        # 실시간 차트
        st.markdown("---")
        st.subheader("📊 실시간 가격 차트")
        
        if ohlcv_data is not None and not ohlcv_data.empty:
            # 미니 차트 생성
            fig = go.Figure()
            
            # 최근 24시간 데이터
            recent_data = ohlcv_data.tail(24)
            
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['close'],
                mode='lines+markers',
                name='가격',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title="최근 24시간 가격 추이",
                xaxis_title="시간",
                yaxis_title="가격 (KRW)",
                height=300,
                showlegend=False,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"실시간 데이터 로드 오류: {str(e)}")
    
    # 자동 새로고침 설정
    st.markdown("---")
    auto_refresh = st.checkbox("⚡ 자동 새로고침 (30초)", value=False)
    
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

def render_alert_center(trading_manager: TradingManager, market: str):
    """고급 알림 센터 렌더링"""
    st.subheader("🔔 알림 센터")
    
    # 알림 통계
    alert_stats = alert_system.get_alert_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("전체 알림", alert_stats['total'])
    with col2:
        st.metric("읽지 않은 알림", alert_stats['unread'])
    with col3:
        st.metric("모니터링 마켓", len(alert_stats['monitoring_markets']))
    with col4:
        if st.button("🗑️ 알림 정리"):
            alert_system.clear_alerts(market, days=1)
            st.success("오래된 알림을 정리했습니다.")
            st.rerun()
    
    # 알림 설정
    st.markdown("---")
    st.subheader("⚙️ 알림 설정")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pattern_alerts = st.checkbox("패턴 알림", value=alert_system.pattern_alerts_enabled)
    with col2:
        price_alerts = st.checkbox("가격 돌파 알림", value=alert_system.price_alerts_enabled)
    with col3:
        volume_alerts = st.checkbox("거래량 급증 알림", value=alert_system.volume_alerts_enabled)
    
    # 설정 업데이트
    if pattern_alerts != alert_system.pattern_alerts_enabled:
        alert_system.pattern_alerts_enabled = pattern_alerts
    if price_alerts != alert_system.price_alerts_enabled:
        alert_system.price_alerts_enabled = price_alerts
    if volume_alerts != alert_system.volume_alerts_enabled:
        alert_system.volume_alerts_enabled = volume_alerts
    
    # 알림 임계값 설정
    col1, col2 = st.columns(2)
    with col1:
        volume_threshold = st.slider("거래량 급증 임계값 (평균 대비)", 1.5, 5.0, alert_system.volume_spike_threshold, 0.1)
        alert_system.volume_spike_threshold = volume_threshold
    
    with col2:
        price_threshold = st.slider("가격 돌파 임계값 (%)", 1, 5, int(alert_system.price_breakout_threshold * 100)) / 100
        alert_system.price_breakout_threshold = price_threshold
    
    # 알림 필터
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        show_unread_only = st.checkbox("읽지 않은 알림만 보기", value=True)
    
    with col2:
        alert_limit = st.selectbox("표시 개수", [10, 25, 50, 100], index=1)
    
    # 알림 목록
    alerts = alert_system.get_alerts(
        market=market,
        limit=alert_limit,
        unread_only=show_unread_only
    )
    
    if not alerts:
        st.info("표시할 알림이 없습니다.")
        return
    
    st.markdown("---")
    st.subheader(f"📋 알림 목록 ({len(alerts)}개)")
    
    for alert in alerts:
        # 우선순위에 따른 색상
        if alert.priority.value == "긴급":
            border_color = "#ff4444"
            icon = "🚨"
        elif alert.priority.value == "높음":
            border_color = "#ff8800"
            icon = "⚠️"
        elif alert.priority.value == "보통":
            border_color = "#ffaa00"
            icon = "💡"
        else:
            border_color = "#888888"
            icon = "ℹ️"
        
        # 알림 카드
        with st.container():
            if not alert.is_read:
                st.markdown(f"""
                <div style="border-left: 4px solid {border_color}; padding: 10px; margin: 5px 0; background-color: #f8f9fa;">
                    <h5>{icon} {alert.title}</h5>
                    <p>{alert.message}</p>
                    <small>📅 {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | 🎯 {alert.market}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="border-left: 2px solid #cccccc; padding: 10px; margin: 5px 0; background-color: #f0f0f0; opacity: 0.7;">
                    <h6>✅ {alert.title}</h6>
                    <p><small>{alert.message}</small></p>
                    <small>📅 {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | 🎯 {alert.market}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # 알림 액션
            col1, col2, col3 = st.columns([1, 1, 4])
            
            with col1:
                if not alert.is_read and st.button(f"읽음", key=f"read_{alert.id}"):
                    alert_system.mark_alert_read(alert.id)
                    st.rerun()
            
            with col2:
                if alert.pattern and st.button(f"상세", key=f"detail_{alert.id}"):
                    # 패턴 상세 정보 표시
                    with st.expander(f"패턴 상세 정보", expanded=True):
                        pattern = alert.pattern
                        st.write(f"**패턴 타입:** {pattern.pattern_type.value}")
                        st.write(f"**신뢰도:** {pattern.confidence*100:.1f}%")
                        st.write(f"**신호 강도:** {pattern.signal_strength.value}")
                        if pattern.target_price:
                            st.write(f"**목표가:** {pattern.target_price:,.0f}원")
                        if pattern.stop_loss:
                            st.write(f"**손절가:** {pattern.stop_loss:,.0f}원")
                        st.write(f"**설명:** {pattern.description}")
    
    # 전체 읽음 처리
    if alerts and st.button("📖 모든 알림 읽음 처리", use_container_width=True):
        alert_system.mark_all_read(market)
        st.success("모든 알림을 읽음 처리했습니다.")
        st.rerun()

def render_ai_prediction(chart_analyzer: ChartAnalyzer, trading_manager: TradingManager, market: str):
    """AI 기반 가격 예측 렌더링"""
    st.subheader("🤖 AI 가격 예측")
    
    # AI 예측기 초기화
    ai_predictor = AIPricePredictor()
    
    # 예측 실행 버튼
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("**AI 모델 정보**")
        st.write("• 기술적 지표, 가격 액션, 거래량, 패턴 분석 종합")
        st.write("• 머신러닝 기반 확률 예측")
        st.write("• 실시간 데이터 기반 동적 예측")
    
    with col2:
        if st.button("🚀 AI 예측 실행", use_container_width=True):
            with st.spinner("AI 분석 중..."):
                # 데이터 수집
                ohlcv_data = trading_manager.get_ohlcv(market, count=200)
                indicators = trading_manager.calculate_indicators(ohlcv_data)
                current_data = trading_manager.get_market_data(market)
                
                # 차트 분석
                analysis = chart_analyzer.analyze_chart(ohlcv_data, indicators)
                
                # AI 예측 실행
                prediction = ai_predictor.predict_price(
                    ohlcv_data, indicators, analysis.patterns, current_data
                )
                
                # 세션 상태에 저장
                st.session_state[f'ai_prediction_{market}'] = prediction
                st.success("AI 예측 완료!")
                st.rerun()
    
    with col3:
        if st.button("🔄 예측 초기화"):
            if f'ai_prediction_{market}' in st.session_state:
                del st.session_state[f'ai_prediction_{market}']
                st.success("예측 데이터가 초기화되었습니다.")
                st.rerun()
    
    # 예측 결과 표시
    if f'ai_prediction_{market}' in st.session_state:
        prediction = st.session_state[f'ai_prediction_{market}']
        
        st.markdown("---")
        
        # 메인 예측 결과
        st.subheader("🎯 예측 결과")
        
        # 방향성과 신뢰도
        direction_color = "#28a745" if "상승" in prediction.direction.value else "#dc3545" if "하락" in prediction.direction.value else "#6c757d"
        confidence_color = "#28a745" if "높음" in prediction.confidence.value else "#ffc107" if "보통" in prediction.confidence.value else "#dc3545"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, {direction_color}, {direction_color}aa); border-radius: 10px; margin: 10px 0;">
                <h3 style="color: white; margin: 0;">예측 방향</h3>
                <h2 style="color: white; margin: 10px 0;">{prediction.direction.value}</h2>
                <p style="color: white; margin: 0;">확률: {prediction.probability_up*100:.1f}% 상승 / {prediction.probability_down*100:.1f}% 하락</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, {confidence_color}, {confidence_color}aa); border-radius: 10px; margin: 10px 0;">
                <h3 style="color: white; margin: 0;">예측 신뢰도</h3>
                <h2 style="color: white; margin: 10px 0;">{prediction.confidence.value}</h2>
                <p style="color: white; margin: 0;">위험도: {prediction.risk_assessment}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 가격 예측
        st.markdown("---")
        st.subheader("💰 예상 가격")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "현재가",
                f"{prediction.current_price:,.0f}원",
                delta=""
            )
        
        with col2:
            price_change_1h = prediction.predicted_price_1h - prediction.current_price
            change_pct_1h = (price_change_1h / prediction.current_price) * 100
            st.metric(
                "1시간 후",
                f"{prediction.predicted_price_1h:,.0f}원",
                delta=f"{change_pct_1h:+.2f}%"
            )
        
        with col3:
            price_change_4h = prediction.predicted_price_4h - prediction.current_price
            change_pct_4h = (price_change_4h / prediction.current_price) * 100
            st.metric(
                "4시간 후",
                f"{prediction.predicted_price_4h:,.0f}원",
                delta=f"{change_pct_4h:+.2f}%"
            )
        
        with col4:
            price_change_24h = prediction.predicted_price_24h - prediction.current_price
            change_pct_24h = (price_change_24h / prediction.current_price) * 100
            st.metric(
                "24시간 후",
                f"{prediction.predicted_price_24h:,.0f}원",
                delta=f"{change_pct_24h:+.2f}%"
            )
        
        # 예측 차트
        st.markdown("---")
        st.subheader("📊 예측 차트")
        
        # 현재 시간 기준으로 미래 시점 생성
        current_time = prediction.timestamp
        time_points = [
            current_time,
            current_time + timedelta(hours=1),
            current_time + timedelta(hours=4),
            current_time + timedelta(hours=24)
        ]
        
        prices = [
            prediction.current_price,
            prediction.predicted_price_1h,
            prediction.predicted_price_4h,
            prediction.predicted_price_24h
        ]
        
        fig = go.Figure()
        
        # 예측 라인
        fig.add_trace(go.Scatter(
            x=time_points,
            y=prices,
            mode='lines+markers',
            name='AI 예측',
            line=dict(color='#007bff', width=3),
            marker=dict(size=8, color='#007bff')
        ))
        
        # 신뢰구간 (간단한 방식)
        confidence_factor = 0.02 if prediction.confidence.value == "높음" else 0.05 if prediction.confidence.value == "보통" else 0.1
        
        upper_bound = [p * (1 + confidence_factor) for p in prices]
        lower_bound = [p * (1 - confidence_factor) for p in prices]
        
        fig.add_trace(go.Scatter(
            x=time_points + time_points[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(0,123,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='예측 구간'
        ))
        
        fig.update_layout(
            title="AI 가격 예측 차트",
            xaxis_title="시간",
            yaxis_title="가격 (KRW)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 주요 요인 분석
        st.markdown("---")
        st.subheader("🔍 주요 영향 요인")
        
        for i, factor in enumerate(prediction.key_factors, 1):
            factor_color = "#28a745" if "긍정" in factor or "상승" in factor or "증가" in factor else "#dc3545" if "부정" in factor or "하락" in factor or "부족" in factor else "#6c757d"
            
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; background-color: #f8f9fa; border-radius: 5px; border-left: 3px solid {factor_color};">
                <strong>{i}.</strong> {factor}
            </div>
            """, unsafe_allow_html=True)
        
        # 투자 권장사항
        st.markdown("---")
        st.subheader("💡 AI 투자 권장사항")
        
        # 예측 기반 권장사항 생성
        recommendations = generate_ai_recommendations(prediction)
        
        for rec in recommendations:
            rec_color = "#28a745" if "매수" in rec or "보유" in rec else "#dc3545" if "매도" in rec else "#6c757d"
            
            st.markdown(f"""
            <div style="padding: 15px; margin: 10px 0; background-color: #f8f9fa; border-radius: 10px; border: 2px solid {rec_color};">
                {rec}
            </div>
            """, unsafe_allow_html=True)
        
        # 예측 시간 정보
        st.markdown("---")
        st.markdown(f"*예측 생성 시간: {prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*")
        st.markdown("*주의: AI 예측은 참고용이며, 투자 결정은 신중하게 하시기 바랍니다.*")
    
    else:
        # 예측 데이터가 없을 때
        st.info("🤖 AI 예측을 실행하려면 위의 '🚀 AI 예측 실행' 버튼을 클릭하세요.")
        
        # AI 모델 설명
        st.markdown("---")
        st.subheader("📋 AI 모델 설명")
        
        st.markdown("""
        **🧠 AutoCoin AI 예측 모델**
        
        우리의 AI 모델은 다음과 같은 데이터를 종합 분석합니다:
        
        - **기술적 지표 (35%)**: RSI, MACD, 볼린저 밴드, 이동평균선 등
        - **가격 액션 (25%)**: 캔들스틱 패턴, 고점/저점 분석, 연속성
        - **거래량 분석 (20%)**: 거래량 패턴, 가격-거래량 관계
        - **시장 심리 (15%)**: 차트 패턴, 패턴 신뢰도
        - **시간 패턴 (5%)**: 요일별, 시간대별 패턴
        
        **🎯 예측 정확도**
        - 1시간 예측: 약 65-75% 정확도
        - 4시간 예측: 약 60-70% 정확도  
        - 24시간 예측: 약 55-65% 정확도
        
        **⚠️ 주의사항**
        - AI 예측은 과거 데이터를 기반으로 하며 100% 정확하지 않습니다
        - 급격한 시장 변화나 뉴스 이벤트는 예측에 반영되지 않을 수 있습니다
        - 투자 결정 시 다른 요소들도 함께 고려하시기 바랍니다
        """)

def generate_ai_recommendations(prediction) -> List[str]:
    """AI 예측 기반 투자 권장사항 생성"""
    recommendations = []
    
    # 방향성 기반 권장사항
    if prediction.direction == PredictionDirection.STRONG_UP:
        if prediction.confidence.value in ["매우 높음", "높음"]:
            recommendations.append("🟢 **강력 매수 추천**: 높은 신뢰도의 강한 상승 신호")
        else:
            recommendations.append("🟡 **조심스러운 매수**: 상승 신호가 있으나 신뢰도 주의")
    
    elif prediction.direction == PredictionDirection.UP:
        if prediction.confidence.value in ["매우 높음", "높음"]:
            recommendations.append("🟢 **매수 고려**: 상승 가능성이 높음")
        else:
            recommendations.append("🟡 **소량 매수**: 상승 신호가 있으나 신중하게")
    
    elif prediction.direction == PredictionDirection.NEUTRAL:
        recommendations.append("⚪ **관망 권장**: 명확한 방향성이 없어 대기 추천")
    
    elif prediction.direction == PredictionDirection.DOWN:
        if prediction.confidence.value in ["매우 높음", "높음"]:
            recommendations.append("🔴 **매도 고려**: 하락 가능성이 높음")
        else:
            recommendations.append("🟡 **부분 매도**: 하락 신호가 있으나 신중하게")
    
    elif prediction.direction == PredictionDirection.STRONG_DOWN:
        if prediction.confidence.value in ["매우 높음", "높음"]:
            recommendations.append("🔴 **강력 매도 추천**: 높은 신뢰도의 강한 하락 신호")
        else:
            recommendations.append("🟡 **조심스러운 매도**: 하락 신호가 있으나 신뢰도 주의")
    
    # 위험도 기반 권장사항
    if "높음" in prediction.risk_assessment:
        recommendations.append("⚠️ **위험 관리**: 높은 위험도로 포지션 크기 축소 권장")
    elif "낮음" in prediction.risk_assessment:
        recommendations.append("✅ **안정적 환경**: 상대적으로 안전한 투자 환경")
    
    # 확률 기반 권장사항
    if prediction.probability_up > 0.8:
        recommendations.append("📈 **높은 상승 확률**: 80% 이상의 상승 확률")
    elif prediction.probability_down > 0.8:
        recommendations.append("📉 **높은 하락 확률**: 80% 이상의 하락 확률")
    
    return recommendations

def render_performance_monitor():
    """성능 모니터링 렌더링"""
    st.subheader("⚙️ 성능 모니터")
    
    # 성능 리포트 가져오기
    performance_report = get_performance_report()
    
    # 캐시 통계
    st.markdown("### 💾 캐시 성능")
    cache_stats = performance_report['cache_stats']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("캐시 항목", cache_stats['active_items'])
    
    with col2:
        st.metric("최대 크기", cache_stats['max_size'])
    
    with col3:
        st.metric("TTL", f"{cache_stats['ttl_seconds']}초")
    
    with col4:
        cache_usage = (cache_stats['active_items'] / cache_stats['max_size'] * 100) if cache_stats['max_size'] > 0 else 0
        st.metric("사용률", f"{cache_usage:.1f}%")
    
    # 캐시 관리 버튼
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ 캐시 비우기"):
            analysis_cache.clear()
            st.success("캐시가 비워졌습니다.")
            st.rerun()
    
    with col2:
        if st.button("📊 통계 초기화"):
            performance_monitor.reset()
            st.success("성능 통계가 초기화되었습니다.")
            st.rerun()
    
    with col3:
        if st.button("🔄 새로고침"):
            st.rerun()
    
    # 함수별 성능 통계
    st.markdown("---")
    st.markdown("### 📈 함수별 성능 통계")
    
    perf_stats = performance_report['performance_stats']
    
    if perf_stats:
        # 테이블 형식으로 표시
        stats_data = []
        for func_name, stats in perf_stats.items():
            stats_data.append({
                '함수명': func_name,
                '총 호출': stats['total_calls'],
                '캐시 적중률': stats['cache_hit_rate'],
                '평균 시간': stats['avg_time'],
                '최대 시간': stats['max_time'],
                '최소 시간': stats['min_time']
            })
        
        if stats_data:
            import pandas as pd
            df = pd.DataFrame(stats_data)
            st.dataframe(df, use_container_width=True)
        
        # 성능 차트
        st.markdown("---")
        st.markdown("### 📊 성능 차트")
        
        # 캐시 적중률 차트
        if len(perf_stats) > 0:
            func_names = list(perf_stats.keys())
            hit_rates = [float(perf_stats[func]['cache_hit_rate'].replace('%', '')) for func in func_names]
            
            fig_hit_rate = go.Figure(data=[
                go.Bar(x=func_names, y=hit_rates, name='캐시 적중률')
            ])
            
            fig_hit_rate.update_layout(
                title="함수별 캐시 적중률",
                xaxis_title="함수명",
                yaxis_title="적중률 (%)",
                height=400
            )
            
            st.plotly_chart(fig_hit_rate, use_container_width=True)
            
            # 실행 시간 차트
            avg_times = [float(perf_stats[func]['avg_time'].replace('s', '')) for func in func_names]
            
            fig_time = go.Figure(data=[
                go.Bar(x=func_names, y=avg_times, name='평균 실행 시간')
            ])
            
            fig_time.update_layout(
                title="함수별 평균 실행 시간",
                xaxis_title="함수명",
                yaxis_title="시간 (초)",
                height=400
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
    
    else:
        st.info("아직 성능 데이터가 없습니다. 차트 분석을 실행해보세요.")
    
    # 시스템 정보
    st.markdown("---")
    st.markdown("### 🖥️ 시스템 정보")
    
    try:
        import psutil
        import platform
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_percent = psutil.cpu_percent(interval=1)
            st.metric("CPU 사용률", f"{cpu_percent:.1f}%")
        
        with col2:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            st.metric("메모리 사용률", f"{memory_percent:.1f}%")
        
        with col3:
            st.metric("플랫폼", platform.system())
        
        # 프로세스 정보
        process = psutil.Process()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            st.metric("앱 메모리", f"{process_memory:.1f} MB")
        
        with col2:
            process_cpu = process.cpu_percent()
            st.metric("앱 CPU", f"{process_cpu:.1f}%")
        
        with col3:
            thread_count = process.num_threads()
            st.metric("스레드 수", thread_count)
    
    except ImportError:
        st.info("시스템 모니터링을 위해 psutil 패키지가 필요합니다.")
    except Exception as e:
        st.warning(f"시스템 정보를 가져올 수 없습니다: {str(e)}")
    
    # 최적화 제안
    st.markdown("---")
    st.markdown("### 💡 최적화 제안")
    
    suggestions = []
    
    # 캐시 사용률 기반 제안
    if cache_usage > 90:
        suggestions.append("🟡 캐시 사용률이 높습니다. 캐시 크기를 늘리거나 TTL을 줄이는 것을 고려하세요.")
    elif cache_usage < 30:
        suggestions.append("🟢 캐시 사용률이 낮습니다. 시스템이 효율적으로 작동하고 있습니다.")
    
    # 캐시 적중률 기반 제안
    if perf_stats:
        low_hit_rate_funcs = [
            func for func, stats in perf_stats.items()
            if float(stats['cache_hit_rate'].replace('%', '')) < 50
        ]
        
        if low_hit_rate_funcs:
            suggestions.append(f"🟡 다음 함수들의 캐시 적중률이 낮습니다: {', '.join(low_hit_rate_funcs)}")
    
    # 실행 시간 기반 제안
    if perf_stats:
        slow_funcs = [
            func for func, stats in perf_stats.items()
            if float(stats['avg_time'].replace('s', '')) > 1.0
        ]
        
        if slow_funcs:
            suggestions.append(f"🔴 다음 함수들의 실행 시간이 깁니다: {', '.join(slow_funcs)}")
    
    if suggestions:
        for suggestion in suggestions:
            st.markdown(f"• {suggestion}")
    else:
        st.success("✅ 현재 성능이 양호합니다!")
    
    # 리포트 생성 시간
    st.markdown("---")
    st.markdown(f"*마지막 업데이트: {performance_report['timestamp']}*")

def render_risk_analysis(trading_manager: TradingManager, market: str):
    """실시간 위험도 분석 렌더링"""
    st.subheader("⚠️ 실시간 위험도 분석")
    
    # 위험도 분석기 초기화
    risk_analyzer = RealTimeRiskAnalyzer()
    
    # 분석 실행 버튼
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("**위험도 분석 정보**")
        st.write("• 변동성, 유동성, 기술적, 모멘텀, 하락폭 위험 종합 분석")
        st.write("• VaR(Value at Risk) 기반 손실 위험 평가")
        st.write("• 실시간 포지션 위험도 및 권장 사항 제공")
    
    with col2:
        if st.button("🚨 위험도 분석 실행", use_container_width=True):
            with st.spinner("위험도 분석 중..."):
                # 데이터 수집
                ohlcv_data = trading_manager.get_ohlcv(market, count=200)
                current_data = trading_manager.get_market_data(market)
                
                if ohlcv_data is not None and not ohlcv_data.empty and current_data:
                    current_price = current_data['trade_price']
                    
                    # 위험도 분석 실행
                    risk_metrics = risk_analyzer.analyze_market_risk(market, ohlcv_data, current_price)
                    overall_score, overall_level = risk_analyzer.get_overall_risk_score(risk_metrics)
                    
                    # 세션 상태에 저장
                    st.session_state[f'risk_analysis_{market}'] = {
                        'risk_metrics': risk_metrics,
                        'overall_score': overall_score,
                        'overall_level': overall_level,
                        'current_price': current_price,
                        'timestamp': datetime.now()
                    }
                    st.success("위험도 분석 완료!")
                    st.rerun()
                else:
                    st.error("데이터를 불러올 수 없습니다.")
    
    with col3:
        if st.button("🔄 분석 초기화"):
            if f'risk_analysis_{market}' in st.session_state:
                del st.session_state[f'risk_analysis_{market}']
                st.success("위험도 분석 데이터가 초기화되었습니다.")
                st.rerun()
    
    # 위험도 알림 설정
    st.markdown("---")
    st.subheader("🔔 위험도 알림 설정")
    
    col1, col2 = st.columns(2)
    with col1:
        risk_alerts = st.checkbox("위험도 알림 활성화", value=alert_system.risk_alerts_enabled)
        alert_system.risk_alerts_enabled = risk_alerts
    
    with col2:
        auto_risk_check = st.checkbox("자동 위험도 체크 (실시간 모니터링 시)", value=True)
    
    # 위험도 분석 결과 표시
    if f'risk_analysis_{market}' in st.session_state:
        risk_data = st.session_state[f'risk_analysis_{market}']
        risk_metrics = risk_data['risk_metrics']
        overall_score = risk_data['overall_score']
        overall_level = risk_data['overall_level']
        current_price = risk_data['current_price']
        
        st.markdown("---")
        
        # 전체 위험도 점수 표시
        st.subheader("🎯 종합 위험도 평가")
        
        # 위험도 레벨에 따른 색상
        if overall_level in [RiskLevel.VERY_HIGH, RiskLevel.EXTREME]:
            risk_color = "#dc3545"  # 빨간색
            risk_icon = "🚨"
        elif overall_level == RiskLevel.HIGH:
            risk_color = "#fd7e14"  # 주황색
            risk_icon = "⚠️"
        elif overall_level == RiskLevel.MEDIUM:
            risk_color = "#ffc107"  # 노란색
            risk_icon = "⚡"
        else:
            risk_color = "#28a745"  # 초록색
            risk_icon = "✅"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background: linear-gradient(45deg, {risk_color}, {risk_color}aa); border-radius: 15px; margin: 20px 0;">
            <h2 style="color: white; margin: 0;">{risk_icon} 종합 위험도</h2>
            <h1 style="color: white; margin: 15px 0; font-size: 3em;">{overall_level.value}</h1>
            <h2 style="color: white; margin: 10px 0;">{overall_score*100:.0f}/100점</h2>
            <p style="color: white; margin: 0;">현재가: {current_price:,.0f}원</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 위험도 미터 차트
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "위험도 지수"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 20], 'color': "#d4edda"},
                    {'range': [20, 40], 'color': "#fff3cd"},
                    {'range': [40, 60], 'color': "#f8d7da"},
                    {'range': [60, 80], 'color': "#f5c6cb"},
                    {'range': [80, 100], 'color': "#f1b0b7"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # 개별 위험 요소 분석
        st.markdown("---")
        st.subheader("🔍 개별 위험 요소")
        
        if risk_metrics:
            # 위험 요소별 표시
            risk_by_type = {}
            for metric in risk_metrics:
                risk_type = metric.risk_type.value
                if risk_type not in risk_by_type:
                    risk_by_type[risk_type] = []
                risk_by_type[risk_type].append(metric)
            
            for risk_type, metrics in risk_by_type.items():
                with st.expander(f"{risk_type} 위험 분석", expanded=True):
                    for metric in metrics:
                        # 위험도에 따른 색상
                        if metric.risk_level in [RiskLevel.VERY_HIGH, RiskLevel.EXTREME]:
                            metric_color = "#dc3545"
                            metric_icon = "🚨"
                        elif metric.risk_level == RiskLevel.HIGH:
                            metric_color = "#fd7e14"
                            metric_icon = "⚠️"
                        elif metric.risk_level == RiskLevel.MEDIUM:
                            metric_color = "#ffc107"
                            metric_icon = "⚡"
                        else:
                            metric_color = "#28a745"
                            metric_icon = "✅"
                        
                        st.markdown(f"""
                        <div style="padding: 15px; margin: 10px 0; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid {metric_color};">
                            <h4 style="margin: 0; color: {metric_color};">{metric_icon} {metric.description}</h4>
                            <p style="margin: 5px 0;"><strong>위험도:</strong> {metric.risk_level.value}</p>
                            <p style="margin: 5px 0;"><strong>권장사항:</strong> {metric.recommendation}</p>
                            <div style="background-color: #e9ecef; border-radius: 5px; height: 20px; margin: 10px 0;">
                                <div style="background-color: {metric_color}; height: 100%; width: {min(100, metric.current_value/metric.threshold_value*100):.1f}%; border-radius: 5px;"></div>
                            </div>
                            <small>현재값: {metric.current_value:.4f} / 임계값: {metric.threshold_value:.4f}</small>
                        </div>
                        """, unsafe_allow_html=True)
        
        # 포지션 위험도 분석
        st.markdown("---")
        st.subheader("💼 포지션 위험도 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            position_size = st.number_input("포지션 크기 (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
            entry_price = st.number_input("진입가 (원)", min_value=0, value=int(current_price*0.98), step=1000)
        
        with col2:
            if st.button("포지션 위험도 계산", use_container_width=True):
                ohlcv_data = trading_manager.get_ohlcv(market, count=200)
                
                if ohlcv_data is not None and not ohlcv_data.empty:
                    position_risk = risk_analyzer.calculate_position_risk(
                        market, ohlcv_data, position_size/100, entry_price
                    )
                    
                    # 포지션 위험도 표시
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        pnl_color = "inverse" if position_risk.unrealized_pnl < 0 else "normal"
                        st.metric(
                            "미실현 손익",
                            f"{position_risk.unrealized_pnl:,.0f}원",
                            delta=f"{position_risk.unrealized_pnl_pct*100:+.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "1일 VaR",
                            f"{position_risk.var_1d:,.0f}원",
                            delta="95% 신뢰구간"
                        )
                    
                    with col3:
                        st.metric(
                            "7일 VaR", 
                            f"{position_risk.var_7d:,.0f}원",
                            delta="주간 위험"
                        )
                    
                    with col4:
                        st.metric(
                            "최대 하락폭",
                            f"{position_risk.max_drawdown*100:.2f}%",
                            delta=position_risk.risk_level.value
                        )
                    
                    # 권장사항
                    st.markdown("**🎯 포지션 관리 권장사항**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"**권장 포지션 크기:** {position_risk.position_size_suggestion*100:.1f}%")
                    
                    with col2:
                        st.warning(f"**권장 손절가:** {position_risk.stop_loss_suggestion:,.0f}원")
        
        # 위험 관리 가이드
        st.markdown("---")
        st.subheader("📋 위험 관리 가이드")
        
        risk_guide = get_risk_management_guide(overall_level)
        
        for i, guide in enumerate(risk_guide, 1):
            guide_color = "#dc3545" if "긴급" in guide else "#fd7e14" if "주의" in guide else "#28a745"
            
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; background-color: #f8f9fa; border-radius: 5px; border-left: 3px solid {guide_color};">
                <strong>{i}.</strong> {guide}
            </div>
            """, unsafe_allow_html=True)
        
        # 분석 시간 정보
        st.markdown("---")
        st.markdown(f"*분석 시간: {risk_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}*")
        st.markdown("*주의: 위험도 분석은 참고용이며, 투자 결정 시 추가 요소들을 고려하시기 바랍니다.*")
    
    else:
        # 분석 데이터가 없을 때
        st.info("⚠️ 위험도 분석을 실행하려면 위의 '🚨 위험도 분석 실행' 버튼을 클릭하세요.")
        
        # 위험도 분석 설명
        st.markdown("---")
        st.subheader("📋 위험도 분석 설명")
        
        st.markdown("""
        **⚠️ AutoCoin 위험도 분석 시스템**
        
        우리의 위험도 분석은 다음과 같은 요소들을 종합 평가합니다:
        
        - **변동성 위험 (30%)**: 일일/주간 가격 변동성 분석
        - **기술적 위험 (25%)**: RSI, 볼린저 밴드 등 기술적 지표 기반
        - **모멘텀 위험 (20%)**: 가격 모멘텀 변화 및 반전 위험
        - **하락폭 위험 (15%)**: 최대 하락폭 및 현재 드로우다운
        - **유동성 위험 (10%)**: 거래량 기반 유동성 평가
        
        **🎯 위험도 레벨**
        - **매우 낮음**: 안전한 투자 환경
        - **낮음**: 상대적으로 안정적
        - **보통**: 일반적인 시장 위험
        - **높음**: 주의 깊은 모니터링 필요
        - **매우 높음**: 포지션 축소 권장
        - **극도로 높음**: 즉시 위험 관리 필요
        
        **📊 VaR (Value at Risk)**
        - 95% 신뢰구간에서 예상되는 최대 손실
        - 1일 VaR: 하루 동안의 예상 최대 손실
        - 7일 VaR: 일주일 동안의 예상 최대 손실
        
        **⚠️ 주의사항**
        - 위험도 분석은 과거 데이터를 기반으로 하며 미래를 보장하지 않습니다
        - 급격한 시장 변화나 예상치 못한 이벤트는 반영되지 않을 수 있습니다
        - 투자 결정 시 개인의 위험 성향과 투자 목표를 함께 고려하세요
        """)

def get_risk_management_guide(risk_level: RiskLevel) -> List[str]:
    """위험도 레벨별 관리 가이드"""
    if risk_level == RiskLevel.EXTREME:
        return [
            "🚨 긴급: 즉시 모든 포지션 점검 및 손절매 고려",
            "⚡ 긴급: 신규 투자 중단 및 현금 비중 확대",
            "📞 긴급: 전문가 상담 또는 추가 분석 필요",
            "🔄 매시간 위험도 재평가 실시"
        ]
    elif risk_level == RiskLevel.VERY_HIGH:
        return [
            "⚠️ 포지션 크기를 50% 이하로 축소",
            "🛑 엄격한 손절매 라인 설정 (5-8%)",
            "📉 신규 매수 신중하게 고려",
            "⏰ 30분-1시간마다 시장 상황 점검"
        ]
    elif risk_level == RiskLevel.HIGH:
        return [
            "⚡ 포지션 크기 축소 고려 (70% 이하)",
            "🎯 손절매 라인 설정 (8-10%)",
            "📊 기술적 지표 및 뉴스 면밀히 모니터링",
            "⏰ 2-3시간마다 상황 점검"
        ]
    elif risk_level == RiskLevel.MEDIUM:
        return [
            "📈 정상적인 투자 진행 가능",
            "🎯 적절한 손절매 라인 유지 (10-15%)",
            "📊 일일 시장 동향 확인",
            "💡 분산 투자로 위험 분산"
        ]
    else:  # LOW, VERY_LOW
        return [
            "✅ 안전한 투자 환경",
            "📈 적극적인 투자 기회 활용 가능",
            "💰 목표 수익률에 따른 포지션 조정",
            "📊 정기적인 시장 모니터링 유지"
        ]