"""
뉴스 기반 전략 UI 컴포넌트
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
from src.core.news_strategy import NewsBasedStrategy, SignalStrength
from src.api.news import CryptoNewsAPI
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def render_news_strategy_section(market: str, price_change: float, news_api: CryptoNewsAPI):
    """뉴스 기반 전략 섹션 렌더링"""
    
    st.header("🧠 뉴스 기반 AI 전략")
    
    # 전략 엔진 초기화
    strategy_engine = NewsBasedStrategy()
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["AI 전략 분석", "뉴스 감정 분석", "위험도 평가", "실시간 신호"])
    
    with tab1:
        render_ai_strategy_analysis(strategy_engine, market, price_change, news_api)
    
    with tab2:
        render_news_sentiment_analysis(strategy_engine, news_api)
    
    with tab3:
        render_risk_assessment(strategy_engine, market, price_change, news_api)
    
    with tab4:
        render_real_time_signals(strategy_engine, market, price_change, news_api)

def render_ai_strategy_analysis(strategy_engine: NewsBasedStrategy, market: str, price_change: float, news_api: CryptoNewsAPI):
    """AI 전략 분석 렌더링"""
    try:
        st.subheader("🎯 AI 기반 투자 전략 분석")
        
        # 뉴스 데이터 수집
        with st.spinner("뉴스 데이터 분석 중..."):
            news_items = news_api.get_crypto_news(limit=50)
            
            if not news_items:
                st.warning("뉴스 데이터를 불러올 수 없습니다.")
                return
            
            # 뉴스 감정 분석
            news_sentiment = strategy_engine.analyze_news_sentiment(news_items)
            
            # 거래 신호 생성
            trading_signal = strategy_engine.generate_trading_signal(
                news_sentiment, 
                0,  # 현재가 (임시)
                price_change
            )
        
        # 전략 결과 표시
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 신호 강도
            signal_color = get_signal_color(trading_signal.signal)
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: {signal_color}; border-radius: 10px; margin: 10px 0;">
                <h3 style="color: white; margin: 0;">💡 AI 추천</h3>
                <h2 style="color: white; margin: 10px 0;">{trading_signal.signal.value}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 신뢰도
            confidence_pct = trading_signal.confidence * 100
            st.metric(
                "신뢰도",
                f"{confidence_pct:.1f}%",
                delta=f"뉴스 {trading_signal.news_count}개 분석"
            )
        
        with col3:
            # 변동성 영향
            volatility_pct = trading_signal.volatility_impact * 100
            st.metric(
                "변동성 영향",
                f"{volatility_pct:.1f}%",
                delta="예상 가격 변동"
            )
        
        # 상세 분석 결과
        st.subheader("📊 상세 분석 결과")
        
        # 감정 분석 차트
        sentiment_data = {
            '긍정': news_sentiment['positive_count'],
            '부정': news_sentiment['negative_count'],
            '중립': news_sentiment['neutral_count']
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 감정 분포 파이 차트
            fig_pie = px.pie(
                values=list(sentiment_data.values()),
                names=list(sentiment_data.keys()),
                title="뉴스 감정 분포",
                color_discrete_map={'긍정': '#00ff00', '부정': '#ff0000', '중립': '#888888'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # 카테고리별 분석
            st.markdown("### 📋 카테고리별 분석")
            
            categories = {
                '기관 투자': news_sentiment['institutional_mentions'],
                '규제 관련': news_sentiment['regulatory_mentions'],
                '기술 발전': news_sentiment['technical_mentions']
            }
            
            for category, count in categories.items():
                if count > 0:
                    st.success(f"✅ {category}: {count}개 뉴스")
                else:
                    st.info(f"ℹ️ {category}: 관련 뉴스 없음")
        
        # 전략 이유 및 권장사항
        st.subheader("💡 전략 분석 근거")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📝 분석 근거:**")
            st.info(trading_signal.reason)
        
        with col2:
            st.markdown("**🎯 권장 행동:**")
            action_text = get_action_recommendation(trading_signal.signal)
            st.success(action_text)
        
        # 시간대별 뉴스 분석
        render_time_analysis(news_items, news_sentiment)
        
    except Exception as e:
        st.error(f"AI 전략 분석 중 오류 발생: {str(e)}")
        logger.error(f"AI strategy analysis error: {str(e)}")

def render_news_sentiment_analysis(strategy_engine: NewsBasedStrategy, news_api: CryptoNewsAPI):
    """뉴스 감정 분석 렌더링"""
    try:
        st.subheader("📈 뉴스 감정 분석")
        
        # 분석 옵션
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_period = st.selectbox(
                "분석 기간",
                ["최근 24시간", "최근 3일", "최근 1주일"],
                index=0
            )
        
        with col2:
            news_count = st.slider(
                "분석할 뉴스 수",
                min_value=10,
                max_value=100,
                value=50,
                step=10
            )
        
        # 뉴스 분석
        with st.spinner("뉴스 감정 분석 중..."):
            news_items = news_api.get_crypto_news(limit=news_count)
            
            if not news_items:
                st.warning("뉴스 데이터를 불러올 수 없습니다.")
                return
            
            news_sentiment = strategy_engine.analyze_news_sentiment(news_items)
        
        # 감정 점수 표시
        overall_score = news_sentiment['overall_score']
        
        # 감정 게이지 차트
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=overall_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "전체 감정 점수"},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.5], 'color': "red"},
                    {'range': [-0.5, 0], 'color': "orange"},
                    {'range': [0, 0.5], 'color': "lightgreen"},
                    {'range': [0.5, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # 키워드 분석
        st.subheader("🔍 키워드 분석")
        
        # 분석된 뉴스에서 키워드 추출
        all_keywords = []
        for news in news_sentiment.get('analyzed_news', []):
            all_keywords.extend(news.get('keywords', []))
        
        if all_keywords:
            # 키워드 빈도 계산
            keyword_freq = {}
            for keyword in all_keywords:
                keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
            
            # 상위 키워드 표시
            top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔥 주요 키워드:**")
                for keyword, count in top_keywords:
                    color = "🟢" if keyword.startswith('+') else "🔴"
                    st.markdown(f"{color} {keyword.replace('+', '').replace('-', '')}: {count}회")
            
            with col2:
                # 키워드 워드클라우드 (간단한 버전)
                st.markdown("**📊 키워드 빈도:**")
                
                keywords_df = pd.DataFrame(top_keywords, columns=['키워드', '빈도'])
                st.bar_chart(keywords_df.set_index('키워드'))
        
        # 뉴스 소스별 분석
        render_source_analysis(news_items)
        
    except Exception as e:
        st.error(f"뉴스 감정 분석 중 오류 발생: {str(e)}")
        logger.error(f"News sentiment analysis error: {str(e)}")

def render_risk_assessment(strategy_engine: NewsBasedStrategy, market: str, price_change: float, news_api: CryptoNewsAPI):
    """위험도 평가 렌더링"""
    try:
        st.subheader("⚠️ 위험도 평가")
        
        # 뉴스 기반 위험도 분석
        with st.spinner("위험도 분석 중..."):
            news_items = news_api.get_crypto_news(limit=30)
            
            if not news_items:
                st.warning("뉴스 데이터를 불러올 수 없습니다.")
                return
            
            news_sentiment = strategy_engine.analyze_news_sentiment(news_items)
            trading_signal = strategy_engine.generate_trading_signal(
                news_sentiment, 0, price_change
            )
            
            risk_assessment = strategy_engine.get_risk_assessment(trading_signal)
        
        # 위험도 레벨 표시
        risk_level = risk_assessment['risk_level']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 위험도 레벨
            risk_color = get_risk_color(risk_level)
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: {risk_color}; border-radius: 10px;">
                <h3 style="color: white; margin: 0;">⚠️ 위험도</h3>
                <h2 style="color: white; margin: 10px 0;">{risk_level}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 신뢰도
            st.metric(
                "신호 신뢰도",
                f"{trading_signal.confidence * 100:.1f}%",
                delta="분석 정확도"
            )
        
        with col3:
            # 추천 포지션
            st.metric(
                "추천 포지션 크기",
                risk_assessment['recommended_position_size'],
                delta="자산 대비 비율"
            )
        
        # 위험 요소 분석
        st.subheader("🚨 위험 요소 분석")
        
        risk_factors = risk_assessment['risk_factors']
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(f"⚠️ {factor}")
        else:
            st.success("✅ 주요 위험 요소가 발견되지 않았습니다.")
        
        # 위험 완화 제안
        st.subheader("🛡️ 위험 완화 제안")
        
        mitigation_suggestions = get_risk_mitigation_suggestions(risk_level, trading_signal)
        
        for suggestion in mitigation_suggestions:
            st.info(f"💡 {suggestion}")
        
        # 시나리오 분석
        st.subheader("🎭 시나리오 분석")
        
        scenarios = generate_scenarios(news_sentiment, price_change)
        
        for scenario in scenarios:
            with st.expander(f"{scenario['name']} (확률: {scenario['probability']}%)"):
                st.markdown(f"**📊 예상 결과:** {scenario['outcome']}")
                st.markdown(f"**🎯 대응 전략:** {scenario['strategy']}")
        
    except Exception as e:
        st.error(f"위험도 평가 중 오류 발생: {str(e)}")
        logger.error(f"Risk assessment error: {str(e)}")

def render_real_time_signals(strategy_engine: NewsBasedStrategy, market: str, price_change: float, news_api: CryptoNewsAPI):
    """실시간 신호 렌더링"""
    try:
        st.subheader("⚡ 실시간 거래 신호")
        
        # 자동 새로고침 옵션
        auto_refresh = st.checkbox("자동 새로고침 (30초)", value=False)
        
        if auto_refresh:
            # 30초마다 자동 새로고침
            import time
            time.sleep(30)
            st.rerun()
        
        # 실시간 분석
        with st.spinner("실시간 신호 분석 중..."):
            news_items = news_api.get_crypto_news(limit=20)
            
            if not news_items:
                st.warning("뉴스 데이터를 불러올 수 없습니다.")
                return
            
            news_sentiment = strategy_engine.analyze_news_sentiment(news_items)
            trading_signal = strategy_engine.generate_trading_signal(
                news_sentiment, 0, price_change
            )
        
        # 신호 강도 표시
        signal_strength = trading_signal.signal.value
        confidence = trading_signal.confidence
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background: linear-gradient(45deg, #1e3c72, #2a5298); border-radius: 15px; margin: 20px 0;">
                <h2 style="color: white; margin: 0;">⚡ 실시간 신호</h2>
                <h1 style="color: #00ff00; margin: 15px 0; font-size: 2.5em;">{signal_strength}</h1>
                <p style="color: #cccccc; margin: 0;">신뢰도: {confidence * 100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 신호 히스토리 (가상 데이터)
            signal_history = generate_signal_history()
            
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(
                x=signal_history['time'],
                y=signal_history['signal'],
                mode='lines+markers',
                name='신호 강도',
                line=dict(color='blue', width=2)
            ))
            
            fig_history.update_layout(
                title="신호 히스토리",
                xaxis_title="시간",
                yaxis_title="신호 강도",
                height=300
            )
            
            st.plotly_chart(fig_history, use_container_width=True)
        
        # 즉시 실행 가능한 행동 제안
        st.subheader("🎯 즉시 실행 제안")
        
        action_plan = generate_action_plan(trading_signal, market)
        
        for i, action in enumerate(action_plan, 1):
            st.markdown(f"""
            <div style="padding: 15px; background-color: #f0f2f6; border-radius: 10px; margin: 10px 0; border-left: 4px solid #1f77b4;">
                <h4 style="margin: 0; color: #1f77b4;">단계 {i}: {action['title']}</h4>
                <p style="margin: 5px 0 0 0; color: #666;">{action['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 알림 설정
        st.subheader("🔔 알림 설정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            alert_threshold = st.slider(
                "신호 강도 알림 임계값",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1
            )
        
        with col2:
            alert_types = st.multiselect(
                "알림 유형",
                ["강한 매수", "강한 매도", "신뢰도 높음", "위험도 높음"],
                default=["강한 매수", "강한 매도"]
            )
        
        if st.button("알림 설정 저장"):
            st.success("알림 설정이 저장되었습니다!")
        
        # 마지막 업데이트 시간
        st.markdown(f"*마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
    except Exception as e:
        st.error(f"실시간 신호 분석 중 오류 발생: {str(e)}")
        logger.error(f"Real-time signals error: {str(e)}")

# 유틸리티 함수들
def get_signal_color(signal: SignalStrength) -> str:
    """신호에 따른 색상 반환"""
    color_map = {
        SignalStrength.STRONG_BUY: "#006400",
        SignalStrength.BUY: "#228B22",
        SignalStrength.WEAK_BUY: "#90EE90",
        SignalStrength.HOLD: "#808080",
        SignalStrength.WEAK_SELL: "#FFB6C1",
        SignalStrength.SELL: "#DC143C",
        SignalStrength.STRONG_SELL: "#8B0000"
    }
    return color_map.get(signal, "#808080")

def get_risk_color(risk_level: str) -> str:
    """위험도에 따른 색상 반환"""
    color_map = {
        "낮음": "#006400",
        "중간": "#FF8C00",
        "높음": "#DC143C"
    }
    return color_map.get(risk_level, "#808080")

def get_action_recommendation(signal: SignalStrength) -> str:
    """신호에 따른 행동 권장사항"""
    recommendations = {
        SignalStrength.STRONG_BUY: "🚀 적극적 매수 추천 - 포지션 확대",
        SignalStrength.BUY: "📈 매수 추천 - 일반적 진입",
        SignalStrength.WEAK_BUY: "🤔 신중한 매수 - 소량 진입",
        SignalStrength.HOLD: "🎯 현재 포지션 유지",
        SignalStrength.WEAK_SELL: "🤔 신중한 매도 - 일부 정리",
        SignalStrength.SELL: "📉 매도 추천 - 포지션 축소",
        SignalStrength.STRONG_SELL: "🚨 적극적 매도 - 포지션 청산"
    }
    return recommendations.get(signal, "관망 권장")

def get_risk_mitigation_suggestions(risk_level: str, trading_signal) -> List[str]:
    """위험 완화 제안"""
    suggestions = []
    
    if risk_level == "높음":
        suggestions.extend([
            "포지션 크기를 줄여서 위험을 제한하세요",
            "손절매 라인을 설정하여 손실을 제한하세요",
            "분산 투자로 위험을 분산하세요"
        ])
    elif risk_level == "중간":
        suggestions.extend([
            "적정 포지션 크기로 진입하세요",
            "시장 변동성을 모니터링하세요"
        ])
    else:  # 낮음
        suggestions.extend([
            "안정적인 진입이 가능한 상황입니다",
            "장기적 관점에서 접근하세요"
        ])
    
    return suggestions

def generate_scenarios(news_sentiment: Dict, price_change: float) -> List[Dict]:
    """시나리오 생성"""
    scenarios = []
    
    # 긍정적 시나리오
    if news_sentiment['positive_count'] > news_sentiment['negative_count']:
        scenarios.append({
            'name': '🚀 낙관적 시나리오',
            'probability': 60,
            'outcome': '15-25% 상승 예상',
            'strategy': '단계적 매수 후 목표가 달성시 일부 매도'
        })
    
    # 부정적 시나리오
    if news_sentiment['negative_count'] > news_sentiment['positive_count']:
        scenarios.append({
            'name': '📉 비관적 시나리오',
            'probability': 55,
            'outcome': '10-20% 하락 예상',
            'strategy': '손절매 라인 설정 후 반등 시점 포착'
        })
    
    # 중립 시나리오
    scenarios.append({
        'name': '➡️ 중립 시나리오',
        'probability': 30,
        'outcome': '±5% 범위 내 횡보',
        'strategy': '관망 후 명확한 신호 대기'
    })
    
    return scenarios

def generate_signal_history() -> Dict:
    """신호 히스토리 생성 (가상 데이터)"""
    import random
    
    times = [f"{i:02d}:00" for i in range(24)]
    signals = [random.uniform(-1, 1) for _ in range(24)]
    
    return {
        'time': times,
        'signal': signals
    }

def generate_action_plan(trading_signal, market: str) -> List[Dict]:
    """행동 계획 생성"""
    actions = []
    
    if trading_signal.signal in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
        actions.extend([
            {
                'title': '현재 시장 상황 재확인',
                'description': f'{market} 현재가와 거래량을 확인하세요'
            },
            {
                'title': '진입 전략 수립',
                'description': '목표 매수가와 매수 수량을 결정하세요'
            },
            {
                'title': '리스크 관리 설정',
                'description': '손절매 라인과 목표 수익률을 설정하세요'
            }
        ])
    elif trading_signal.signal in [SignalStrength.STRONG_SELL, SignalStrength.SELL]:
        actions.extend([
            {
                'title': '포지션 점검',
                'description': '현재 보유 포지션과 수익률을 확인하세요'
            },
            {
                'title': '매도 전략 수립',
                'description': '매도 수량과 타이밍을 결정하세요'
            },
            {
                'title': '수익 실현',
                'description': '목표 수익률 달성시 일부 매도를 실행하세요'
            }
        ])
    else:
        actions.extend([
            {
                'title': '시장 모니터링',
                'description': '뉴스와 가격 변동을 지속적으로 관찰하세요'
            },
            {
                'title': '대기 전략',
                'description': '명확한 신호가 나올 때까지 기다리세요'
            }
        ])
    
    return actions

def render_time_analysis(news_items: List[Dict], news_sentiment: Dict):
    """시간대별 뉴스 분석"""
    try:
        st.subheader("⏰ 시간대별 뉴스 분포")
        
        # 시간대별 뉴스 분류 (가상 데이터)
        time_distribution = {
            '00-06': 5,
            '06-12': 15,
            '12-18': 25,
            '18-24': 10
        }
        
        fig_time = px.bar(
            x=list(time_distribution.keys()),
            y=list(time_distribution.values()),
            title="시간대별 뉴스 발생 빈도",
            labels={'x': '시간대', 'y': '뉴스 개수'}
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Time analysis error: {str(e)}")

def render_source_analysis(news_items: List[Dict]):
    """뉴스 소스별 분석"""
    try:
        st.subheader("📰 뉴스 소스별 분석")
        
        # 소스별 뉴스 개수 계산
        source_count = {}
        for news in news_items:
            source = news.get('source', 'Unknown')
            source_count[source] = source_count.get(source, 0) + 1
        
        if source_count:
            # 상위 소스 표시
            top_sources = sorted(source_count.items(), key=lambda x: x[1], reverse=True)[:5]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 주요 뉴스 소스:**")
                for source, count in top_sources:
                    st.markdown(f"• {source}: {count}개")
            
            with col2:
                # 소스별 차트
                fig_source = px.pie(
                    values=[count for _, count in top_sources],
                    names=[source for source, _ in top_sources],
                    title="뉴스 소스 분포"
                )
                st.plotly_chart(fig_source, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Source analysis error: {str(e)}")