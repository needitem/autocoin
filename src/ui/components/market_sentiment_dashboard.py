"""
시장 감정 분석 대시보드 UI 컴포넌트
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def render_market_sentiment_dashboard(market: str, price_data: Dict, 
                                    historical_data: Optional[any], news_api):
    """시장 감정 분석 대시보드 렌더링"""
    
    st.header("🧠 향상된 시장 감정 분석")
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs([
        "종합 감정 지표", "구성 요소 분석", "소셜 미디어", "Fear & Greed Index"
    ])
    
    with tab1:
        render_overall_sentiment(market, price_data, historical_data, news_api)
    
    with tab2:
        render_component_analysis(market, price_data, historical_data, news_api)
    
    with tab3:
        render_social_media_sentiment(market, price_data, historical_data, news_api)
    
    with tab4:
        render_fear_greed_index(market, price_data, historical_data, news_api)

def render_overall_sentiment(market: str, price_data: Dict, 
                            historical_data: Optional[any], news_api):
    """종합 감정 지표 렌더링"""
    try:
        with st.spinner("시장 감정 분석 중..."):
            # 동기 함수 호출
            sentiment_data = news_api.get_market_sentiment(market, price_data, historical_data)
        
        # 메인 감정 지표
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 감정 상태
            sentiment = sentiment_data.get('sentiment', 'neutral')
            color_map = {
                'bullish': '#00ff00',
                'bearish': '#ff0000',
                'neutral': '#808080'
            }
            emoji_map = {
                'bullish': '🚀',
                'bearish': '📉',
                'neutral': '➡️'
            }
            
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, {color_map[sentiment]}22, {color_map[sentiment]}44); border-radius: 15px; border: 2px solid {color_map[sentiment]};">
                <h1 style="color: {color_map[sentiment]}; margin: 0; font-size: 3em;">{emoji_map[sentiment]}</h1>
                <h2 style="color: white; margin: 10px 0;">{sentiment.upper()}</h2>
                <p style="color: #cccccc; margin: 0;">시장 감정</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 감정 점수 게이지
            score = sentiment_data.get('score', 0.5)
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "감정 점수"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "darkred"},
                        {'range': [20, 40], 'color': "red"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "lightgreen"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': score * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col3:
            # 신뢰도
            confidence = sentiment_data.get('confidence', 0.5)
            st.metric(
                "분석 신뢰도",
                f"{confidence * 100:.1f}%",
                delta=f"{'높음' if confidence > 0.7 else '보통' if confidence > 0.4 else '낮음'}"
            )
            
            # Fear & Greed Index
            fgi = sentiment_data.get('fear_greed_index', 50)
            st.metric(
                "Fear & Greed Index",
                f"{fgi:.0f}",
                delta=f"{'극도의 탐욕' if fgi > 80 else '탐욕' if fgi > 60 else '중립' if fgi > 40 else '공포' if fgi > 20 else '극도의 공포'}"
            )
        
        # 주요 시그널
        st.subheader("📊 주요 시장 시그널")
        
        reasons = sentiment_data.get('reasons', [])
        if reasons:
            for reason in reasons:
                st.info(f"• {reason}")
        else:
            st.info("시장 시그널을 분석 중입니다...")
        
        # 시장 지표 요약
        if 'components' in sentiment_data and sentiment_data['components']:
            st.subheader("📈 시장 지표 요약")
            
            components = sentiment_data['components']
            
            # 컴포넌트 시각화
            fig_components = go.Figure()
            
            component_names = list(components.keys())
            component_values = list(components.values())
            
            # 막대 그래프
            colors = ['green' if v > 0 else 'red' for v in component_values]
            
            fig_components.add_trace(go.Bar(
                x=component_names,
                y=component_values,
                marker_color=colors,
                text=[f"{v:.2f}" for v in component_values],
                textposition='auto'
            ))
            
            fig_components.update_layout(
                title="감정 구성 요소",
                yaxis_title="점수",
                xaxis_title="구성 요소",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig_components, use_container_width=True)
        
    except Exception as e:
        st.error(f"감정 분석 중 오류 발생: {str(e)}")
        logger.error(f"Overall sentiment error: {str(e)}")

def render_component_analysis(market: str, price_data: Dict, 
                            historical_data: Optional[any], news_api):
    """구성 요소 상세 분석"""
    try:
        with st.spinner("구성 요소 분석 중..."):
            sentiment_data = news_api.get_market_sentiment(market, price_data, historical_data)
        
        components = sentiment_data.get('components', {})
        
        if not components:
            st.warning("구성 요소 데이터가 없습니다.")
            return
        
        # 각 구성 요소별 상세 분석
        st.subheader("🔍 구성 요소별 상세 분석")
        
        # 2열로 구성
        col1, col2 = st.columns(2)
        
        component_details = {
            'price_momentum': {
                'name': '가격 모멘텀',
                'icon': '📈',
                'description': '24시간 가격 변동과 이동평균 분석'
            },
            'volume_analysis': {
                'name': '거래량 분석',
                'icon': '📊',
                'description': '거래량 패턴과 가격-거래량 상관관계'
            },
            'volatility': {
                'name': '변동성',
                'icon': '⚡',
                'description': '가격 변동성과 리스크 수준'
            },
            'news_sentiment': {
                'name': '뉴스 감정',
                'icon': '📰',
                'description': '뉴스 기사의 긍정/부정 분석'
            },
            'social_sentiment': {
                'name': '소셜 미디어',
                'icon': '🐦',
                'description': '트위터, 레딧 등 소셜 미디어 감정'
            },
            'institutional_flow': {
                'name': '기관 자금 흐름',
                'icon': '🏦',
                'description': '대량 거래와 기관 투자자 동향'
            }
        }
        
        for i, (key, value) in enumerate(components.items()):
            with col1 if i % 2 == 0 else col2:
                detail = component_details.get(key, {'name': key, 'icon': '📊', 'description': ''})
                
                # 컴포넌트 카드
                color = '#00ff00' if value > 0.3 else '#ff0000' if value < -0.3 else '#ffff00'
                
                st.markdown(f"""
                <div style="padding: 20px; background-color: #1e1e1e; border-radius: 10px; margin: 10px 0; border-left: 4px solid {color};">
                    <h3 style="margin: 0; color: white;">{detail['icon']} {detail['name']}</h3>
                    <p style="color: #888; margin: 5px 0; font-size: 0.9em;">{detail['description']}</p>
                    <h2 style="color: {color}; margin: 10px 0;">{value:.2f}</h2>
                    <div style="background-color: #333; border-radius: 5px; height: 10px; margin-top: 10px;">
                        <div style="background-color: {color}; height: 100%; width: {abs(value) * 50 + 50}%; border-radius: 5px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # 구성 요소 간 상관관계
        st.subheader("🔗 구성 요소 간 상관관계")
        
        # 히트맵 생성
        import numpy as np
        
        # 시뮬레이션 상관관계 (실제로는 계산 필요)
        comp_names = list(components.keys())
        n = len(comp_names)
        correlation_matrix = np.random.rand(n, n)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        fig_heatmap = px.imshow(
            correlation_matrix,
            x=comp_names,
            y=comp_names,
            color_continuous_scale='RdBu',
            aspect='auto',
            title="구성 요소 상관관계 매트릭스"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    except Exception as e:
        st.error(f"구성 요소 분석 중 오류 발생: {str(e)}")
        logger.error(f"Component analysis error: {str(e)}")

def render_social_media_sentiment(market: str, price_data: Dict, 
                                historical_data: Optional[any], news_api):
    """소셜 미디어 감정 분석"""
    try:
        with st.spinner("소셜 미디어 분석 중..."):
            sentiment_data = news_api.get_market_sentiment(market, price_data, historical_data)
        
        social_metrics = sentiment_data.get('social_metrics', {})
        
        st.subheader("🐦 소셜 미디어 감정 분석")
        
        if not social_metrics:
            st.info("소셜 미디어 데이터를 수집 중입니다...")
            return
        
        # 주요 지표
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mentions = social_metrics.get('twitter_mentions', 0)
            st.metric(
                "트위터 멘션",
                f"{mentions:,}",
                delta="24시간"
            )
        
        with col2:
            reddit_posts = social_metrics.get('reddit_posts', 0)
            st.metric(
                "레딧 포스트",
                f"{reddit_posts:,}",
                delta="활성 토론"
            )
        
        with col3:
            positive_ratio = social_metrics.get('positive_ratio', 0.5)
            st.metric(
                "긍정 비율",
                f"{positive_ratio:.1%}",
                delta=f"{'긍정적' if positive_ratio > 0.6 else '부정적' if positive_ratio < 0.4 else '중립적'}"
            )
        
        with col4:
            engagement = social_metrics.get('engagement_rate', 0)
            st.metric(
                "참여율",
                f"{engagement:.1%}",
                delta="커뮤니티 활성도"
            )
        
        # 소셜 미디어 트렌드 차트
        st.subheader("📊 소셜 미디어 트렌드")
        
        # 시간대별 멘션 수 (시뮬레이션)
        import random
        hours = list(range(24))
        mentions_by_hour = [random.randint(100, 1000) for _ in hours]
        positive_by_hour = [random.uniform(0.3, 0.7) for _ in hours]
        
        fig_social = go.Figure()
        
        # 멘션 수
        fig_social.add_trace(go.Bar(
            x=hours,
            y=mentions_by_hour,
            name='멘션 수',
            marker_color='lightblue',
            yaxis='y'
        ))
        
        # 긍정 비율
        fig_social.add_trace(go.Scatter(
            x=hours,
            y=positive_by_hour,
            name='긍정 비율',
            line=dict(color='green', width=3),
            yaxis='y2'
        ))
        
        fig_social.update_layout(
            title="24시간 소셜 미디어 활동",
            xaxis_title="시간",
            yaxis=dict(title="멘션 수", side='left'),
            yaxis2=dict(title="긍정 비율", side='right', overlaying='y', range=[0, 1]),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_social, use_container_width=True)
        
        # 주요 키워드/해시태그
        st.subheader("🏷️ 인기 키워드 & 해시태그")
        
        # 워드 클라우드 시뮬레이션
        keywords = {
            '#Bitcoin': random.randint(1000, 5000),
            '#Crypto': random.randint(800, 4000),
            '#HODL': random.randint(500, 3000),
            '#BullRun': random.randint(300, 2000),
            '#Altcoin': random.randint(200, 1500),
            'moon': random.randint(500, 2000),
            'pump': random.randint(300, 1000),
            'dip': random.randint(200, 800)
        }
        
        # 키워드 빈도 차트
        fig_keywords = px.bar(
            x=list(keywords.values()),
            y=list(keywords.keys()),
            orientation='h',
            title="상위 키워드 빈도",
            labels={'x': '멘션 수', 'y': '키워드'}
        )
        
        st.plotly_chart(fig_keywords, use_container_width=True)
        
    except Exception as e:
        st.error(f"소셜 미디어 분석 중 오류 발생: {str(e)}")
        logger.error(f"Social media sentiment error: {str(e)}")

def render_fear_greed_index(market: str, price_data: Dict, 
                          historical_data: Optional[any], news_api):
    """Fear & Greed Index 렌더링"""
    try:
        with st.spinner("Fear & Greed Index 계산 중..."):
            sentiment_data = news_api.get_market_sentiment(market, price_data, historical_data)
        
        fgi = sentiment_data.get('fear_greed_index', 50)
        
        st.subheader("😨😎 Fear & Greed Index")
        
        # 대형 게이지 차트
        fig_fgi = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fgi,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "암호화폐 시장 Fear & Greed Index", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': '#8B0000', 'name': '극도의 공포'},
                    {'range': [20, 40], 'color': '#DC143C', 'name': '공포'},
                    {'range': [40, 60], 'color': '#FFD700', 'name': '중립'},
                    {'range': [60, 80], 'color': '#90EE90', 'name': '탐욕'},
                    {'range': [80, 100], 'color': '#006400', 'name': '극도의 탐욕'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': fgi
                }
            }
        ))
        
        fig_fgi.update_layout(
            height=400,
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        st.plotly_chart(fig_fgi, use_container_width=True)
        
        # FGI 해석
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 현재 시장 상태")
            
            if fgi >= 80:
                status = "극도의 탐욕"
                color = "#006400"
                advice = "시장이 과열되었을 수 있습니다. 조정 가능성에 주의하세요."
            elif fgi >= 60:
                status = "탐욕"
                color = "#90EE90"
                advice = "시장이 낙관적입니다. 신중한 접근이 필요합니다."
            elif fgi >= 40:
                status = "중립"
                color = "#FFD700"
                advice = "시장이 균형잡힌 상태입니다. 기회를 모색하세요."
            elif fgi >= 20:
                status = "공포"
                color = "#DC143C"
                advice = "시장이 두려움에 빠져있습니다. 매수 기회일 수 있습니다."
            else:
                status = "극도의 공포"
                color = "#8B0000"
                advice = "시장이 극도로 비관적입니다. 역발상 투자 기회를 고려하세요."
            
            st.markdown(f"""
            <div style="padding: 20px; background-color: {color}22; border-radius: 10px; border: 2px solid {color};">
                <h2 style="color: {color}; margin: 0;">{status}</h2>
                <p style="color: white; margin: 10px 0;">{advice}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📈 지표 구성")
            
            # FGI 구성 요소
            components = {
                '가격 모멘텀': 25,
                '거래량': 20,
                '변동성': 20,
                '소셜 미디어': 20,
                '시장 지배력': 15
            }
            
            for component, weight in components.items():
                st.progress(weight/100, f"{component}: {weight}%")
        
        # 역사적 FGI 추이 (시뮬레이션)
        st.subheader("📉 Fear & Greed Index 추이")
        
        import pandas as pd
        import random
        
        # 30일 데이터 시뮬레이션
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        fgi_history = [random.uniform(20, 80) for _ in range(30)]
        fgi_history[-1] = fgi  # 현재 값
        
        fig_history = go.Figure()
        
        fig_history.add_trace(go.Scatter(
            x=dates,
            y=fgi_history,
            mode='lines+markers',
            name='FGI',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,100,200,0.2)'
        ))
        
        # 구간 표시
        fig_history.add_hrect(y0=0, y1=20, fillcolor="red", opacity=0.1)
        fig_history.add_hrect(y0=20, y1=40, fillcolor="orange", opacity=0.1)
        fig_history.add_hrect(y0=40, y1=60, fillcolor="yellow", opacity=0.1)
        fig_history.add_hrect(y0=60, y1=80, fillcolor="lightgreen", opacity=0.1)
        fig_history.add_hrect(y0=80, y1=100, fillcolor="darkgreen", opacity=0.1)
        
        fig_history.update_layout(
            title="30일 Fear & Greed Index 변화",
            xaxis_title="날짜",
            yaxis_title="FGI",
            yaxis=dict(range=[0, 100]),
            height=400
        )
        
        st.plotly_chart(fig_history, use_container_width=True)
        
        # 투자 전략 제안
        st.subheader("💡 FGI 기반 투자 전략")
        
        strategies = {
            "극도의 공포 (0-20)": [
                "역발상 매수 전략 고려",
                "단계적 분할 매수",
                "장기 투자 관점 유지"
            ],
            "공포 (20-40)": [
                "선별적 매수 기회 탐색",
                "우량 자산 중심 투자",
                "리스크 관리 강화"
            ],
            "중립 (40-60)": [
                "균형잡힌 포트폴리오 유지",
                "시장 동향 면밀히 관찰",
                "단기 거래 기회 활용"
            ],
            "탐욕 (60-80)": [
                "이익 실현 고려",
                "포지션 축소 검토",
                "하락 리스크 대비"
            ],
            "극도의 탐욕 (80-100)": [
                "적극적 이익 실현",
                "신규 진입 자제",
                "현금 비중 확대"
            ]
        }
        
        current_range = None
        if fgi < 20:
            current_range = "극도의 공포 (0-20)"
        elif fgi < 40:
            current_range = "공포 (20-40)"
        elif fgi < 60:
            current_range = "중립 (40-60)"
        elif fgi < 80:
            current_range = "탐욕 (60-80)"
        else:
            current_range = "극도의 탐욕 (80-100)"
        
        if current_range:
            st.markdown(f"**현재 구간: {current_range}**")
            for strategy in strategies[current_range]:
                st.info(f"• {strategy}")
        
    except Exception as e:
        st.error(f"Fear & Greed Index 분석 중 오류 발생: {str(e)}")
        logger.error(f"FGI error: {str(e)}")