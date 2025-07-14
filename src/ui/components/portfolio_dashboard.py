"""
AI 포트폴리오 전략 대시보드
전체 자금 배분, 매매 추천, 리스크 관리를 시각화하는 통합 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import logging

from src.core.ai_portfolio_strategy import AIPortfolioStrategy, RiskLevel
from src.core.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)

def create_portfolio_dashboard():
    """AI 포트폴리오 전략 대시보드 생성"""
    
    st.title("🤖 AI 포트폴리오 전략 대시보드")
    st.markdown("---")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 포트폴리오 설정")
        
        # 리스크 레벨 선택
        risk_level_map = {
            "보수적 (Conservative)": RiskLevel.CONSERVATIVE,
            "중간 (Moderate)": RiskLevel.MODERATE,
            "공격적 (Aggressive)": RiskLevel.AGGRESSIVE
        }
        
        selected_risk = st.selectbox(
            "리스크 수준:",
            list(risk_level_map.keys()),
            index=1  # 기본값: 중간
        )
        
        risk_level = risk_level_map[selected_risk]
        
        # 초기 자금 설정
        initial_capital = st.number_input(
            "초기 투자 자금 (원):",
            min_value=100000,
            max_value=1000000000,
            value=10000000,
            step=1000000,
            format="%d"
        )
        
        # 자동 새로고침
        auto_refresh = st.checkbox("자동 새로고침 (10초)", value=False)
        
        if st.button("📊 포트폴리오 분석 실행", type="primary"):
            st.session_state.run_analysis = True
    
    # 메인 컨텐츠
    if hasattr(st.session_state, 'run_analysis') and st.session_state.run_analysis:
        run_portfolio_analysis(risk_level, initial_capital)
    else:
        show_portfolio_overview()
    
    # 자동 새로고침
    if auto_refresh:
        st.rerun()

def run_portfolio_analysis(risk_level: RiskLevel, initial_capital: float):
    """포트폴리오 분석 실행"""
    
    try:
        # 로딩 표시
        with st.spinner("AI 포트폴리오 전략을 분석하고 있습니다..."):
            
            # 포트폴리오 매니저 초기화
            portfolio_manager = PortfolioManager(initial_capital, risk_level)
            
            # API 객체 가져오기 (세션에서)
            if 'upbit_api' not in st.session_state or 'news_api' not in st.session_state:
                st.error("API 연결이 필요합니다. 메인 페이지에서 먼저 시작하세요.")
                return
            
            upbit_api = st.session_state.upbit_api
            news_api = st.session_state.news_api
            
            # 비동기 분석 실행
            try:
                # 동기적으로 실행 (Streamlit 환경)
                analysis_result = asyncio.run(
                    portfolio_manager.analyze_and_recommend(upbit_api, news_api)
                )
                
                if 'error' in analysis_result:
                    st.error(f"분석 중 오류: {analysis_result['error']}")
                    return
                
                # 분석 결과를 세션에 저장
                st.session_state.portfolio_analysis = analysis_result
                st.session_state.portfolio_manager = portfolio_manager
                
                # 결과 표시
                display_portfolio_results(analysis_result, portfolio_manager)
                
            except Exception as e:
                logger.error(f"포트폴리오 분석 실행 오류: {str(e)}")
                st.error(f"분석 실행 중 오류가 발생했습니다: {str(e)}")
                
    except Exception as e:
        logger.error(f"포트폴리오 분석 초기화 오류: {str(e)}")
        st.error(f"분석 초기화 중 오류가 발생했습니다: {str(e)}")

def display_portfolio_results(analysis_result: dict, portfolio_manager: PortfolioManager):
    """포트폴리오 분석 결과 표시"""
    
    # 1. 전체 요약 정보
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "총 포트폴리오 가치",
            f"{analysis_result['portfolio_value']:,.0f}원",
            f"{analysis_result['total_return']:+.2f}%"
        )
    
    with col2:
        st.metric(
            "가용 현금",
            f"{analysis_result['available_cash']:,.0f}원",
            f"{analysis_result['available_cash']/analysis_result['portfolio_value']*100:.1f}%"
        )
    
    with col3:
        recommendations = analysis_result.get('recommendations', [])
        rebalancing_count = len([r for r in recommendations if r['action'] != '보유'])
        st.metric(
            "리밸런싱 필요",
            f"{rebalancing_count}개 코인",
            "필요" if analysis_result.get('rebalancing_needed', False) else "불필요"
        )
    
    with col4:
        risk_metrics = analysis_result.get('risk_metrics', {})
        sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
        st.metric(
            "샤프 비율",
            f"{sharpe_ratio:.2f}",
            "우수" if sharpe_ratio > 1 else "보통" if sharpe_ratio > 0.5 else "개선필요"
        )
    
    st.markdown("---")
    
    # 2. 탭으로 구분된 상세 정보
    tab1, tab2, tab3, tab4 = st.tabs(["📊 포트폴리오 배분", "💰 매매 추천", "📈 코인 분석", "⚠️ 리스크 관리"])
    
    with tab1:
        display_portfolio_allocation(analysis_result)
    
    with tab2:
        display_trading_recommendations(analysis_result)
    
    with tab3:
        display_coin_analysis(analysis_result)
    
    with tab4:
        display_risk_management(analysis_result, portfolio_manager)

def display_portfolio_allocation(analysis_result: dict):
    """포트폴리오 배분 표시"""
    
    st.subheader("🎯 목표 vs 현재 배분")
    
    coin_analyses = analysis_result.get('coin_analyses', [])
    
    if not coin_analyses:
        st.warning("코인 분석 데이터가 없습니다.")
        return
    
    # 데이터 준비
    symbols = []
    target_allocations = []
    current_allocations = []
    scores = []
    
    for coin in coin_analyses:
        symbols.append(coin['symbol'].replace('KRW-', ''))
        target_allocations.append(coin['target_allocation'])
        current_allocations.append(coin['current_allocation'])
        scores.append(coin['overall_score'])
    
    # 포트폴리오 배분 차트
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('목표 배분', '현재 배분'),
        specs=[[{"type": "pie"}, {"type": "pie"}]]
    )
    
    # 목표 배분 파이 차트
    fig.add_trace(
        go.Pie(
            labels=symbols,
            values=target_allocations,
            name="목표 배분",
            hole=0.3
        ),
        row=1, col=1
    )
    
    # 현재 배분 파이 차트  
    fig.add_trace(
        go.Pie(
            labels=symbols,
            values=current_allocations,
            name="현재 배분",
            hole=0.3
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        title_text="포트폴리오 배분 비교"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 배분 차이 테이블
    st.subheader("📋 상세 배분 현황")
    
    df = pd.DataFrame({
        '코인': symbols,
        '현재 배분 (%)': current_allocations,
        '목표 배분 (%)': target_allocations,
        '차이 (%)': [t - c for t, c in zip(target_allocations, current_allocations)],
        '종합 점수': scores
    })
    
    # 차이에 따른 색상 적용
    def highlight_diff(val):
        if abs(val) > 5:
            return 'background-color: #ffcccc'  # 빨간색 (큰 차이)
        elif abs(val) > 2:
            return 'background-color: #ffffcc'  # 노란색 (중간 차이)
        else:
            return 'background-color: #ccffcc'  # 녹색 (작은 차이)
    
    styled_df = df.style.map(highlight_diff, subset=['차이 (%)'])
    st.dataframe(styled_df, use_container_width=True)

def display_trading_recommendations(analysis_result: dict):
    """매매 추천 표시"""
    
    st.subheader("💡 AI 매매 추천")
    
    recommendations = analysis_result.get('recommendations', [])
    
    if not recommendations:
        st.warning("매매 추천이 없습니다.")
        return
    
    # 액션별 필터링
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_buy = st.checkbox("매수 추천", value=True)
    with col2:
        show_sell = st.checkbox("매도 추천", value=True)
    with col3:
        show_hold = st.checkbox("보유 추천", value=False)
    
    # 추천 카드 표시
    for rec in recommendations:
        action = rec['action']
        
        # 필터링
        if action == '매수' and not show_buy:
            continue
        elif action == '매도' and not show_sell:
            continue
        elif action == '보유' and not show_hold:
            continue
        
        # 액션별 색상
        if action == '매수':
            border_color = '#28a745'  # 녹색
            icon = '📈'
        elif action == '매도':
            border_color = '#dc3545'  # 빨간색
            icon = '📉'
        else:
            border_color = '#6c757d'  # 회색
            icon = '⏸️'
        
        with st.container():
            st.markdown(f"""
            <div style="border: 2px solid {border_color}; border-radius: 10px; padding: 15px; margin: 10px 0;">
                <h4>{icon} {rec['name']} ({rec['symbol']})</h4>
                <p><strong>추천 액션:</strong> {action}</p>
                <p><strong>현재 가격:</strong> {rec['current_price']:,.0f}원</p>
            """, unsafe_allow_html=True)
            
            # 매수/매도인 경우 추가 정보
            if action in ['매수', '매도']:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <p><strong>목표 가격:</strong> {rec.get('target_price', 0):,.0f}원</p>
                    <p><strong>수량:</strong> {rec.get('quantity', 0):.4f}개</p>
                    <p><strong>거래 금액:</strong> {rec.get('trade_amount', 0):,.0f}원</p>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <p><strong>목표 배분:</strong> {rec.get('target_allocation', 0):.1f}%</p>
                    <p><strong>현재 배분:</strong> {rec.get('current_allocation', 0):.1f}%</p>
                    <p><strong>신뢰도:</strong> {rec.get('confidence', 0):.1%}</p>
                    """, unsafe_allow_html=True)
                
                # 손절매/익절가 (매수인 경우)
                if action == '매수':
                    st.markdown(f"""
                    <p><strong>손절가:</strong> {rec.get('stop_loss', 0):,.0f}원</p>
                    <p><strong>익절가:</strong> {rec.get('take_profit', 0):,.0f}원</p>
                    """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <p><strong>추천 이유:</strong> {rec.get('reason', '정보 없음')}</p>
            </div>
            """, unsafe_allow_html=True)

def display_coin_analysis(analysis_result: dict):
    """코인별 상세 분석 표시"""
    
    st.subheader("🔍 코인별 상세 분석")
    
    coin_analyses = analysis_result.get('coin_analyses', [])
    
    if not coin_analyses:
        st.warning("코인 분석 데이터가 없습니다.")
        return
    
    # 점수별 정렬
    sorted_coins = sorted(coin_analyses, key=lambda x: x['overall_score'], reverse=True)
    
    # 점수 분포 히스토그램
    scores = [coin['overall_score'] for coin in coin_analyses]
    
    fig = go.Figure(data=[
        go.Histogram(
            x=scores,
            nbinsx=10,
            name="점수 분포",
            marker_color='skyblue'
        )
    ])
    
    fig.update_layout(
        title="코인별 종합 점수 분포",
        xaxis_title="종합 점수",
        yaxis_title="코인 수",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 상세 분석 테이블
    df_data = []
    for coin in sorted_coins:
        df_data.append({
            '순위': len(df_data) + 1,
            '코인': coin['symbol'],
            '현재가': f"{coin['current_price']:,.0f}원",
            '종합점수': f"{coin['overall_score']:.3f}",
            '목표배분': f"{coin['target_allocation']:.1f}%",
            '현재배분': f"{coin['current_allocation']:.1f}%",
            '신뢰도': f"{coin['confidence']:.1%}"
        })
    
    df = pd.DataFrame(df_data)
    
    # 점수별 색상 적용
    def color_score(val):
        score = float(val)
        if score >= 0.7:
            return 'background-color: #d4edda'  # 녹색 (우수)
        elif score >= 0.5:
            return 'background-color: #fff3cd'  # 노란색 (보통)
        else:
            return 'background-color: #f8d7da'  # 빨간색 (부진)
    
    styled_df = df.style.map(color_score, subset=['종합점수'])
    st.dataframe(styled_df, use_container_width=True)

def display_risk_management(analysis_result: dict, portfolio_manager: PortfolioManager):
    """리스크 관리 표시"""
    
    st.subheader("⚠️ 리스크 관리 현황")
    
    # 리스크 한계 확인
    risk_warnings = portfolio_manager.check_risk_limits()
    
    if risk_warnings:
        st.error("🚨 리스크 경고:")
        for warning in risk_warnings:
            st.error(f"• {warning}")
    else:
        st.success("✅ 모든 리스크 지표가 안전 범위 내에 있습니다.")
    
    # 리스크 지표
    risk_metrics = analysis_result.get('risk_metrics', {})
    
    if risk_metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_dd = risk_metrics.get('max_drawdown', 0)
            st.metric(
                "최대 낙폭 (MDD)",
                f"{max_dd:.1%}",
                "양호" if max_dd < 0.15 else "주의" if max_dd < 0.25 else "위험"
            )
        
        with col2:
            var_95 = risk_metrics.get('var_95', 0)
            st.metric(
                "VaR (95%)",
                f"{var_95:.1%}",
                "일일 최대 예상 손실"
            )
        
        with col3:
            volatility = risk_metrics.get('volatility', 0)
            st.metric(
                "연간 변동성",
                f"{volatility:.1%}",
                "낮음" if volatility < 0.3 else "보통" if volatility < 0.5 else "높음"
            )
    
    # 포트폴리오 성과
    performance = analysis_result.get('performance', {})
    
    if performance:
        st.subheader("📊 포트폴리오 성과")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "총 수익률",
                f"{performance.get('total_return_pct', 0):.2f}%"
            )
        
        with col2:
            st.metric(
                "평균 일일 수익률",
                f"{performance.get('avg_daily_return', 0):.3f}%"
            )
        
        with col3:
            st.metric(
                "승률",
                f"{performance.get('win_rate', 0):.1%}"
            )
        
        with col4:
            st.metric(
                "수익률 변동성",
                f"{performance.get('volatility', 0):.3f}%"
            )

def show_portfolio_overview():
    """포트폴리오 개요 표시"""
    
    st.markdown("""
    ## 🎯 AI 포트폴리오 전략 시스템
    
    ### 주요 기능:
    
    #### 📊 **지능형 자금 배분**
    - 10개 주요 코인에 대한 실시간 종합 분석
    - 3단계 티어 시스템 (BTC/ETH > 주요 알트코인 > 신규 코인)
    - 리스크 수준별 맞춤 배분 (보수적/중간/공격적)
    
    #### 🤖 **AI 기반 매매 신호**
    - 기술적 분석 + 감정 분석 + 거래량 분석 통합
    - 실시간 뉴스 감정 스코어링
    - 모멘텀 및 변동성 기반 타이밍 최적화
    
    #### ⚡ **리스크 관리**
    - 실시간 포지션 크기 모니터링
    - 자동 손절매/익절매 레벨 계산
    - 포트폴리오 전체 리스크 한계 관리
    
    #### 📈 **성과 추적**
    - 실시간 수익률 및 변동성 계산
    - 샤프 비율, VaR, 최대낙폭 등 전문 지표
    - 일일/주간/월간 성과 분석
    
    ### 시작하기:
    1. 왼쪽 사이드바에서 리스크 수준과 초기 자금을 설정하세요
    2. "📊 포트폴리오 분석 실행" 버튼을 클릭하세요
    3. AI가 실시간으로 최적의 포트폴리오를 제안합니다
    
    ---
    
    ⚠️ **주의사항**: 이 시스템은 투자 참고용이며, 실제 투자 결정은 신중하게 하시기 바랍니다.
    """)