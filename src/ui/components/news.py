"""
뉴스 및 시장 분석 컴포넌트
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from src.api.news import CryptoNewsAPI
import logging

logger = logging.getLogger(__name__)

def render_news_section(market: str, price_change: float = 0):
    """뉴스 및 시장 분석 섹션 렌더링"""
    
    st.header("📰 뉴스 & 시장 분석")
    
    # 뉴스 API 초기화
    news_api = CryptoNewsAPI()
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["최신 뉴스", "가격 변동 분석", "시장 감정", "🧠 AI 전략"])
    
    with tab1:
        render_latest_news(news_api)
    
    with tab2:
        render_price_analysis(news_api, market, price_change)
    
    with tab3:
        render_market_sentiment(news_api, market)
    
    with tab4:
        from src.ui.components.news_strategy import render_news_strategy_section
        render_news_strategy_section(market, price_change, news_api)

def render_latest_news(news_api: CryptoNewsAPI):
    """최신 뉴스 렌더링 (페이지네이션 지원)"""
    try:
        # 세션 상태 초기화
        if 'news_page' not in st.session_state:
            st.session_state.news_page = 1
        if 'news_per_page' not in st.session_state:
            st.session_state.news_per_page = 20
        if 'news_hours_filter' not in st.session_state:
            st.session_state.news_hours_filter = 24
        
        # 페이지 설정 UI
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            # 페이지 당 뉴스 수 선택
            per_page = st.selectbox(
                "페이지 당 뉴스 수",
                [10, 20, 50, 100],
                index=1,  # 기본값 20
                key="news_per_page_select"
            )
            st.session_state.news_per_page = per_page
        
        with col2:
            # 시간 필터 선택
            hours_filter = st.selectbox(
                "시간 필터",
                [1, 6, 12, 24, 48, 72, 0],  # 0은 전체
                index=3,  # 기본값 24시간
                format_func=lambda x: f"{x}시간 이내" if x > 0 else "전체",
                key="news_hours_filter_select"
            )
            st.session_state.news_hours_filter = hours_filter
        
        with col3:
            # 현재 페이지 표시
            st.metric("현재 페이지", st.session_state.news_page)
        
        with col4:
            # 페이지 네비게이션
            nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
            
            with nav_col1:
                if st.button("⏮️ 첫 페이지", disabled=st.session_state.news_page <= 1):
                    st.session_state.news_page = 1
                    st.rerun()
            
            with nav_col2:
                if st.button("◀️ 이전", disabled=st.session_state.news_page <= 1):
                    st.session_state.news_page -= 1
                    st.rerun()
            
            with nav_col3:
                if st.button("▶️ 다음"):
                    st.session_state.news_page += 1
                    st.rerun()
            
            with nav_col4:
                # 페이지 직접 입력
                target_page = st.number_input(
                    "페이지 이동",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.news_page,
                    key="page_input"
                )
                if target_page != st.session_state.news_page:
                    st.session_state.news_page = target_page
                    st.rerun()
        
        st.divider()
        
        # 뉴스 로딩
        filter_text = f"{st.session_state.news_hours_filter}시간 이내" if st.session_state.news_hours_filter > 0 else "전체"
        with st.spinner(f"뉴스를 불러오는 중... (페이지 {st.session_state.news_page}, {filter_text})"):
            news_items = news_api.get_crypto_news(
                limit=st.session_state.news_per_page,
                page=st.session_state.news_page,
                hours_filter=st.session_state.news_hours_filter
            )
        
        if not news_items:
            st.warning("뉴스를 불러올 수 없습니다.")
            return
        
        # 뉴스 통계 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📰 총 {len(news_items)}개의 뉴스")
        with col2:
            st.info(f"📄 페이지 {st.session_state.news_page}")
        with col3:
            filter_text = f"{st.session_state.news_hours_filter}시간 이내" if st.session_state.news_hours_filter > 0 else "전체"
            st.info(f"🕐 {filter_text}")
        
        # 뉴스 표시
        for i, news in enumerate(news_items):
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # 제목과 링크 (새 탭에서 열기)
                    st.markdown(f"**<a href='{news['link']}' target='_blank' style='text-decoration: none; color: #1f77b4;'>{news['title']}</a>**", unsafe_allow_html=True)
                    
                    # 요약
                    if news.get('summary'):
                        summary_text = news['summary'][:200] + "..." if len(news['summary']) > 200 else news['summary']
                        st.markdown(f"*{summary_text}*")
                
                with col2:
                    # 소스와 시간
                    st.markdown(f"**{news['source']}**")
                    
                    # 시간 포맷팅
                    try:
                        if news.get('published_at'):
                            # 다양한 날짜 형식 처리
                            pub_date_str = news['published_at']
                            pub_time = None
                            
                            # ISO 형식 시도
                            try:
                                pub_time = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                            except:
                                # RFC 2822 형식 시도 (RSS 표준)
                                try:
                                    from email.utils import parsedate_to_datetime
                                    pub_time = parsedate_to_datetime(pub_date_str)
                                except:
                                    # 기타 형식 시도
                                    try:
                                        pub_time = datetime.strptime(pub_date_str, '%Y-%m-%d %H:%M:%S')
                                    except:
                                        pass
                            
                            if pub_time:
                                # 현재 시간과 비교하여 적절한 표시
                                now = datetime.now()
                                if pub_time.tzinfo:
                                    # 시간대 정보가 있으면 로컬 시간으로 변환
                                    import pytz
                                    pub_time = pub_time.astimezone(pytz.timezone('Asia/Seoul'))
                                    pub_time = pub_time.replace(tzinfo=None)
                                
                                time_diff = now - pub_time
                                
                                if time_diff.days > 0:
                                    st.markdown(f"*{time_diff.days}일 전*")
                                elif time_diff.seconds > 3600:
                                    hours = time_diff.seconds // 3600
                                    st.markdown(f"*{hours}시간 전*")
                                elif time_diff.seconds > 60:
                                    minutes = time_diff.seconds // 60
                                    st.markdown(f"*{minutes}분 전*")
                                else:
                                    st.markdown(f"*{pub_time.strftime('%m-%d %H:%M')}*")
                            else:
                                st.markdown("*날짜 불명*")
                        else:
                            st.markdown("*날짜 불명*")
                    except Exception as e:
                        st.markdown("*날짜 불명*")
                
                # 구분선
                if i < len(news_items) - 1:
                    st.divider()
        
        # 하단 페이지 네비게이션
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⏮️ 첫 페이지", key="bottom_first", disabled=st.session_state.news_page <= 1):
                st.session_state.news_page = 1
                st.rerun()
        
        with col2:
            page_info = f"페이지 {st.session_state.news_page} | {st.session_state.news_per_page}개씩 표시"
            st.markdown(f"<div style='text-align: center; padding: 8px;'>{page_info}</div>", unsafe_allow_html=True)
        
        with col3:
            if st.button("▶️ 다음", key="bottom_next"):
                st.session_state.news_page += 1
                st.rerun()
                    
    except Exception as e:
        st.error(f"뉴스 로딩 중 오류 발생: {str(e)}")
        logger.error(f"News rendering error: {str(e)}")

def render_price_analysis(news_api: CryptoNewsAPI, market: str, price_change: float):
    """체계적인 가격 변동 분석 렌더링"""
    try:
        st.subheader(f"📊 {market} 체계적 가격 분석")
        
        # 체계적 분석기 임포트 및 초기화
        from src.core.price_analyzer import SystematicPriceAnalyzer
        analyzer = SystematicPriceAnalyzer()
        
        # 실제 시장 데이터 가져오기 (거래소별 API 선택)
        try:
            current_data = None
            historical_data = None
            
            # 마켓 코드로 거래소 판단
            if market.startswith('KRW-'):
                # 업비트 API 사용
                from src.api.upbit import UpbitTradingSystem
                exchange_api = UpbitTradingSystem()
                current_data = exchange_api.get_current_price(market)
                historical_data = exchange_api.fetch_ohlcv(market, interval='minute1', count=200)
                
            elif market.endswith('_KRW'):
                # 빗썸 API 사용
                from src.api.bithumb import BithumbAPI
                exchange_api = BithumbAPI()
                current_data = exchange_api.get_current_price(market)
                # 빗썸 API 메서드 확인 필요
                try:
                    historical_data = exchange_api.fetch_ohlcv(market, interval='1m', count=200)
                except:
                    historical_data = None
                
            else:
                # 기본적으로 업비트 사용
                from src.api.upbit import UpbitTradingSystem
                exchange_api = UpbitTradingSystem()
                current_data = exchange_api.get_current_price(market)
                historical_data = exchange_api.fetch_ohlcv(market, interval='minute1', count=200)
            
        except Exception as api_error:
            st.warning(f"실시간 데이터 로드 실패: {str(api_error)}")
            # 폴백 데이터 사용
            current_data = {
                'trade_price': 50000000,
                'signed_change_rate': price_change / 100,
                'acc_trade_volume_24h': 1000000
            }
            historical_data = None
        
        # 분석 실행
        with st.spinner("종합 분석 중..."):
            analysis = analyzer.analyze_comprehensive(
                market, current_data, historical_data, 
                exchange_api=exchange_api, enable_multi_timeframe=True
            )
        
        # 1. 기본 정보 표시
        st.subheader("📈 기본 정보")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if analysis.price_change_24h > 0:
                st.metric("24시간 변동", f"+{analysis.price_change_24h:.2f}%", 
                         delta=f"+{analysis.price_change_24h:.2f}%")
            else:
                st.metric("24시간 변동", f"{analysis.price_change_24h:.2f}%", 
                         delta=f"{analysis.price_change_24h:.2f}%")
        
        with col2:
            st.metric("1시간 변동", f"{analysis.price_change_1h:.2f}%")
        
        with col3:
            trend_emoji = {
                "강한 상승": "🚀",
                "상승": "📈", 
                "횡보": "➡️",
                "하락": "📉",
                "강한 하락": "⬇️"
            }
            st.metric("트렌드", f"{trend_emoji.get(analysis.trend_direction.value, '📊')} {analysis.trend_direction.value}")
        
        with col4:
            volatility_emoji = {
                "매우 높음": "🔥",
                "높음": "⚡",
                "보통": "📊",
                "낮음": "😴",
                "매우 낮음": "💤"
            }
            st.metric("변동성", f"{volatility_emoji.get(analysis.volatility_level.value, '📊')} {analysis.volatility_level.value}")
        
        # 2. 핵심 인사이트
        st.subheader("🔍 핵심 인사이트")
        for insight in analysis.key_insights:
            st.info(insight)
        
        # 3. 기술적 분석
        st.subheader("🔧 기술적 분석")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**주요 지표**")
            if analysis.technical_indicators['rsi'] != '데이터 부족':
                st.markdown(f"• RSI: {analysis.technical_indicators['rsi']}")
            if analysis.technical_indicators['macd'] != '데이터 부족':
                st.markdown(f"• MACD: {analysis.technical_indicators['macd']}")
            if analysis.technical_indicators['bb_position'] != '데이터 부족':
                st.markdown(f"• 볼린저 밴드: {analysis.technical_indicators['bb_position']}")
            
            signal = analysis.technical_indicators['signal']
            if signal == '매수 신호':
                st.success(f"📈 종합 신호: {signal}")
            elif signal == '매도 신호':
                st.error(f"📉 종합 신호: {signal}")
            else:
                st.info(f"➡️ 종합 신호: {signal}")
        
        with col2:
            st.markdown("**지지/저항 분석**")
            if analysis.support_resistance['current_level'] != '분석 불가':
                st.markdown(f"• 현재 위치: {analysis.support_resistance['current_level']}")
            
            if analysis.support_resistance['resistance_levels']:
                st.markdown("• 저항선:")
                for level in analysis.support_resistance['resistance_levels'][:2]:
                    st.markdown(f"  - {level:,.0f}")
            
            if analysis.support_resistance['support_levels']:
                st.markdown("• 지지선:")
                for level in analysis.support_resistance['support_levels'][:2]:
                    st.markdown(f"  - {level:,.0f}")
        
        # 4. 차트 패턴 분석
        st.subheader("🔍 차트 패턴 분석")
        pattern_analysis = analysis.pattern_analysis
        
        if pattern_analysis.get('primary_pattern'):
            primary = pattern_analysis['primary_pattern']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("주요 패턴", primary.get('name', '없음'))
            with col2:
                pattern_type = primary.get('type', '중립')
                if pattern_type == '반전':
                    st.metric("패턴 타입", f"🔄 {pattern_type}")
                elif pattern_type == '지속':
                    st.metric("패턴 타입", f"⏭️ {pattern_type}")
                else:
                    st.metric("패턴 타입", f"➡️ {pattern_type}")
            with col3:
                confidence = primary.get('confidence', 0)
                st.metric("신뢰도", f"{confidence:.1%}")
            
            # 패턴 신호 및 설명
            pattern_signal = primary.get('signal', '중립')
            if pattern_signal in ['강한 매수', '매수']:
                st.success(f"📈 패턴 신호: {pattern_signal}")
            elif pattern_signal in ['강한 매도', '매도']:
                st.error(f"📉 패턴 신호: {pattern_signal}")
            else:
                st.info(f"➡️ 패턴 신호: {pattern_signal}")
            
            st.markdown(f"**설명:** {primary.get('description', '설명 없음')}")
            
            # 감지된 모든 패턴 표시
            detected_patterns = pattern_analysis.get('detected_patterns', [])
            if detected_patterns:
                with st.expander(f"🔍 감지된 모든 패턴 ({len(detected_patterns)}개)"):
                    for i, pattern in enumerate(detected_patterns[:10], 1):  # 최대 10개만 표시
                        st.markdown(f"**{i}. {pattern.korean_name}**")
                        st.markdown(f"   - 타입: {pattern.pattern_type.value}")
                        st.markdown(f"   - 신호: {pattern.signal.value}")
                        st.markdown(f"   - 신뢰도: {pattern.confidence:.1%}")
                        st.markdown(f"   - 설명: {pattern.description}")
                        st.markdown("---")
        else:
            st.info("🔍 현재 명확한 패턴이 감지되지 않았습니다")
        
        # 5. 다중 시간대 분석
        if analysis.multi_timeframe_analysis:
            st.subheader("⏰ 다중 시간대 분석")
            
            mtf = analysis.multi_timeframe_analysis
            timeframe_analyses = mtf.get('timeframe_analyses', {})
            summary = mtf.get('summary', {})
            
            if timeframe_analyses:
                # 시간대별 요약 표시
                st.markdown("### 📊 시간대별 트렌드 요약")
                
                # 시간대별 신호 매트릭스
                timeframes = list(timeframe_analyses.keys())
                cols = st.columns(min(len(timeframes), 5))  # 최대 5개 컬럼
                
                for i, (tf, analysis_data) in enumerate(timeframe_analyses.items()):
                    if i < len(cols):
                        with cols[i]:
                            trend = analysis_data.get('trend_direction', '불분명')
                            strength = analysis_data.get('trend_strength', 0)
                            signal = analysis_data.get('signal_direction', '중립')
                            
                            # 트렌드 이모지
                            if '상승' in trend:
                                trend_emoji = "📈"
                                color = "green"
                            elif '하락' in trend:
                                trend_emoji = "📉"
                                color = "red"
                            else:
                                trend_emoji = "➡️"
                                color = "gray"
                            
                            st.metric(
                                label=tf,
                                value=f"{trend_emoji} {trend}",
                                delta=f"신호: {signal}"
                            )
                            
                            # 강도 프로그레스 바
                            st.progress(strength)
                            st.caption(f"강도: {strength:.1%}")
                
                # 종합 요약
                if summary:
                    st.markdown("### 🎯 다중 시간대 종합 분석")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        overall_trend = summary.get('overall_trend', '중립')
                        alignment = summary.get('trend_alignment', 0)
                        
                        if overall_trend == '상승':
                            st.success(f"📈 전체 트렌드: {overall_trend}")
                        elif overall_trend == '하락':
                            st.error(f"📉 전체 트렌드: {overall_trend}")
                        else:
                            st.info(f"➡️ 전체 트렌드: {overall_trend}")
                        
                        st.metric("트렌드 일치도", f"{alignment:.1%}")
                    
                    with col2:
                        volume_consensus = summary.get('volume_consensus', '혼재')
                        convergence = mtf.get('convergence_score', 0)
                        
                        if '매수' in volume_consensus:
                            st.success(f"💰 거래량: {volume_consensus}")
                        elif '매도' in volume_consensus:
                            st.error(f"💸 거래량: {volume_consensus}")
                        else:
                            st.warning(f"📊 거래량: {volume_consensus}")
                        
                        st.metric("신호 수렴도", f"{convergence:.1%}")
                    
                    with col3:
                        reliability = summary.get('reliability_average', 0)
                        
                        if reliability > 0.7:
                            st.success(f"✅ 신뢰도: {reliability:.1%}")
                        elif reliability > 0.5:
                            st.warning(f"⚠️ 신뢰도: {reliability:.1%}")
                        else:
                            st.error(f"❌ 신뢰도: {reliability:.1%}")
                
                # 시간대별 충돌 경고
                conflicts = mtf.get('conflicts', [])
                if conflicts:
                    st.warning(f"⚠️ 시간대별 신호 충돌: {', '.join(conflicts)}")
                
                # 상세 시간대별 분석 (확장 가능)
                with st.expander("📊 상세 시간대별 분석"):
                    for tf, analysis_data in timeframe_analyses.items():
                        st.markdown(f"### {tf}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**기본 정보**")
                            st.markdown(f"- 트렌드: {analysis_data.get('trend_direction', '불분명')}")
                            st.markdown(f"- 강도: {analysis_data.get('trend_strength', 0):.1%}")
                            st.markdown(f"- 신호: {analysis_data.get('signal_direction', '중립')}")
                            st.markdown(f"- 신뢰도: {analysis_data.get('reliability_score', 0):.1%}")
                        
                        with col2:
                            st.markdown("**거래량 분석**")
                            vol_analysis = analysis_data.get('volume_analysis', {})
                            buy_sell_ratio = vol_analysis.get('buy_sell_ratio', 1.0)
                            
                            if buy_sell_ratio > 1.2:
                                st.markdown(f"- 매수/매도 비율: {buy_sell_ratio:.2f} (매수 우세)")
                            elif buy_sell_ratio < 0.8:
                                st.markdown(f"- 매수/매도 비율: {buy_sell_ratio:.2f} (매도 우세)")
                            else:
                                st.markdown(f"- 매수/매도 비율: {buy_sell_ratio:.2f} (균형)")
                            
                            st.markdown(f"- 거래량 트렌드: {vol_analysis.get('volume_trend', '보통')}")
                            st.markdown(f"- 기관 활동: {vol_analysis.get('institutional_activity', '보통')}")
                        
                        # 핵심 인사이트
                        insights = analysis_data.get('key_insights', [])
                        if insights:
                            st.markdown("**핵심 인사이트**")
                            for insight in insights:
                                st.markdown(f"- {insight}")
                        
                        # 주요 패턴
                        patterns = analysis_data.get('primary_patterns', [])
                        if patterns:
                            st.markdown("**주요 패턴**")
                            for pattern in patterns:
                                st.markdown(f"- {pattern}")
                        
                        st.markdown("---")
                
                # 다중 시간대 추천사항
                recommendations = summary.get('recommendations', [])
                if recommendations:
                    st.markdown("### 💡 다중 시간대 기반 추천사항")
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"**{i}.** {rec}")
            
            else:
                st.info("⏰ 다중 시간대 분석 데이터가 부족합니다")
        
        # 6. 거래량 및 시장 심리
        st.subheader("💰 거래량 & 시장 심리")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**거래량 분석**")
            volume = analysis.volume_analysis
            st.markdown(f"• 거래량 수준: {volume.get('volume_level', '알 수 없음')}")
            st.markdown(f"• 거래량 추세: {volume.get('volume_trend', '알 수 없음')}")
            st.markdown(f"• 가격-거래량 상관관계: {volume.get('volume_price_correlation', '알 수 없음')}")
        
        with col2:
            st.markdown("**시장 심리**")
            sentiment = analysis.market_sentiment
            overall = sentiment.get('overall', '중립')
            
            if overall in ['매우 긍정', '긍정']:
                st.success(f"😊 전체 심리: {overall}")
            elif overall in ['매우 부정', '부정']:
                st.error(f"😰 전체 심리: {overall}")
            else:
                st.info(f"😐 전체 심리: {overall}")
            
            st.progress(sentiment.get('score', 0.5))
            st.markdown(f"• 신뢰도: {sentiment.get('confidence', '보통')}")
        
        # 5. 리스크 평가
        st.subheader("⚠️ 리스크 평가")
        risk = analysis.risk_assessment
        risk_level = risk.get('level', '보통')
        
        if risk_level in ['매우 높음', '높음']:
            st.error(f"🔴 리스크 수준: {risk_level}")
        elif risk_level in ['매우 낮음', '낮음']:
            st.success(f"🟢 리스크 수준: {risk_level}")
        else:
            st.warning(f"🟡 리스크 수준: {risk_level}")
        
        if risk.get('factors'):
            st.markdown("**리스크 요인:**")
            for factor in risk['factors']:
                st.markdown(f"• {factor}")
        
        # 6. 투자 권장사항
        st.subheader("💡 투자 권장사항")
        for i, recommendation in enumerate(analysis.recommendations, 1):
            st.markdown(f"**{i}.** {recommendation}")
        
        # 7. 상세 분석 정보 (확장 가능)
        with st.expander("📊 상세 분석 데이터"):
            analysis_data = {
                'technical_indicators': analysis.technical_indicators,
                'volume_analysis': analysis.volume_analysis,
                'market_sentiment': analysis.market_sentiment,
                'risk_assessment': analysis.risk_assessment,
                'pattern_analysis': {
                    'pattern_count': len(analysis.pattern_analysis.get('detected_patterns', [])),
                    'primary_pattern': analysis.pattern_analysis.get('primary_pattern'),
                    'pattern_signals': analysis.pattern_analysis.get('pattern_signals', []),
                    'pattern_reliability': analysis.pattern_analysis.get('pattern_reliability', 'LOW')
                }
            }
            
            # 다중 시간대 분석 추가
            if analysis.multi_timeframe_analysis:
                analysis_data['multi_timeframe_analysis'] = {
                    'timeframe_count': len(analysis.multi_timeframe_analysis.get('timeframe_analyses', {})),
                    'summary': analysis.multi_timeframe_analysis.get('summary', {}),
                    'convergence_score': analysis.multi_timeframe_analysis.get('convergence_score', 0),
                    'conflicts': analysis.multi_timeframe_analysis.get('conflicts', [])
                }
            
            st.json(analysis_data)
        
        # 분석 시간 표시
        st.markdown(f"*분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
    except Exception as e:
        st.error(f"체계적 가격 분석 중 오류 발생: {str(e)}")
        logger.error(f"Systematic price analysis error: {str(e)}")
        
        # 기본 분석으로 폴백
        st.subheader("💡 기본 변동 원인 분석")
        reasons = news_api.analyze_price_movement(market, price_change)
        for i, reason in enumerate(reasons, 1):
            st.markdown(f"**{i}.** {reason}")

def render_market_sentiment(news_api: CryptoNewsAPI, market: str):
    """시장 감정 분석 렌더링"""
    try:
        st.subheader("🎯 시장 감정 분석")
        
        # 감정 분석 가져오기
        sentiment_data = news_api.get_market_sentiment(market)
        
        # 감정 점수 표시
        sentiment_score = sentiment_data.get('score', 0.5)
        sentiment = sentiment_data.get('sentiment', 'neutral')
        
        # 감정 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("감정 점수", f"{sentiment_score:.1f}/1.0")
            
            # 감정 상태 표시
            if sentiment == 'positive':
                st.success("😊 긍정적")
            elif sentiment == 'negative':
                st.error("😰 부정적")
            else:
                st.info("😐 중립")
        
        with col2:
            # 감정 차트 (간단한 프로그레스 바)
            st.progress(sentiment_score)
            
            if sentiment_score > 0.7:
                st.success("매우 긍정적")
            elif sentiment_score > 0.6:
                st.info("긍정적")
            elif sentiment_score > 0.4:
                st.warning("중립")
            else:
                st.error("부정적")
        
        # 감정 분석 근거
        st.subheader("📋 분석 근거")
        
        reasons = sentiment_data.get('reasons', [])
        for i, reason in enumerate(reasons, 1):
            st.markdown(f"**{i}.** {reason}")
        
        # 투자 조언
        st.subheader("💼 투자 관점")
        
        if sentiment_score > 0.6:
            st.success("**매수 관점**: 시장 심리가 긍정적입니다.")
        elif sentiment_score < 0.4:
            st.error("**매도 관점**: 시장 심리가 부정적입니다.")
        else:
            st.info("**관망 관점**: 시장이 불확실합니다.")
        
        # 주의사항
        st.warning("⚠️ 이 분석은 참고용이며, 투자 결정은 신중하게 하시기 바랍니다.")
        
    except Exception as e:
        st.error(f"시장 감정 분석 중 오류 발생: {str(e)}")
        logger.error(f"Market sentiment error: {str(e)}")

def render_news_ticker(news_api: CryptoNewsAPI):
    """뉴스 티커 (상단 스크롤)"""
    try:
        # 최신 뉴스 3개 가져오기
        news_items = news_api.get_crypto_news(limit=3)
        
        if news_items:
            # 티커 스타일
            ticker_text = " | ".join([f"📰 {news['title']}" for news in news_items])
            
            # 스크롤링 텍스트 효과
            st.markdown(
                f"""
                <div style="
                    background-color: #f0f2f6;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    overflow: hidden;
                    white-space: nowrap;
                ">
                    <marquee behavior="scroll" direction="left" scrollamount="3">
                        {ticker_text}
                    </marquee>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    except Exception as e:
        logger.error(f"News ticker error: {str(e)}")
        pass  # 티커 오류는 무시