"""
시장 분석 컴포넌트 테스트
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import streamlit as st

# 프로젝트 루트 디렉토리를 파이썬 패스에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ui.components.analysis import (
    calculate_signal_strength,
    predict_optimal_timing,
    render_market_analysis
)

@pytest.fixture
def sample_market_data():
    """테스트용 시장 데이터 생성"""
    # 샘플 시계열 데이터 생성
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    n = len(dates)
    
    # 기본 OHLCV 데이터
    ohlcv_df = pd.DataFrame({
        'open': np.random.normal(100, 10, n),
        'high': np.random.normal(105, 10, n),
        'low': np.random.normal(95, 10, n),
        'close': np.random.normal(100, 10, n),
        'volume': np.random.normal(1000, 100, n)
    }, index=dates)
    
    # 이동평균선
    ma_data = {
        'MA5': ohlcv_df['close'].rolling(window=5).mean(),
        'MA10': ohlcv_df['close'].rolling(window=10).mean(),
        'MA20': ohlcv_df['close'].rolling(window=20).mean(),
        'MA60': ohlcv_df['close'].rolling(window=60).mean(),
        'MA120': ohlcv_df['close'].rolling(window=120).mean()
    }
    
    # RSI
    delta = ohlcv_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 볼린저 밴드
    bb_period = 20
    bb_std = 2
    middle_band = ohlcv_df['close'].rolling(window=bb_period).mean()
    std = ohlcv_df['close'].rolling(window=bb_period).std()
    upper_band = middle_band + (std * bb_std)
    lower_band = middle_band - (std * bb_std)
    
    # MACD
    exp1 = ohlcv_df['close'].ewm(span=12, adjust=False).mean()
    exp2 = ohlcv_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    
    # Stochastic
    n = 14
    low_min = ohlcv_df['low'].rolling(window=n).min()
    high_max = ohlcv_df['high'].rolling(window=n).max()
    k_fast = 100 * (ohlcv_df['close'] - low_min) / (high_max - low_min)
    d_fast = k_fast.rolling(window=3).mean()
    k_slow = d_fast
    d_slow = k_slow.rolling(window=3).mean()
    
    return {
        'current_price': float(ohlcv_df['close'].iloc[-1]),
        'ohlcv_df': ohlcv_df,
        'volume': ohlcv_df['volume'].tolist(),
        'volatility': float(ohlcv_df['close'].pct_change().std() * 100),
        'indicators': {
            'moving_averages': ma_data,
            'rsi': rsi,
            'bollinger_bands': {
                'upper': upper_band,
                'middle': middle_band,
                'lower': lower_band
            },
            'macd': {
                'macd': macd_line,
                'signal': signal_line,
                'hist': histogram
            },
            'stochastic': {
                'fast': {'k': k_fast, 'd': d_fast},
                'slow': {'k': k_slow, 'd': d_slow}
            }
        }
    }

@pytest.fixture
def empty_market_data():
    """빈 시장 데이터"""
    return {
        'current_price': 0,
        'ohlcv_df': pd.DataFrame(),
        'volume': [],
        'volatility': 0,
        'indicators': {}
    }

@pytest.fixture
def invalid_market_data():
    """잘못된 형식의 시장 데이터"""
    return {
        'current_price': None,
        'ohlcv_df': None,
        'volume': None,
        'volatility': None,
        'indicators': None
    }

def test_calculate_signal_strength_with_valid_data(sample_market_data):
    """유효한 데이터로 신호 강도 계산 테스트"""
    signal_strength, signal_details = calculate_signal_strength(sample_market_data['indicators'])
    
    # 신호 강도가 -1과 1 사이인지 확인
    assert -1 <= signal_strength <= 1
    
    # 신호 상세 내용이 리스트인지 확인
    assert isinstance(signal_details, list)
    
    # 신호 상세 내용이 문자열로 구성되어 있는지 확인
    assert all(isinstance(detail, str) for detail in signal_details)

def test_calculate_signal_strength_with_empty_data(empty_market_data):
    """빈 데이터로 신호 강도 계산 테스트"""
    signal_strength, signal_details = calculate_signal_strength(empty_market_data['indicators'])
    
    # 신호가 중립이어야 함
    assert signal_strength == 0
    assert len(signal_details) == 0

def test_calculate_signal_strength_with_invalid_data(invalid_market_data):
    """잘못된 데이터로 신호 강도 계산 테스트"""
    strength, details = calculate_signal_strength(invalid_market_data['indicators'])
    assert strength == 0
    assert len(details) == 0

def test_predict_optimal_timing_with_valid_data(sample_market_data):
    """유효한 데이터로 최적 매매 시점 예측 테스트"""
    prediction = predict_optimal_timing(sample_market_data['indicators'])
    
    # 예측 결과가 문자열인지 확인
    assert isinstance(prediction, str)
    assert len(prediction) > 0
    assert "오류" not in prediction.lower()

def test_predict_optimal_timing_with_empty_data(empty_market_data):
    """빈 데이터로 최적 매매 시점 예측 테스트"""
    prediction = predict_optimal_timing(empty_market_data['indicators'])
    assert "충분한 데이터가 없습니다" in prediction

def test_predict_optimal_timing_with_invalid_data(invalid_market_data):
    """잘못된 데이터로 최적 매매 시점 예측 테스트"""
    prediction = predict_optimal_timing(invalid_market_data['indicators'])
    assert "충분한 데이터가 없습니다" in prediction

def test_render_market_analysis_with_valid_data(sample_market_data):
    """유효한 데이터로 시장 분석 렌더링 테스트"""
    with patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.columns') as mock_columns, \
         patch('streamlit.expander') as mock_expander, \
         patch('streamlit.metric') as mock_metric, \
         patch('streamlit.warning') as mock_warning, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.write') as mock_write, \
         patch('streamlit.info') as mock_info:
        
        mock_col1, mock_col2 = MagicMock(), MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]
        mock_expander_instance = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_expander_instance
        
        market = "KRW-BTC"
        render_market_analysis(market, sample_market_data)
        
        # 기본 컴포넌트 호출 확인
        assert mock_markdown.call_count > 0
        assert mock_columns.call_count > 0
        assert mock_expander.call_count > 0
        assert mock_metric.call_count > 0
        
        # 에러나 경고가 없어야 함
        assert mock_error.call_count == 0
        
        # 일부 데이터는 경고를 표시할 수 있음
        if mock_warning.call_count > 0:
            warning_messages = [call.args[0] for call in mock_warning.call_args_list]
            assert not any("시장 데이터를 불러올 수 없습니다" in msg for msg in warning_messages)
            assert not any("현재가 데이터를 불러올 수 없습니다" in msg for msg in warning_messages)

def test_render_market_analysis_with_empty_data(empty_market_data):
    """빈 데이터로 시장 분석 렌더링 테스트"""
    with patch('streamlit.warning') as mock_warning:
        render_market_analysis("KRW-XRP", empty_market_data)
        assert mock_warning.call_count > 0

def test_render_market_analysis_with_invalid_data(invalid_market_data):
    """잘못된 데이터로 시장 분석 렌더링 테스트"""
    with patch('streamlit.warning') as mock_warning:
        render_market_analysis("KRW-XRP", invalid_market_data)
        assert mock_warning.call_count > 0
        warning_message = mock_warning.call_args_list[0][0][0]
        assert any(msg in warning_message for msg in [
            "시장 데이터를 불러올 수 없습니다",
            "현재가 데이터를 불러올 수 없습니다",
            "기술적 지표 데이터를 불러올 수 없습니다"
        ])

def test_signal_strength_edge_cases(sample_market_data):
    """신호 강도 계산의 엣지 케이스 테스트"""
    indicators = sample_market_data['indicators']
    
    # RSI 과매도 테스트
    indicators['rsi'].iloc[-1] = 10
    strength, details = calculate_signal_strength(indicators)
    assert strength > 0  # 매수 신호
    
    # RSI 과매수 테스트
    indicators['rsi'].iloc[-1] = 90
    indicators['moving_averages']['MA5'].iloc[-1] = 80  # 하락 추세 추가
    indicators['moving_averages']['MA20'].iloc[-1] = 100
    strength, details = calculate_signal_strength(indicators)
    assert strength < 0  # 매도 신호
    
    # 이동평균선 크로스 테스트
    indicators['moving_averages']['MA5'].iloc[-1] = 100
    indicators['moving_averages']['MA20'].iloc[-1] = 90
    strength, details = calculate_signal_strength(indicators)
    assert any("상향 돌파" in detail for detail in details)
    
    indicators['moving_averages']['MA5'].iloc[-1] = 80
    indicators['moving_averages']['MA20'].iloc[-1] = 90
    strength, details = calculate_signal_strength(indicators)
    assert any("하향 돌파" in detail for detail in details)

def test_optimal_timing_scenarios(sample_market_data):
    """최적 매매 시점 예측 시나리오 테스트"""
    try:
        indicators = sample_market_data['indicators']
        
        # 강한 매수 신호 시나리오
        indicators['rsi'] = pd.Series([20] * len(indicators['rsi'].index), index=indicators['rsi'].index)
        indicators['moving_averages']['MA5'] = pd.Series([110] * len(indicators['moving_averages']['MA5'].index), 
                                                        index=indicators['moving_averages']['MA5'].index)
        indicators['moving_averages']['MA20'] = pd.Series([100] * len(indicators['moving_averages']['MA20'].index), 
                                                         index=indicators['moving_averages']['MA20'].index)
        prediction = predict_optimal_timing(indicators)
        assert "매수" in prediction.lower() or "상승" in prediction.lower()
        
        # 강한 매도 신호 시나리오
        indicators['rsi'] = pd.Series([80] * len(indicators['rsi'].index), index=indicators['rsi'].index)
        indicators['moving_averages']['MA5'] = pd.Series([90] * len(indicators['moving_averages']['MA5'].index), 
                                                        index=indicators['moving_averages']['MA5'].index)
        indicators['moving_averages']['MA20'] = pd.Series([100] * len(indicators['moving_averages']['MA20'].index), 
                                                         index=indicators['moving_averages']['MA20'].index)
        prediction = predict_optimal_timing(indicators)
        assert "매도" in prediction.lower() or "하락" in prediction.lower()
        
        # 중립 시나리오
        indicators['rsi'] = pd.Series([50] * len(indicators['rsi'].index), index=indicators['rsi'].index)
        indicators['moving_averages']['MA5'] = pd.Series([100] * len(indicators['moving_averages']['MA5'].index), 
                                                        index=indicators['moving_averages']['MA5'].index)
        indicators['moving_averages']['MA20'] = pd.Series([100] * len(indicators['moving_averages']['MA20'].index), 
                                                         index=indicators['moving_averages']['MA20'].index)
        prediction = predict_optimal_timing(indicators)
        assert any(word in prediction.lower() for word in ["관망", "중립", "추가 관찰"])
        
    except Exception as e:
        pytest.fail(f"테스트 실행 중 오류 발생: {str(e)}")

def test_volume_analysis(sample_market_data):
    """거래량 분석 테스트"""
    try:
        # OHLCV 데이터 복사
        ohlcv_df = sample_market_data['ohlcv_df'].copy()
        
        # 거래량 급증 시나리오
        volume_data = ohlcv_df['volume']
        avg_volume = volume_data.mean()
        ohlcv_df.loc[ohlcv_df.index[-1], 'volume'] = avg_volume * 3  # 평균 대비 3배
        
        # 거래량 지표 계산
        volume_indicators = TechnicalAnalysis.calculate_volume_indicators(ohlcv_df)
        
        # 지표 형식 검증
        assert isinstance(volume_indicators, dict), "거래량 지표가 딕셔너리 형식이어야 합니다"
        assert all(key in volume_indicators for key in ['obv', 'ad', 'adosc']), "필수 거래량 지표가 누락되었습니다"
        
        # NaN 값 검증
        for key, indicator in volume_indicators.items():
            assert isinstance(indicator, pd.Series), f"{key} 지표가 Series 형식이어야 합니다"
            assert not indicator.isna().all(), f"{key} 지표가 모두 NaN입니다"
            assert not indicator.empty, f"{key} 지표가 비어있습니다"
        
        # 거래량 급감 시나리오
        ohlcv_df.loc[ohlcv_df.index[-1], 'volume'] = avg_volume * 0.3  # 평균 대비 30%
        
        # 거래량 지표 재계산
        volume_indicators = TechnicalAnalysis.calculate_volume_indicators(ohlcv_df)
        
        # 지표 형식 재검증
        assert isinstance(volume_indicators, dict), "거래량 지표가 딕셔너리 형식이어야 합니다"
        assert all(key in volume_indicators for key in ['obv', 'ad', 'adosc']), "필수 거래량 지표가 누락되었습니다"
        
        # NaN 값 재검증
        for key, indicator in volume_indicators.items():
            assert isinstance(indicator, pd.Series), f"{key} 지표가 Series 형식이어야 합니다"
            assert not indicator.isna().all(), f"{key} 지표가 모두 NaN입니다"
            assert not indicator.empty, f"{key} 지표가 비어있습니다"
        
        # 메모리 정리
        del ohlcv_df
        del volume_indicators
        
    except Exception as e:
        pytest.fail(f"거래량 분석 테스트 실패: {str(e)}")

def test_risk_assessment(sample_market_data):
    """위험도 평가 테스트"""
    try:
        # OHLCV 데이터 복사
        ohlcv_df = sample_market_data['ohlcv_df'].copy()
        
        # 고변동성 시나리오 생성 (30% 이상)
        close_prices = ohlcv_df['close']
        std_dev = close_prices.std()
        ohlcv_df.loc[ohlcv_df.index[-5:], 'close'] = close_prices.iloc[-5:] * (1 + np.random.normal(0.3, 0.1, 5))
        
        # 변동성 계산 및 검증
        volatility = (close_prices.pct_change().std() * np.sqrt(252)) * 100
        assert volatility > 30, f"변동성이 30% 이상이어야 합니다. 현재: {volatility:.2f}%"
        
        # 기술적 지표 재계산
        technical_indicators = TechnicalAnalysis.calculate_technical_indicators(ohlcv_df)
        
        # 예측 실행 및 검증
        prediction = TechnicalAnalysis.predict_optimal_timing(technical_indicators)
        assert isinstance(prediction, str) and prediction, "예측 결과가 비어있지 않은 문자열이어야 합니다"
        assert "높은 변동성" in prediction, "고변동성 상황에서 경고 메시지가 포함되어야 합니다"
        
        # 저변동성 시나리오 생성 (10% 미만)
        ohlcv_df = sample_market_data['ohlcv_df'].copy()
        close_prices = ohlcv_df['close']
        ohlcv_df.loc[ohlcv_df.index[-5:], 'close'] = close_prices.iloc[-5:] * (1 + np.random.normal(0.05, 0.02, 5))
        
        # 변동성 계산 및 검증
        volatility = (close_prices.pct_change().std() * np.sqrt(252)) * 100
        assert volatility < 10, f"변동성이 10% 미만이어야 합니다. 현재: {volatility:.2f}%"
        
        # 기술적 지표 재계산
        technical_indicators = TechnicalAnalysis.calculate_technical_indicators(ohlcv_df)
        
        # 예측 실행 및 검증
        prediction = TechnicalAnalysis.predict_optimal_timing(technical_indicators)
        assert isinstance(prediction, str) and prediction, "예측 결과가 비어있지 않은 문자열이어야 합니다"
        assert "높은 변동성" not in prediction, "저변동성 상황에서는 변동성 경고가 없어야 합니다"
        
        # 메모리 정리
        del ohlcv_df
        del technical_indicators
        del close_prices
        
    except Exception as e:
        pytest.fail(f"위험도 평가 테스트 실패: {str(e)}") 