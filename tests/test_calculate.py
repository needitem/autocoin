"""
기술적 분석 지표 계산 테스트
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 프로젝트 루트 디렉토리를 파이썬 패스에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.calculate import TechnicalAnalysis

def test_moving_averages(sample_ohlcv_data):
    """이동평균선 계산 테스트"""
    ma_data = TechnicalAnalysis.calculate_moving_averages(sample_ohlcv_data)
    
    assert 'MA5' in ma_data
    assert 'MA10' in ma_data
    assert 'MA20' in ma_data
    assert 'MA60' in ma_data
    assert 'MA120' in ma_data
    
    # MA 값이 실제 평균과 일치하는지 확인
    assert np.allclose(
        ma_data['MA5'].iloc[4:],
        sample_ohlcv_data['close'].rolling(window=5).mean().iloc[4:],
        equal_nan=True
    )

def test_bollinger_bands(sample_ohlcv_data):
    """볼린저 밴드 계산 테스트"""
    bb = TechnicalAnalysis.calculate_bollinger_bands(sample_ohlcv_data)
    
    assert 'upper' in bb
    assert 'middle' in bb
    assert 'lower' in bb
    assert 'bandwidth' in bb
    
    # 밴드 관계 확인
    assert (bb['upper'] >= bb['middle']).all()
    assert (bb['middle'] >= bb['lower']).all()

def test_ichimoku(sample_ohlcv_data):
    """일목균형표 계산 테스트"""
    ichimoku = TechnicalAnalysis.calculate_ichimoku(sample_ohlcv_data)
    
    assert 'tenkan_sen' in ichimoku
    assert 'kijun_sen' in ichimoku
    assert 'senkou_span_a' in ichimoku
    assert 'senkou_span_b' in ichimoku
    assert 'chikou_span' in ichimoku

def test_pivot_points(sample_ohlcv_data):
    """피봇 포인트 계산 테스트"""
    pivots = TechnicalAnalysis.calculate_pivot_points(sample_ohlcv_data)
    
    assert 'pivot' in pivots
    assert 'r1' in pivots
    assert 'r2' in pivots
    assert 's1' in pivots
    assert 's2' in pivots
    
    # 피봇 레벨 관계 확인
    assert (pivots['r2'] >= pivots['r1']).all()
    assert (pivots['r1'] >= pivots['pivot']).all()
    assert (pivots['pivot'] >= pivots['s1']).all()
    assert (pivots['s1'] >= pivots['s2']).all()

def test_parabolic_sar(sample_ohlcv_data):
    """Parabolic SAR 계산 테스트"""
    sar = TechnicalAnalysis.calculate_parabolic_sar(sample_ohlcv_data)
    assert isinstance(sar, pd.Series)
    assert not sar.empty

def test_price_channels(sample_ohlcv_data):
    """가격 채널 계산 테스트"""
    channels = TechnicalAnalysis.calculate_price_channels(sample_ohlcv_data)
    
    assert 'upper' in channels
    assert 'middle' in channels
    assert 'lower' in channels
    
    # 채널 관계 확인
    assert (channels['upper'] >= channels['middle']).all()
    assert (channels['middle'] >= channels['lower']).all()

def test_disparity(sample_ohlcv_data):
    """이격도 계산 테스트"""
    disparity = TechnicalAnalysis.calculate_disparity(sample_ohlcv_data)
    assert isinstance(disparity, pd.Series)
    assert not disparity.empty

def test_cci(sample_ohlcv_data):
    """CCI 계산 테스트"""
    cci = TechnicalAnalysis.calculate_cci(sample_ohlcv_data)
    assert isinstance(cci, pd.Series)
    assert not cci.empty

def test_cmf(sample_ohlcv_data):
    """Chaikin Money Flow 계산 테스트"""
    cmf = TechnicalAnalysis.calculate_cmf(sample_ohlcv_data)
    assert isinstance(cmf, pd.Series)
    assert not cmf.empty

def test_dmi(sample_ohlcv_data):
    """DMI 계산 테스트"""
    dmi = TechnicalAnalysis.calculate_dmi(sample_ohlcv_data)
    
    assert 'plus_di' in dmi
    assert 'minus_di' in dmi
    assert 'adx' in dmi

def test_macd(sample_ohlcv_data):
    """MACD 계산 테스트"""
    macd = TechnicalAnalysis.calculate_macd(sample_ohlcv_data)
    
    assert 'macd' in macd
    assert 'signal' in macd
    assert 'hist' in macd

def test_rsi(sample_ohlcv_data):
    """RSI 계산 테스트"""
    rsi = TechnicalAnalysis.calculate_rsi(sample_ohlcv_data)
    assert isinstance(rsi, pd.Series)
    assert not rsi.empty
    assert (rsi >= 0).all() and (rsi <= 100).all()

def test_stochastic(sample_ohlcv_data):
    """Stochastic Oscillator 계산 테스트"""
    stoch = TechnicalAnalysis.calculate_stochastic(sample_ohlcv_data)
    
    assert 'fast' in stoch
    assert 'slow' in stoch
    assert 'k' in stoch['fast']
    assert 'd' in stoch['fast']
    assert 'k' in stoch['slow']
    assert 'd' in stoch['slow']

def test_trix(sample_ohlcv_data):
    """TRIX 계산 테스트"""
    trix = TechnicalAnalysis.calculate_trix(sample_ohlcv_data)
    assert isinstance(trix, pd.Series)
    assert not trix.empty

def test_volume_indicators(sample_ohlcv_data):
    """거래량 지표 계산 테스트"""
    vol = TechnicalAnalysis.calculate_volume_indicators(sample_ohlcv_data)
    
    assert 'obv' in vol
    assert 'ad' in vol
    assert 'adosc' in vol

def test_psychological_indicators(sample_ohlcv_data):
    """심리도 지표 계산 테스트"""
    psy = TechnicalAnalysis.calculate_psychological_indicators(sample_ohlcv_data)
    
    assert 'new_psychological' in psy
    assert 'investment_psychological' in psy
    assert (psy['new_psychological'] >= 0).all() and (psy['new_psychological'] <= 100).all()
    assert (psy['investment_psychological'] >= 0).all() and (psy['investment_psychological'] <= 100).all()

def test_elder_ray_index(sample_ohlcv_data):
    """Elder Ray Index 계산 테스트"""
    elder = TechnicalAnalysis.calculate_elder_ray_index(sample_ohlcv_data)
    
    assert 'ema' in elder
    assert 'bull_power' in elder
    assert 'bear_power' in elder

def test_mass_index(sample_ohlcv_data):
    """Mass Index 계산 테스트"""
    mass = TechnicalAnalysis.calculate_mass_index(sample_ohlcv_data)
    assert isinstance(mass, pd.Series)
    assert not mass.empty

def test_nco(sample_ohlcv_data):
    """Net Change Oscillator 계산 테스트"""
    nco = TechnicalAnalysis.calculate_nco(sample_ohlcv_data)
    assert isinstance(nco, pd.Series)
    assert not nco.empty

def test_nvi(sample_ohlcv_data):
    """Negative Volume Index 계산 테스트"""
    nvi = TechnicalAnalysis.calculate_nvi(sample_ohlcv_data)
    assert isinstance(nvi, pd.Series)
    assert not nvi.empty
    assert nvi.iloc[0] == 100.0  # 초기값 확인

def test_pvi(sample_ohlcv_data):
    """Positive Volume Index 계산 테스트"""
    pvi = TechnicalAnalysis.calculate_pvi(sample_ohlcv_data)
    assert isinstance(pvi, pd.Series)
    assert not pvi.empty
    assert pvi.iloc[0] == 100.0  # 초기값 확인

def test_rmi(sample_ohlcv_data):
    """Relative Momentum Index 계산 테스트"""
    rmi = TechnicalAnalysis.calculate_rmi(sample_ohlcv_data)
    assert isinstance(rmi, pd.Series)
    assert not rmi.empty

def test_rvi(sample_ohlcv_data):
    """Relative Volatility Index 계산 테스트"""
    rvi = TechnicalAnalysis.calculate_rvi(sample_ohlcv_data)
    assert isinstance(rvi, pd.Series)
    assert not rvi.empty
    assert (rvi >= 0).all() and (rvi <= 100).all()

def test_sonar(sample_ohlcv_data):
    """SONAR 계산 테스트"""
    sonar = TechnicalAnalysis.calculate_sonar(sample_ohlcv_data)
    
    assert 'sonar' in sonar
    assert 'signal' in sonar
    assert 'upper' in sonar
    assert 'lower' in sonar
    assert (sonar['upper'] >= sonar['lower']).all()

def test_stochastic_momentum(sample_ohlcv_data):
    """Stochastic Momentum Index 계산 테스트"""
    smi = TechnicalAnalysis.calculate_stochastic_momentum(sample_ohlcv_data)
    
    assert 'smi' in smi
    assert 'signal' in smi

def test_stochastic_rsi(sample_ohlcv_data):
    """Stochastic RSI 계산 테스트"""
    stoch_rsi = TechnicalAnalysis.calculate_stochastic_rsi(sample_ohlcv_data)
    
    assert 'k' in stoch_rsi
    assert 'd' in stoch_rsi
    assert (stoch_rsi['k'] >= 0).all() and (stoch_rsi['k'] <= 100).all()
    assert (stoch_rsi['d'] >= 0).all() and (stoch_rsi['d'] <= 100).all()

def test_vr(sample_ohlcv_data):
    """Volume Ratio 계산 테스트"""
    vr = TechnicalAnalysis.calculate_vr(sample_ohlcv_data)
    assert isinstance(vr, pd.Series)
    assert not vr.empty

def test_all_indicators(sample_ohlcv_data):
    """모든 지표 통합 계산 테스트"""
    all_indicators = TechnicalAnalysis.calculate_all_indicators(sample_ohlcv_data)
    
    # 필수 지표들이 포함되어 있는지 확인
    assert 'moving_averages' in all_indicators
    assert 'rsi' in all_indicators
    assert 'macd' in all_indicators
    assert 'bollinger_bands' in all_indicators
    assert 'stochastic' in all_indicators
    
    # 데이터 타입과 형식 확인
    assert isinstance(all_indicators['moving_averages'], dict)
    assert isinstance(all_indicators['rsi'], pd.Series)
    assert isinstance(all_indicators['macd'], dict)
    assert isinstance(all_indicators['bollinger_bands'], dict)
    assert isinstance(all_indicators['stochastic'], dict)
    
    # 값 범위 확인
    assert (all_indicators['rsi'] >= 0).all() and (all_indicators['rsi'] <= 100).all()
    assert (all_indicators['bollinger_bands']['upper'] >= all_indicators['bollinger_bands']['middle']).all()
    assert (all_indicators['bollinger_bands']['middle'] >= all_indicators['bollinger_bands']['lower']).all()

    # 추가 지표들이 포함되어 있는지 확인
    assert 'ichimoku' in all_indicators
    assert 'pivot_points' in all_indicators
    assert 'parabolic_sar' in all_indicators
    assert 'price_channels' in all_indicators
    assert 'disparity' in all_indicators
    assert 'cci' in all_indicators
    assert 'cmf' in all_indicators
    assert 'dmi' in all_indicators
    assert 'stochastic' in all_indicators
    assert 'trix' in all_indicators
    assert 'volume' in all_indicators
    assert 'psychological' in all_indicators
    assert 'additional' in all_indicators
    assert 'elder_ray' in all_indicators
    assert 'mass_index' in all_indicators
    assert 'nco' in all_indicators
    assert 'nvi' in all_indicators
    assert 'pvi' in all_indicators
    assert 'rmi' in all_indicators
    assert 'rvi' in all_indicators
    assert 'sonar' in all_indicators
    assert 'stochastic_momentum' in all_indicators
    assert 'stochastic_rsi' in all_indicators
    assert 'vr' in all_indicators 