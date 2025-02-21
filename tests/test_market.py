"""
마켓 데이터 렌더링 테스트
"""

import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from src.ui.components.market import render_market_data

def test_render_market_data_with_complete_data(sample_market_data):
    """완전한 데이터로 렌더링 테스트"""
    with patch('streamlit.columns') as mock_columns, \
         patch('streamlit.metric') as mock_metric, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.warning') as mock_warning:
        
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        render_market_data(sample_market_data)
        
        assert mock_error.call_count == 0
        assert mock_warning.call_count == 0
        assert mock_metric.call_count > 0

def test_render_market_data_with_missing_data(empty_market_data):
    """일부 데이터가 누락된 경우 테스트"""
    with patch('streamlit.warning') as mock_warning:
        render_market_data(empty_market_data)
        assert mock_warning.call_count == 1
        warning_message = mock_warning.call_args_list[0][0][0]
        assert "일부 시장 데이터가 누락되었습니다" in warning_message

def test_render_market_data_with_invalid_data(invalid_market_data):
    """잘못된 형식의 데이터로 테스트"""
    with patch('streamlit.warning') as mock_warning, \
         patch('streamlit.error') as mock_error:
        
        render_market_data(invalid_market_data)
        assert mock_warning.call_count > 0 or mock_error.call_count > 0

def test_render_market_data_with_zero_values(empty_market_data):
    """0값을 포함한 데이터로 테스트"""
    market_data = empty_market_data.copy()
    market_data.update({
        'current_price': 0,
        'open': 0,
        'high': 0,
        'low': 0,
        'volume': 0,
        'change_rate': 0
    })
    
    with patch('streamlit.columns') as mock_columns, \
         patch('streamlit.metric') as mock_metric, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.warning') as mock_warning:
        
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        render_market_data(market_data)
        
        assert mock_error.call_count == 0
        assert mock_warning.call_count == 0
        assert mock_metric.call_count > 0 