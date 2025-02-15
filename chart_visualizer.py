import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ChartVisualizer:
    def __init__(self):
        self.colors = {
            'background': '#0E1117',
            'text': '#FAFAFA',
            'grid': '#1F2937',
            'up': '#26A69A',
            'down': '#EF5350',
            'line': '#2196F3'
        }
    
    def create_candlestick_chart(self, df: pd.DataFrame) -> go.Figure:
        """캔들스틱 차트 생성"""
        try:
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.03, 
                subplot_titles=('가격', '거래량'),
                row_width=[0.7, 0.3]
            )

            # 캔들스틱 추가
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='캔들',
                    increasing_line_color=self.colors['up'],
                    decreasing_line_color=self.colors['down']
                ), 
                row=1, col=1
            )

            # 거래량 바 추가
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['volume'],
                    name='거래량'
                ), 
                row=2, col=1
            )

            # 이동평균선 추가
            for period in [5, 20, 60]:
                ma = df['close'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=ma,
                        name=f'MA{period}',
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )

            fig.update_layout(
                title='가격 차트',
                yaxis_title='가격',
                yaxis2_title='거래량',
                xaxis_rangeslider_visible=False,
                height=800
            )

            return fig
            
        except Exception as e:
            logger.error(f"캔들스틱 차트 생성 중 오류: {e}")
            return None

    def create_technical_indicators_chart(self, df: pd.DataFrame, indicators: Dict) -> go.Figure:
        """기술적 지표 차트 생성"""
        try:
            # 서브플롯 생성 (3개의 행)
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('가격 & 이동평균선', 'RSI', 'MACD'),
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # 캔들스틱
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="가격",
                    increasing_line_color=self.colors['up'],
                    decreasing_line_color=self.colors['down']
                ),
                row=1, col=1
            )
            
            # 이동평균선
            ma_data = indicators.get('moving_averages', {})
            for period, values in ma_data.items():
                if isinstance(values, (pd.Series, np.ndarray)):
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=values,
                            name=f"MA{period}",
                            line=dict(width=1)
                        ),
                        row=1, col=1
                    )
            
            # RSI
            rsi_values = indicators.get('rsi')
            if rsi_values is not None:
                if isinstance(rsi_values, (float, np.float64, int)):
                    # RSI가 단일 값인 경우
                    rsi_values = pd.Series([float(rsi_values)], index=[df['timestamp'].iloc[-1]])
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=pd.Series(rsi_values),
                        name="RSI",
                        line=dict(color='#9C27B0', width=1)
                    ),
                    row=2, col=1
                )
            
            # 과매수/과매도 영역
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # MACD
            macd_data = indicators.get('macd', {})
            macd_values = macd_data.get('macd')
            signal_values = macd_data.get('signal')
            
            if macd_values is not None and signal_values is not None:
                if isinstance(macd_values, (float, np.float64, int)):
                    # MACD가 단일 값인 경우
                    macd_values = pd.Series([float(macd_values)], index=[df['timestamp'].iloc[-1]])
                    signal_values = pd.Series([float(signal_values)], index=[df['timestamp'].iloc[-1]])
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=pd.Series(macd_values),
                        name="MACD",
                        line=dict(color='#2196F3', width=1)
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=pd.Series(signal_values),
                        name="Signal",
                        line=dict(color='#FF9800', width=1)
                    ),
                    row=3, col=1
                )
            
            # 레이아웃 업데이트
            fig.update_layout(
                height=800,
                template='plotly_dark',
                showlegend=True,
                xaxis_rangeslider_visible=False,
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background'],
                font=dict(color=self.colors['text'])
            )
            
            # Y축 그리드 스타일
            fig.update_yaxes(gridcolor=self.colors['grid'])
            
            return fig
            
        except Exception as e:
            logger.error(f"기술적 지표 차트 생성 중 오류: {str(e)}")
            # 에러 시 빈 차트 반환
            fig = go.Figure()
            fig.update_layout(
                title="차트 생성 실패",
                annotations=[
                    dict(
                        text="기술적 지표 데이터를 불러오는데 실패했습니다.",
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=14)
                    )
                ]
            )
            return fig

    def create_prediction_chart(self, df: pd.DataFrame, pattern_analysis: Dict, current_price: float) -> go.Figure:
        """예측 차트 생성"""
        try:
            fig = go.Figure()
            
            # 최근 30일 데이터
            recent_df = df.tail(30).copy()
            
            # 캔들스틱 추가
            fig.add_trace(
                go.Candlestick(
                    x=recent_df['timestamp'],
                    open=recent_df['open'],
                    high=recent_df['high'],
                    low=recent_df['low'],
                    close=recent_df['close'],
                    name='실제 가격'
                )
            )
            
            # 이동평균선 추가
            for period in [5, 20]:
                ma = df['close'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=recent_df['timestamp'],
                        y=ma.tail(30),
                        name=f'MA{period}',
                        line=dict(width=1)
                    )
                )
            
            # 예측 기간 설정
            future_dates = pd.date_range(
                start=df['timestamp'].iloc[-1],
                periods=15,
                freq='D'
            )[1:]
            
            # 예측선 추가
            if 'predictions' in pattern_analysis and pattern_analysis['predictions']:
                for pred in pattern_analysis['predictions']:
                    # 낙관적 시나리오
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=np.linspace(
                                current_price,
                                pred['predicted_range']['high'],
                                len(future_dates)
                            ),
                            name=f"{pred['pattern']} 낙관적",
                            line=dict(color='rgba(0, 255, 0, 0.3)', dash='dash'),
                            fill=None
                        )
                    )
                    
                    # 비관적 시나리오
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=np.linspace(
                                current_price,
                                pred['predicted_range']['low'],
                                len(future_dates)
                            ),
                            name=f"{pred['pattern']} 비관적",
                            line=dict(color='rgba(255, 0, 0, 0.3)', dash='dash'),
                            fill='tonexty'
                        )
                    )
                    
                    # 가장 가능성 높은 시나리오
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=np.linspace(
                                current_price,
                                pred['predicted_range']['most_likely'],
                                len(future_dates)
                            ),
                            name=f"{pred['pattern']} 예상",
                            line=dict(color='rgba(255, 165, 0, 0.8)', dash='solid')
                        )
                    )
                    
                    # 주요 레벨 표시
                    for resistance in pred['key_levels']['resistance']:
                        fig.add_hline(
                            y=resistance,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"저항선 {resistance:,.0f}원"
                        )
                    
                    for support in pred['key_levels']['support']:
                        fig.add_hline(
                            y=support,
                            line_dash="dash",
                            line_color="green",
                            annotation_text=f"지지선 {support:,.0f}원"
                        )
            
            fig.update_layout(
                title='향후 14일 예상 차트',
                yaxis_title='가격',
                xaxis_title='날짜',
                height=600,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                yaxis=dict(
                    tickformat=',',
                    ticksuffix='원'
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"예측 차트 생성 중 오류: {e}")
            return None 

    def _update_layout(self, fig: go.Figure, title: str):
        """차트 레이아웃 업데이트"""
        fig.update_layout(
            title=title,
            template='plotly_dark',
            showlegend=True,
            xaxis_rangeslider_visible=False,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            yaxis=dict(gridcolor=self.colors['grid'])
        ) 