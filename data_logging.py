import csv
import time
import json
import numpy as np
import pandas as pd

def get_fear_and_greed_index() -> float:
    return 41.0  # 임의 값

def fetch_latest_news() -> list:
    return ["BTC reaches 30K again", "Market remains volatile"]

def log_correlation_to_csv(timestamp: float, corr_up_bn_price: float, corr_up_bn_vol: float, corr_bn_price_bn_vol: float):
    with open("correlation_log.csv", mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, corr_up_bn_price, corr_up_bn_vol, corr_bn_price_bn_vol]) 