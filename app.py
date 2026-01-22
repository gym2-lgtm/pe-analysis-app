import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import requests
import json
import re
import os
import matplotlib.font_manager as fm
import urllib.request
import base64
from PIL import Image, ImageOps

# ==========================================
# 設定：新しいAPIキー (更新済み)
# ==========================================
API_KEY = "AIzaSyBk5RvAlljh3UbdoXUUn941_w0pOrsSgKc"

# ==========================================
# 0. 日本語フォント設定 (自給自足版)
# ==========================================
def setup_japanese_font():
    font_path = "NotoSansJP-Regular.ttf"
    font_url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
    try:
        if not os.path.exists(font_path):
            urllib.request.urlretrieve(font_url, font_path)
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Noto Sans JP'
    except:
        pass

# ==========================================
# 1. AI読み取りエンジン (直通電話版)
# ==========================================
def analyze_image_with_direct_api(img_bytes):
    # 画像を文字データ(Base64)に変換
    base64_data = base64.b64encode(img_bytes).decode('utf-8')
    
    # 直通アドレス (Gemini 1.5 Flash)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
    
    headers = {'Content-Type': 'application/json'}
    
    # AIへの命令文
    prompt_text = """
    持久走の記録用紙を読み取ってください。
    
    【ルール】
    - 男子3000m、女子2100m。
    - 名前、性別("男子"or"女子")、完走距離(m)、全ラップタイム(秒)を抽出。
    - 分秒表記(1'20)は秒(80)に変換。
    
    回答は以下のJSON形式のみで出力してください。Markdownなどの装飾は不要です。
    {"name": "名前", "gender": "男子", "distances": [3000], "laps": [70, 72]}
    """
    
    # データパック作成
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt_text},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_data}}
            ]
        }]
    }
    
    try:
        # 直通電話をかける (POST送信)
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        
        # エラーチェック
        if "error" in result:
            return None, f"AIエラー: {result['error']['message']}"
            
        if 'candidates' not in result:
             return None, "解析できませんでした。画像が鮮明か確認してください。"

        # 答えを取り出す
        text = result['candidates'][0]['content']['parts'][0]['text']
