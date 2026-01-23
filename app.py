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
import time
from PIL import Image, ImageOps

# ==========================================
# 設定：APIキー
# ==========================================
# ★★★ ここに新しいAPIキーを貼り付けてください ★★★
API_KEY = "AIzaSyB1chpD8a-KlJj81rhuWwRoCmZ2DiR2zeU"

# ==========================================
# 0. 日本語フォント設定
# ==========================================
@st.cache_resource
def get_japanese_font_prop():
    font_filename = "NotoSansJP-Regular.ttf"
    font_url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
    try:
        if not os.path.exists(font_filename):
            urllib.request.urlretrieve(font_filename, font_filename) # 修正: URL引数順序
            urllib.request.urlretrieve(font_url, font_filename)
        fm.fontManager.addfont(font_filename)
        plt.rcParams['font.family'] = 'Noto Sans JP'
        return fm.FontProperties(fname=font_filename)
    except Exception:
        return None

# ==========================================
# 1. AI読み取りエンジン (モデル自動検出機能付き)
# ==========================================
def get_valid_model_name():
    """
    Googleのサーバーに問い合わせて、このAPIキーで『確実に使えるモデル名』を取得する。
    名前当てクイズを回避する最強の手段。
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "error" in data:
            return None, f"キーエラー: {data['error']['message']}"
            
        # 画像認識(generateContent)に対応しているモデルだけを抜き出す
        valid_models = []
        if 'models' in data:
            for m in data['models']:
                if 'supportedGenerationMethods' in m and 'generateContent' in m['supportedGenerationMethods']:
                    # 実験版(exp)は不安定なので除外、安定版を優先
                    name = m['name'].replace('models/', '')
                    valid_models.append(name)
        
        if not valid_models:
            return None, "使用可能なモデルが見つかりませんでした。"

        # 優先順位: Flash -> Pro -> その他
        # Flashが一番速くて制限にかかりにくいので最優先
        for m in valid_models:
            if "flash" in m and "exp" not in m: return m, None
        for m in valid_models:
            if "pro" in m and "exp" not in m: return m, None
            
        # どうしてもなければリストの先頭を使う
        return valid_models[0], None
        
    except Exception as e:
        return None, f"モデル一覧取得エラー: {e}"

def analyze_image(img_bytes):
    # ★ステップ1: 使えるモデルを自動検出
    model_name, error = get_valid_model_name()
    if not model_name:
        return None, error

    # ★ステップ2: そのモデルを使って解析
    base64_data = base64.b64encode(img_bytes).decode('utf-8')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
    
    prompt = """
    Analyze the running record sheet image.
    Return JSON ONLY.
    
    JSON Structure:
    {
      "name": "Student Name (or '選手')",
      "long_run_dist": 4050,  // Integer (meters) from 15min/12min run section. 0 if empty.
      "time_trial_laps": [65, 68, 70] // Array of numbers (seconds) from 3000m/2100m section.
    }
    """
    
    headers = {'Content-Type': 'application/json'}
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_data}}]}],
        "safetySettings": safety_settings
    }
    
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        result
