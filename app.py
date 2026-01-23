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
            urllib.request.urlretrieve(font_url, font_filename)
        fm.fontManager.addfont(font_filename)
        plt.rcParams['font.family'] = 'Noto Sans JP'
        return fm.FontProperties(fname=font_filename)
    except Exception:
        return None

# ==========================================
# 1. AI読み取りエンジン (モデル自動検出機能)
# ==========================================
def get_valid_model_name():
    """Googleのサーバーに問い合わせて、使えるモデル名を自動取得する"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "error" in data:
            return None, f"キーエラー: {data['error']['message']}"
            
        valid_models = []
        if 'models' in data:
            for m in data['models']:
                if 'supportedGenerationMethods' in m and 'generateContent' in m['supportedGenerationMethods']:
                    name = m['name'].replace('models/', '')
                    valid_models.append(name)
        
        if not valid_models:
            return None, "使用可能なモデルが見つかりませんでした。"

        # Flash優先、次にPro
        for m in valid_models:
            if "flash" in m and "exp" not in m: return m, None
        for m in valid_models:
            if "pro" in m and "exp" not in m: return m, None
            
        return valid_models[0], None
        
    except Exception as e:
        return None, f"モデル一覧取得エラー: {e}"

def analyze_image(img_bytes):
    # ステップ1: モデル自動検出
    model_name, error = get_valid_model_name()
    if not model_name:
        return None, error

    # ステップ2: 解析リクエスト
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
        result_json = res.json()
        
        if "error" in result_json:
            return None, f"APIエラー({model_name}): {result_json['error']['message']}"
            
        if 'candidates' in result_json and len(result_json['candidates']) > 0:
            candidate = result_json['candidates'][0]
            if candidate.get('finishReason') not in ['STOP', 'MAX_TOKENS', None]:
                 return None, f"AIブロック: {candidate.get('finishReason')}"
                 
            text = candidate['content']['parts'][0]['text']
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0)), None
            else:
                return None, "データ形式エラー (JSONが見つかりませんでした)"
        else:
            return None, "AIからの応答がありませんでした"

    except Exception as e:
        return None, f"通信エラー: {str(e)}"

# ==========================================
# 2. 科学的分析ロジック
# ==========================================
class ScienceEngine:
    def __init__(self, data):
        self.name = data.get("name", "選手")
        try:
            val = data.get("long_run_dist", 0)
            self.long_run_dist = float(val) if val is not None else 0
        except: self.long_run_dist = 0
        
        laps = data.get("time_trial_laps", [])
        if not isinstance(laps, list): laps = []
        clean_laps = []
        for x in laps:
            try: clean_laps.append(float(x))
            except: pass
        self.tt_laps = np.array(clean_laps)
        
        self.is_male = True if self.long_run_dist > 3200 else False 
        self.target_dist = 3000 if self.is_male else 2100
        self.long_run_min = 15 if self.is_male else 12

    def get_potential_time(self):
        if self.long_run_dist == 0: return None
        t1 = self.long_run_min * 60
        d1 = self.long_run_dist
        d2 = self.target_dist
        return t1 * (d2 / d1)**1.06

    def get_vo2_max(self):
        if self.long_run_dist == 0: return 0
        dist_12min = self.long_run_dist * (12 / self.long_run_min)
        return max((dist_12min - 504.9) / 44.73, 0)

# ==========================================
# 3. レポート描画
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_dashboard(data):
        plt.close('all')
        fp = get_japanese_font_prop()
        font_kwargs = {'fontproperties': fp} if fp else {}

        engine = ScienceEngine(data)
        potential_sec = engine.get_potential_time()
        
        fig = plt.figure(figsize=(11.69, 8.27), dpi=100, facecolor='white')
        
        # ヘッダー
        fig.text(0.05, 0.95, f"科学的分析レポート: {engine.name}", fontsize=22, weight='bold', color='#1a237e', **font_kwargs)
        fig.text(0.05, 0.92, f"基準: {engine.long_run_min}分間走 {int(engine.long_run_dist)}m", fontsize=12, color='gray', **font_kwargs)

        # ① 左上
        ax1 = fig.add_axes([0.05, 0.60, 0.40, 0.25])
        ax1.set_axis_off()
