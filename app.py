import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import google.generativeai as genai
from PIL import Image, ImageOps
import json
import re
import os
import matplotlib.font_manager as fm
import urllib.request

# ==========================================
# 1. 設定：APIキー & 通信設定
# ==========================================
# 増本さんの新しいAPIキーをセット
API_KEY = "AIzaSyAM8y4fI6X_-HB6xJ_FsHK3AHImPraqbHw"

# 通信の安定化
try:
    genai.configure(api_key=API_KEY, transport='rest')
except Exception as e:
    st.error(f"初期設定エラー: {e}")

# ==========================================
# 2. 自動モデル選択関数
# ==========================================
def get_best_model():
    """利用可能な最新のモデルを自動で見つける"""
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # 2.0-flashがあれば優先、なければ1.5-flash、それもなければ最初に見つかったもの
        for target in ['models/gemini-2.0-flash', 'models/gemini-1.5-flash', 'models/gemini-1.5-pro']:
            if target in models:
                return target
        return models[0] if models else "gemini-1.5-flash"
    except:
        return "gemini-1.5-flash" # 取得失敗時はデフォルトを試す

# ==========================================
# 3. AI読み取りエンジン
# ==========================================
def analyze_image_with_gemini(img_obj):
    # 自動で最適なモデルを選択
    target_model = get_best_model()
    model = genai.GenerativeModel(target_model)
    
    prompt = """
    持久走の記録用紙を読み取ってください。
    名前、性別（男子/女子）、各周のラップタイム（秒）を抽出してください。
    Output JSON format only:
    {"name": "名前", "gender": "男子", "distances": [3000], "laps": [70, 72]}
    """
    
    try:
        response = model.generate_content([prompt, img_obj])
        text = response.text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0)), None
        else:
            return None, f"解析失敗。使用モデル: {target_model}\nAIの応答: {text}"
    except Exception as e:
        return None, f"通信エラー（モデル: {target_model}）: {e}\nAPIキーの権限設定を確認してください。"

# ==========================================
# 4. 日本語フォント設定
# ==========================================
def setup_japanese_font():
    font_path = "NotoSansJP-Regular.ttf"
    font_url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
    try:
        if not os.path.exists(font_path):
            urllib.request.urlretrieve(font_url, font_path)
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Noto Sans JP'
    except: pass

# ==========================================
# 5. 分析・レポートロジック（ScienceEngineなどは前回のまま）
# ==========================================
# [以前のコードの ScienceEngine, ReportGenerator クラスをここに配置]
# (文字数制限のため省略していますが、増本さんの手元のコードのままでOKです)

# ==========================================
# 6. アプリUI
# ==========================================
def main():
    st.set_page_config(page_title="持久走分析", layout="centered")
    st.title("🏃‍♂️ 持久走分析アプリ")
    
    uploaded_file = st.file_uploader("写真を撮る", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        with st.spinner("AI分析中..."):
            try:
                image = Image.open(uploaded_file)
                image = ImageOps.exif_transpose(image)
                st.image(image, caption="送信画像", width=200)
                
                data, error = analyze_image_with_gemini(image)
                if data:
                    # ReportGeneratorで画像作成（前回のクラスが必要です）
                    from __main__ import ReportGenerator
                    img_buf = ReportGenerator.create_image(data)
                    if img_buf:
                        st.image(img_buf, use_column_width=True)
                    else:
                        st.write(data) # 画像化失敗時はデータのみ表示
                else:
                    st.error(error)
            except Exception as e:
                st.error(f"実行エラー: {e}")

if __name__ == "__main__":
    main()
