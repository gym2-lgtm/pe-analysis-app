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
# 設定：APIキー
# ==========================================
# ★ステップ1: ここを「新しく作り直したキー」に差し替えてください
API_KEY = "AIzaSyAM8y4fI6X_-HB6xJ_FsHK3AHImPraqbHw"

# 通信の安定化設定
genai.configure(api_key=API_KEY, transport='rest')

# ==========================================
# 1. AI読み取りエンジン
# ==========================================
def analyze_image_with_gemini(img_obj):
    # ★ステップ2: 最も標準的なモデル名 'gemini-1.5-flash' を使用
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = """
    持久走の記録用紙を読み取ってください。
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
            return None, "データを読み取れませんでした。AIの回答: " + text
    except Exception as e:
        # ここでエラーの詳細を表示するように変更
        return None, f"通信エラーが発生しました。新しいAPIキーを試してください。\n詳細: {e}"

# (以下の ScienceEngine, ReportGenerator, main は変更なしでOKです)
