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
API_KEY = "AIzaSyATM7vIfyhj6vKsZga3fydYLHvAMRVNdzg"

# ==========================================
# 0. 日本語フォント設定 (japanize-matplotlibの代わり)
# ==========================================
def setup_japanese_font():
    # 日本語フォント(NotoSansJP)をダウンロードして適用する
    font_path = "NotoSansJP-Regular.ttf"
    font_url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
    
    try:
        if not os.path.exists(font_path):
            with st.spinner("日本語フォントを準備中..."):
                urllib.request.urlretrieve(font_url, font_path)
        
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Noto Sans JP'
    except Exception as e:
        # 失敗したら英語フォントのまま進める（エラーで止まるよりマシ）
        st.warning(f"フォント設定エラー: {e}")

# ==========================================
# 1. AI読み取りエンジン
# ==========================================
def analyze_image_with_gemini(img_obj):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = """
    持久走の記録用紙を読み取ってください。
    
    【距離設定】男子3000m、女子2100m。
    【抽出項目】
    1. 名前 (name): 読めなければ "あなた"
    2. 性別 (gender): "男子" or "女子"
    3. 距離 (distances): 完走距離のリスト
    4. ラップ (laps): 1周ごとのタイム(秒)のリスト
       - 分秒表記(1'20)は秒(80)に変換
       - 累積タイムなら引き算して計算
       
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
            return None, "データを読み取れませんでした。"
    except Exception as e:
        return None, f"エラー: {e}"

# ==========================================
# 2. 分析ロジック
# ==========================================
class ScienceEngine:
    def __init__(self, gender="男子"):
        self.gender = gender
        self.target_dist = 2100 if gender == "女子" else 3000

    def analyze(self, laps, total_dist):
        if not laps: return "", None
        laps_np = np.array(laps)
        avg_pace = np.mean(laps_np)
        
        at_point = None
        for i in range(1, len(laps)):
            if laps[i] - laps[i-1] >= 3.0:
                at_point = i + 1
                break
        
        current_time = sum(laps)
        pred_time = current_time
        if total_dist < self.target_dist:
            remaining = self.target_dist - total_dist
            lap_dist = total_dist / len(laps) if len(laps) > 0 else 0
            if lap_dist > 0:
                pred_time += (remaining / lap_dist) * avg_pace * 1.05

        m, s = divmod(pred_time, 60)
        advice = f"【{self.target_dist}m 予測】{int(m)}分{int(s):02d}秒\n"
        if at_point: advice += f"⚠️ {at_point}周目にペースダウン（AT値）\n"
        else: advice += "✅ 安定したペース配分です！\n"
        
        return advice, at_point

# ==========================================
# 3. レポート描画
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_image(data):
        plt.close('all')
        setup_japanese_font() # ★ここでフォント設定を実行
        
        try:
            name = data.get("name", "あなた")
            gender = data.get("gender", "男子")
            laps = data.get("laps", [])
            if isinstance(laps, str): laps = [float(x) for x in re.findall(r"[\d\.]+", laps)]
            dists = data.get("distances", [3000])
            if isinstance(dists, str): dists = [float(x) for x in re.findall(r"[\d\.]+", dists)]
            total_dist = max(dists) if dists else 3000
        except: return None

        if not laps: return None
        engine = ScienceEngine(gender)
        advice, at_point = engine.analyze(laps, total_dist)
        
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100, facecolor='white')
        plt.axis('off')
        fig.text(0.5, 0.95, f"{name}さんの分析レポート", fontsize=24, ha='center', weight='bold')
        
        ax1 = fig.add_axes([0.1, 0.75, 0.8, 0.15])
        ax1.set_axis_off(); ax1.add_patch(plt.Rectangle((0,0),1,1,color='#E6F3FF',transform=ax1.transAxes))
        m, s = divmod(sum(laps), 60)
        ax1.text(0.5, 0.5, f"距離: {total_dist}m\nタイム: {int(m)}分{int(s):02d}秒", fontsize=18, ha='center', va='center')

        ax2 = fig.add_axes([0.1, 0.45, 0.8, 0.25])
        ax2.plot(range(1, len(laps)+1), laps, marker='o', linewidth=3, color='#FF6B6B')
        ax2.set_title("ラップ推移"); ax2.grid(True, linestyle='--', alpha=0.5)
        if at_point: ax2.axvline(x=at_point, color='blue', linestyle='--', label='AT値'); ax2.legend()

        ax3 = fig.add_axes([0.1, 0.10, 0.8, 0.30])
        ax3.set_axis_off(); ax3.add_patch(plt.Rectangle((0,0),1,1,fill=False,edgecolor='#333',linewidth=2,transform=ax3.transAxes))
        ax3.text(0.05, 0.5, advice, fontsize=14, va='center')

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

# ==========================================
# 4. アプリUI
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
                    img_buf = ReportGenerator.create_image(data)
                    if img_buf:
                        st.image(img_buf, use_column_width=True)
                        st.markdown("画像を長押しで保存")
                else:
                    st.error(error)
            except Exception as e:
                st.error(f"エラー: {e}")

if __name__ == "__main__":
    main()
