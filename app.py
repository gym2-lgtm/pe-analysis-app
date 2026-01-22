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
# 設定：新しいAPIキー
# ==========================================
API_KEY = "AIzaSyBk5RvAlljh3UbdoXUUn941_w0pOrsSgKc"

# ==========================================
# 0. 日本語フォント設定
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
# 1. AI読み取りエンジン (モデル名修正版)
# ==========================================
def analyze_image_with_direct_api(img_bytes):
    # 画像を文字データ(Base64)に変換
    base64_data = base64.b64encode(img_bytes).decode('utf-8')
    
    # ★修正点: モデル名を 'gemini-1.5-flash-latest' に変更して特定
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"
    
    headers = {'Content-Type': 'application/json'}
    
    prompt_text = """
    持久走の記録用紙を読み取ってください。
    
    【ルール】
    - 男子3000m、女子2100m。
    - 名前、性別("男子"or"女子")、完走距離(m)、全ラップタイム(秒)を抽出。
    - 分秒表記(1'20)は秒(80)に変換。
    
    回答は以下のJSON形式のみで出力してください。
    {"name": "名前", "gender": "男子", "distances": [3000], "laps": [70, 72]}
    """
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt_text},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_data}}
            ]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        
        # エラーハンドリング
        if "error" in result:
            return None, f"AIエラー: {result['error']['message']}"
            
        if 'candidates' not in result:
             return None, "解析できませんでした。"

        text = result['candidates'][0]['content']['parts'][0]['text']
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0)), None
        else:
            return None, "データを読み取れませんでした。"
            
    except Exception as e:
        return None, f"通信エラー: {e}"

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
        
        pred_time = sum(laps)
        if total_dist < self.target_dist:
            remaining = self.target_dist - total_dist
            lap_dist = total_dist / len(laps) if len(laps) > 0 else 0
            if lap_dist > 0:
                pred_time += (remaining / lap_dist) * avg_pace * 1.05

        m, s = divmod(pred_time, 60)
        advice = f"【{self.target_dist}m 予測】{int(m)}分{int(s):02d}秒\n"
        if at_point: advice += f"⚠️ {at_point}周目にAT値到達（ペースダウン）\n"
        else: advice += "✅ 最後まで安定した走りです！\n"
        
        return advice, at_point

# ==========================================
# 3. レポート描画
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_image(data):
        plt.close('all')
        setup_japanese_font()
        
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
    st.write("記録用紙の写真をアップロードしてください。")
    
    uploaded_file = st.file_uploader("写真を撮る", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        with st.spinner("AI分析中..."):
            try:
                # 画像処理 (回転対応)
                image = Image.open(uploaded_file)
                image = ImageOps.exif_transpose(image)
                st.image(image, caption="送信画像", width=200)
                
                # 直通API用にバイトデータに変換（JPEG強制指定）
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                # AI解析実行
                data, error = analyze_image_with_direct_api(img_bytes)
                
                if data:
                    img_buf = ReportGenerator.create_image(data)
                    if img_buf:
                        st.success("完了！")
                        st.image(img_buf, use_column_width=True)
                        st.markdown("画像を長押しで保存できます")
                    else:
                        st.error("レポート生成エラー")
                else:
                    st.error(error)
            except Exception as e:
                st.error(f"システムエラー: {e}")

if __name__ == "__main__":
    main()
