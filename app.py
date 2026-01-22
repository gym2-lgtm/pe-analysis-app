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
# 設定：APIキー (確認済み)
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
    except Exception:
        pass

# ==========================================
# 1. AI読み取りエンジン (最強のリトライ機能付き)
# ==========================================
def analyze_image_with_direct_api(img_bytes):
    base64_data = base64.b64encode(img_bytes).decode('utf-8')
    
    # ★ここが最強の修正点:
    # どれか一つでも繋がればOK！上から順にドアをノックしに行きます。
    models_to_try = [
        "gemini-1.5-flash",          # 基本
        "gemini-1.5-flash-001",      # バージョン固定
        "gemini-1.5-flash-latest",   # 最新
        "gemini-1.5-pro",            # 高性能版
        "gemini-1.5-pro-001"         # 高性能固定
    ]
    
    prompt_text = """
    持久走の記録用紙を読み取ってください。
    
    【ルール】
    - 男子3000m、女子2100m。
    - 名前、性別("男子"or"女子")、完走距離(m)、全ラップタイム(秒)を抽出。
    - 分秒表記(1'20)は秒(80)に変換。
    
    回答は以下のJSON形式のみで出力してください。Markdown装飾は不要です。
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
    
    headers = {'Content-Type': 'application/json'}
    last_error = ""
    
    # ループ処理：つながるモデルが見つかるまで試す
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
        try:
            # print(f"Testing model: {model_name}...") # デバッグ用
            response = requests.post(url, headers=headers, json=payload)
            result = response.json()
            
            # エラーがあれば次へ
            if "error" in result:
                error_msg = result['error']['message']
                # 「見つからない(Not Found)」系のエラーなら次を試す
                if "not found" in error_msg.lower() or "supported" in error_msg.lower():
                    last_error = f"{model_name} NG: {error_msg}"
                    continue
                else:
                    # 認証エラーなどは即終了
                    return None, f"AIエラー: {error_msg}" 
            
            # 成功したらデータを返す
            if 'candidates' in result:
                text = result['candidates'][0]['content']['parts'][0]['text']
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0)), None
            
        except Exception as e:
            last_error = str(e)
            continue

    return None, f"全てのモデルで失敗しました。最後のエラー: {last_error}"

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
        else: advice += "✅ ペース配分が完璧です！\n"
        
        return advice, at_point

# ==========================================
# 3. レポート描画 (ここが途切れていた部分です！)
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
            # ★ここから補完
            total_dist = max(dists) if dists else 3000
        except: return None

        if not laps: return None
        engine = ScienceEngine(gender)
        advice, at_point = engine.analyze(laps, total_dist)
        
        # A4レイアウト
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100, facecolor='white')
        plt.axis('off')
        
        fig.text(0.5, 0.95, f"{name}さんの分析レポート", fontsize=24, ha='center', weight='bold')
        
        # サマリ
        ax1 = fig.add_axes([0.1, 0.78, 0.8, 0.12])
        ax1.set_axis_off()
        ax1.add_patch(plt.Rectangle((0,0),1,1,color='#E6F3FF', transform=ax1.transAxes))
        m, s = divmod(sum(laps), 60)
        summary = f"距離: {total_dist}m   タイム: {int(m)}分{int(s):02d}秒"
        ax1.text(0.5, 0.5, summary, fontsize=18, ha='center', va='center')

        # グラフ
        ax2 = fig.add_axes([0.1, 0.45, 0.8, 0.30])
        ax2.plot(range(1, len(laps)+1), laps, marker='o', linewidth=3, color='#FF6B6B')
        ax2.set_title("ラップ推移")
        ax2.grid(True, linestyle='--', alpha=0.5)
        if at_point:
            ax2.axvline(x=at_point, color='blue', linestyle='--', label='AT値')
            ax2.legend()

        # アドバイス
        ax3 = fig.add_axes([0.1, 0.10, 0.8, 0.30])
        ax3.set_axis_off()
        ax3.add_patch(plt.Rectangle((0,0),1,1,fill=False, edgecolor='#333', linewidth=2, transform=ax3.transAxes))
        ax3.text(0.05, 0.5, advice, fontsize=14, va='center')

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

# ==========================================
# 4. アプリUI (メイン処理)
# ==========================================
def main():
    st.set_page_config(page_title="持久走分析", layout="centered")
    st.title("🏃‍♂️ 持久走分析アプリ")
    st.write("記録用紙を撮影してアップロードしてください。")
    
    uploaded_file = st.file_uploader("カメラ起動", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        with st.spinner("AIが複数のモデルで解析を試みています..."):
            try:
                # 画像処理
                image = Image.open(uploaded_file)
                image = ImageOps.exif_transpose(image)
                st.image(image, caption="送信画像", width=200)
                
                # JPEG変換 (エラー回避)
                img_byte_arr = io.BytesIO()
                image = image.convert('RGB')
                image.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                # 解析実行
                data, error = analyze_image_with_direct_api(img_bytes)
                
                if data:
                    img_buf = ReportGenerator.create_image(data)
                    if img_buf:
                        st.success("分析完了！")
                        st.image(img_buf, use_column_width=True)
                        st.markdown("画像を長押しで保存")
                    else:
                        st.error("レポート作成失敗")
                else:
                    st.error(error)
            except Exception as e:
                st.error(f"システムエラー: {e}")

if __name__ == "__main__":
    main()
