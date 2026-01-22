import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import io
import google.generativeai as genai
from PIL import Image, ImageOps
import json
import re

# ==========================================
# 設定：APIキー
# ==========================================
API_KEY = "AIzaSyATM7vIfyhj6vKsZga3fydYLHvAMRVNdzg"

# ==========================================
# 1. AI読み取りエンジン (Gemini Pro Vision版)
# ==========================================
def analyze_image_with_gemini(img_obj):
    genai.configure(api_key=API_KEY)
    
    # ★変更点: 古い環境でも動く "gemini-pro-vision" を使用
    # ※設定もシンプルにしてエラーを回避
    model = genai.GenerativeModel('gemini-pro-vision')
    
    prompt = """
    あなたは持久走の記録係です。画像の記録用紙から数値を読み取ってください。
    
    【ルール】
    - 男子は3000m、女子は2100mが基準。
    - 名前、性別、距離、全てのラップタイム(秒)を抽出すること。
    - 分秒表記(例: 1'20)は秒(80)に変換すること。
    
    回答は以下のJSON形式の文字列だけで答えてください。余計な挨拶は不要です。
    {"name": "名前", "gender": "男子", "distances": [3000], "laps": [70, 72, 75]}
    """
    
    try:
        # 古いバージョン向けに設定を削除してシンプルに呼び出す
        response = model.generate_content([prompt, img_obj])
        text = response.text
        
        # JSON部分を探し出す
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0)), None
        else:
            return None, "データを読み取れませんでした。画像が鮮明か確認してください。"
    except Exception as e:
        return None, f"エラー: {e}"

# ==========================================
# 2. 分析ロジック
# ==========================================
class ScienceEngine:
    def __init__(self, gender="男子"):
        self.gender = gender
        if self.gender == "女子":
            self.target_dist = 2100
        else:
            self.target_dist = 3000

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
                laps_needed = remaining / lap_dist
                pred_time += laps_needed * avg_pace * 1.05

        advice = f"【{self.target_dist}m 分析結果】\n"
        m, s = divmod(pred_time, 60)
        advice += f"🏁 予測タイム: {int(m)}分{int(s):02d}秒\n"
        
        if at_point:
            advice += f"⚠️ {at_point}周目にペースダウンしています。\nここが『スタミナの壁(AT値)』です。\n"
        else:
            advice += "✅ 最後まで安定した素晴らしい走りです！\n"
            
        target = avg_pace * 0.98
        advice += f"\n💡 次回の目標ラップ: {target:.0f}秒\n"

        return advice, at_point

# ==========================================
# 3. レポート描画
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_image(data):
        plt.close('all')
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
        
        fig.text(0.5, 0.95, f"{name}さんの分析レポート", fontsize=24, ha='center', weight='bold', color='#1A2A3A')

        ax1 = fig.add_axes([0.1, 0.75, 0.8, 0.15])
        ax1.set_axis_off()
        ax1.add_patch(plt.Rectangle((0,0),1,1,color='#E6F3FF',transform=ax1.transAxes, zorder=0))
        m, s = divmod(sum(laps), 60)
        summary = f"距離: {total_dist}m\nタイム: {int(m)}分{int(s):02d}秒\n平均ラップ: {np.mean(laps):.1f}秒"
        ax1.text(0.5, 0.5, summary, fontsize=18, ha='center', va='center', linespacing=1.8)

        ax2 = fig.add_axes([0.1, 0.45, 0.8, 0.25])
        ax2.plot(range(1, len(laps)+1), laps, marker='o', linewidth=3, color='#FF6B6B')
        ax2.set_title("ラップ推移", fontsize=16)
        ax2.grid(True, linestyle='--', alpha=0.5)
        if at_point:
            ax2.axvline(x=at_point, color='blue', linestyle='--', label='AT値')
            ax2.legend(fontsize=12)

        ax3 = fig.add_axes([0.1, 0.10, 0.8, 0.30])
        ax3.set_axis_off()
        ax3.add_patch(plt.Rectangle((0,0),1,1,fill=False,edgecolor='#333',linewidth=2,transform=ax3.transAxes))
        ax3.text(0.05, 0.9, "コーチからのアドバイス", fontsize=16, weight='bold')
        ax3.text(0.05, 0.5, advice, fontsize=14, linespacing=1.8, va='center')

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
    st.write("写真をアップロードしてください。")
    
    uploaded_file = st.file_uploader("カメラで撮影", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        with st.spinner("AIが分析中..."):
            try:
                image = Image.open(uploaded_file)
                image = ImageOps.exif_transpose(image)
                st.image(image, caption="画像を確認中...", width=200)
                
                data, error = analyze_image_with_gemini(image)
                
                if data:
                    japanize_matplotlib.japanize()
                    img_buf = ReportGenerator.create_image(data)
                    if img_buf:
                        st.success("完了！")
                        st.image(img_buf, use_column_width=True)
                        st.markdown("画像を長押しで保存できます")
                    else:
                        st.error("データ読み取り失敗")
                else:
                    st.error(error)
            except Exception as e:
                st.error(f"エラー: {e}")

if __name__ == "__main__":
    main()
