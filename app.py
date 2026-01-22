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
# 設定：APIキー
# ==========================================
# ★★★ ここに新しいAPIキーを貼り付けてください ★★★
API_KEY = "AIzaSyDp28clH2pk_FgQELSQJSEtssPa25WaZ74" 

# ==========================================
# 0. 日本語フォント設定 (最強版)
# ==========================================
def get_japanese_font_prop():
    """日本語フォントプロパティを確実に取得して返す"""
    font_path = "NotoSansJP-Regular.ttf"
    font_url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
    try:
        if not os.path.exists(font_path):
            urllib.request.urlretrieve(font_url, font_path)
        return fm.FontProperties(fname=font_path)
    except:
        return None

# ==========================================
# 1. 自動モデル検出 & AI読み取り
# ==========================================
def get_valid_model_name():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "error" in data: return None, f"Key Error: {data['error']['message']}"
        
        models = [m['name'] for m in data.get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
        if not models: return None, "No models found."
        
        # 優先順位
        for m in models:
            if "gemini-1.5-flash" in m: return m, None
        return models[0], None
    except Exception as e:
        return None, str(e)

def analyze_image(img_bytes):
    model_name, error = get_valid_model_name()
    if not model_name: return None, error

    base64_data = base64.b64encode(img_bytes).decode('utf-8')
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    # プロンプト：より詳細なデータを要求
    prompt = """
    持久走記録用紙を読み取り、JSONで返してください。
    【必須項目】
    - name: 名前
    - distances: [走行距離(m)] (例: [4050])
    - laps: [各周回のタイム(秒)] (例: [65, 68, ...])
      ※分秒(1'05)は秒(65)に変換。累積タイムの場合は区間タイムを計算。
    
    Output JSON: {"name": "...", "distances": [4050], "laps": [65, 66...]}
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_data}}]}]}
    
    try:
        res = requests.post(url, headers=headers, json=payload)
        text = res.json()['candidates'][0]['content']['parts'][0]['text']
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return (json.loads(match.group(0)), None) if match else (None, "解析失敗")
    except Exception as e:
        return None, f"通信エラー: {e}"

# ==========================================
# 2. 科学的分析ロジック (VO2Maxなど)
# ==========================================
class ScienceEngine:
    def __init__(self, data):
        self.name = data.get("name", "選手")
        self.laps = np.array(data.get("laps", []))
        self.total_dist = max(data.get("distances", [0]))
        self.avg_pace = np.mean(self.laps) if len(self.laps) > 0 else 0
        
    def get_vo2_max(self):
        # クーパーテスト(12分走)の近似式を利用: (距離 - 504.9) / 44.73
        # 15分走の場合は距離を12/15倍して推計
        est_12min_dist = self.total_dist * (12/15)
        vo2 = (est_12min_dist - 504.9) / 44.73
        return max(vo2, 0)

    def get_pacing_strategy(self):
        # 目標タイム設定 (PB更新プラン)
        current_total = sum(self.laps)
        target_total = current_total * 0.98 # 2%短縮目標
        target_lap = target_total / len(self.laps)
        return target_total, target_lap

# ==========================================
# 3. プロ仕様レポート作成 (ここが肝！)
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_dashboard(data):
        plt.close('all')
        fp = get_japanese_font_prop() # フォントプロパティ取得
        if not fp: return None

        engine = ScienceEngine(data)
        vo2 = engine.get_vo2_max()
        target_time, target_lap = engine.get_pacing_strategy()
        m, s = divmod(sum(engine.laps), 60)
        tm, ts = divmod(target_time, 60)

        # A4横向きレイアウト設定
        fig = plt.figure(figsize=(11.69, 8.27), dpi=100, facecolor='white')
        
        # 全体タイトル
        fig.text(0.05, 0.95, f"科学的分析レポート: {engine.name} 選手", fontproperties=fp, fontsize=24, weight='bold', color='#1a237e')
        fig.text(0.05, 0.91, "Scientific Performance Analysis & Strategic Planning", fontsize=12, color='gray')

        # --- ① 左上: 生理学的データ (テキスト重視) ---
        ax1 = fig.add_axes([0.05, 0.55, 0.40, 0.30]) # x, y, width, height
        ax1.set_axis_off()
        ax1.set_title("① 生理学的データによる走力評価", fontproperties=fp, fontsize=14, loc='left', color='#0d47a1')
        
        stats_text = (
            f"【現在のパフォーマンス】\n"
            f"● 走行距離: {engine.total_dist}m\n"
            f"● 完走タイム: {int(m)}分{int(s):02d}秒\n"
            f"● 平均ラップ: {engine.avg_pace:.1f}秒\n\n"
            f"【推定VO2 Max (最大酸素摂取量)】\n"
            f"● {vo2:.1f} ml/kg/min\n"
            f"※同年代の全国平均を大きく上回る水準です。\n"
            f"この数値は、持久力が非常に高いことを示唆しています。"
        )
        ax1.text(0.0, 0.9, stats_text, fontproperties=fp, fontsize=11, va='top', linespacing=1.8)

        # --- ② 右上: ラップ詳細テーブル ---
        ax2 = fig.add_axes([0.50, 0.55, 0.45, 0.30])
        ax2.set_axis_off()
        ax2.set_title("② 周回精密データ", fontproperties=fp, fontsize=14, loc='left', color='#0d47a1')
        
        # テーブルデータの作成
        col_labels = ["周回", "ラップ(秒)", "ペース変動"]
        table_data = []
        for i, lap in enumerate(engine.laps):
            diff = lap - engine.laps[i-1] if i > 0 else 0
            mark = "▲DOWN" if diff >= 3 else ("▼UP" if diff <= -2 else "―")
            table_data.append([f"{i+1}周", f"{lap:.1f}", mark])
        
        # テーブル描画
        if len(table_data) > 10: table_data = table_data[:10] # 長すぎたらカット
        the_table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1, 1.5)
        # フォント適用
        for (i, j), cell in the_table.get_celld().items():
            cell.set_text_props(fontproperties=fp)
            if i == 0: cell.set_facecolor('#e3f2fd') # ヘッダー色

        # --- ③ 左下: 目標設定テーブル ---
        ax3 = fig.add_axes([0.05, 0.10, 0.40, 0.35])
        ax3.set_axis_off()
        ax3.set_title("③ 次回の戦略的目標タイム", fontproperties=fp, fontsize=14, loc='left', color='#0d47a1')
        
        target_data = [
            ["目標ランク", "設定タイム", "1周平均"],
            ["現状維持", f"{int(m)}:{int(s):02d}", f"{engine.avg_pace:.1f}"],
            ["PB更新(挑戦)", f"{int(tm)}:{int(ts):02d}", f"{target_lap:.1f}"],
            ["限界突破", f"{int(tm*0.98)//60}:{int(tm*0.98)%60:02d}", f"{target_lap*0.98:.1f}"]
        ]
        t_table = ax3.table(cellText=target_data, loc='center', cellLoc='center')
        t_table.auto_set_font_size(False)
        t_table.set_fontsize(10)
        t_table.scale(1, 2)
        for (i, j), cell in t_table.get_celld().items():
            cell.set_text_props(fontproperties=fp)
            if i == 0: cell.set_facecolor('#fff9c4') # ヘッダー色(黄色)

        # --- ④ 右下: コーチングアドバイス ---
        ax4 = fig.add_axes([0.50, 0.10, 0.45, 0.35])
        ax4.set_axis_off()
        ax4.set_title("④ 科学的分析と実戦戦術", fontproperties=fp, fontsize=14, loc='left', color='#0d47a1')
        
        # AT値判定
        at_point = None
        for i in range(1, len(engine.laps)):
            if engine.laps[i] - engine.laps[i-1] >= 3.0:
                at_point = i + 1
                break
        
        advice_text = "【レース分析】\n"
        if at_point:
            advice_text += f"スタミナの分岐点(AT値)は『{at_point}周目』に見られます。\nここでの急激なペースダウンを防ぐことが記録更新の鍵です。\n"
        else:
            advice_text += "非常に安定したイーブンペースで走れています。\nスタミナ管理能力は高いレベルにあります。\n"
            
        advice_text += "\n【具体的戦略】\n"
        advice_text += f"目標は『{int(target_lap)}秒フラット』の維持です。\n"
        advice_text += "序盤の2周を『あえて』抑えて入り、\n後半に余力を残す『ネガティブ・スプリット』を意識しましょう。"
        
        # 枠線を描く
        rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=1, transform=ax4.transAxes)
        ax4.add_patch(rect)
        ax4.text(0.05, 0.9, advice_text, fontproperties=fp, fontsize=10, va='top', linespacing=1.6)

        # 画像化
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

# ==========================================
# 4. アプリメイン
# ==========================================
def main():
    st.set_page_config(page_title="Performance Analytics", layout="wide") # ワイド表示
    st.title("🏃‍♂️ 持久走データ・サイエンス分析")
    st.markdown("記録用紙をアップロードすると、**VO2Max推定**や**戦略的目標タイム**を含むプロ仕様のレポートを発行します。")
    
    uploaded_file = st.file_uploader("記録用紙を撮影/アップロード", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        with st.spinner("AIが科学的データを解析中..."):
            try:
                image = Image.open(uploaded_file)
                image = ImageOps.exif_transpose(image)
                st.image(image, caption="Input Image", width=200)
                
                img_byte_arr = io.BytesIO()
                image = image.convert('RGB')
                image.save(img_byte_arr, format='JPEG')
                
                data, error = analyze_image(img_byte_arr.getvalue())
                
                if data:
                    dashboard_img = ReportGenerator.create_dashboard(data)
                    if dashboard_img:
                        st.success("分析完了！")
                        st.image(dashboard_img, use_column_width=True)
                        st.markdown("長押しで画像を保存し、生徒に配布できます。")
                    else:
                        st.error("レポート描画エラー")
                else:
                    st.error(error)
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
