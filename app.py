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
# 設定：APIキー (2026/01/22 更新版)
# ==========================================
API_KEY = "AIzaSyDp28clH2pk_FgQELSQJSEtssPa25WaZ74"

# ==========================================
# 0. 日本語フォント設定 (失敗しても止まらない版)
# ==========================================
def get_japanese_font_prop():
    """
    日本語フォントを取得する。失敗したらNoneを返すが、
    呼び出し元でエラーにせずデフォルトフォントを使うようにする。
    """
    font_path = "NotoSansJP-Regular.ttf"
    font_url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
    try:
        if not os.path.exists(font_path):
            urllib.request.urlretrieve(font_url, font_path)
        return fm.FontProperties(fname=font_path)
    except Exception as e:
        st.warning(f"フォント読み込み注意: {e}") # 画面に警告だけ出す
        return None

# ==========================================
# 1. AI読み取り (自動モデル検出 & エラー詳細化)
# ==========================================
def get_valid_model_name():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "error" in data: return None, f"Key Error: {data['error']['message']}"
        
        models = [m['name'] for m in data.get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
        if not models: return None, "No models found."
        
        # 優先順位: Flash -> Pro
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
    
    # プロンプト：JSON形式を厳格に指定
    prompt = """
    持久走記録用紙を読み取り、以下のJSON形式だけで返答してください。余計な文字は一切不要です。
    
    【必須項目】
    - name: 名前 (読めなければ"選手")
    - gender: "男子" または "女子"
    - distances: [走行距離(m)] (例: [4050])
    - laps: [各周回のタイム(秒)] (例: [65, 68, ...])
      ※分秒(1'05)は秒(65)に変換。累積タイムの場合は引き算して区間タイムを算出すること。
    
    Example Output:
    {"name": "Yamada", "gender": "男子", "distances": [4050], "laps": [65, 66, 67]}
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_data}}]}]}
    
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        result_json = res.json()
        
        if "error" in result_json:
            return None, result_json['error']['message']

        if 'candidates' not in result_json:
             return None, "AIからの応答が空でした。"

        text = result_json['candidates'][0]['content']['parts'][0]['text']
        # JSON部分抽出 (Markdownの ```json 等を除去)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0)), None
        else:
            return None, "データ形式の読み取りに失敗しました"
    except Exception as e:
        return None, f"通信/解析エラー: {e}"

# ==========================================
# 2. 科学的分析ロジック
# ==========================================
class ScienceEngine:
    def __init__(self, data):
        self.name = data.get("name", "選手")
        self.laps = np.array(data.get("laps", []))
        # 距離がリストか数値か両方対応
        d = data.get("distances", [3000])
        self.total_dist = d[0] if isinstance(d, list) and d else (d if isinstance(d, (int, float)) else 3000)
        self.avg_pace = np.mean(self.laps) if len(self.laps) > 0 else 0
        
    def get_vo2_max(self):
        # 15分走(または12分走相当)からの推定
        # 簡易式: (走行距離 - 504.9) / 44.73
        dist_12min = self.total_dist
        # 3000m走などの場合、距離が短いのでペースから12分走距離を推計
        if self.total_dist < 3500 and self.avg_pace > 0:
            dist_12min = (720 / self.avg_pace) * (self.total_dist / len(self.laps))
        
        vo2 = (dist_12min - 504.9) / 44.73
        return max(vo2, 0)

# ==========================================
# 3. プロ仕様レポート作成 (エラー回避版)
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_dashboard(data):
        plt.close('all')
        
        # フォント取得（失敗したらNoneだが進める）
        fp = get_japanese_font_prop() 
        
        try:
            engine = ScienceEngine(data)
            if len(engine.laps) == 0:
                st.error("ラップデータが0件です。画像を読み取れませんでした。")
                return None

            vo2 = engine.get_vo2_max()
            m, s = divmod(sum(engine.laps), 60)
            
            # A4横向きレイアウト
            fig = plt.figure(figsize=(11.69, 8.27), dpi=100, facecolor='white')
            
            # フォント設定ヘルパー（fpがNoneならデフォルトを使う）
            def set_text(obj, **kwargs):
                if fp: obj.set_fontproperties(fp)
                # フォント以外のプロパティを設定
                for k, v in kwargs.items():
                    if k != 'fontproperties': getattr(obj, f"set_{k}")(v)

            # タイトル
            t = fig.text(0.05, 0.95, f"科学的分析レポート: {engine.name}", fontsize=24, weight='bold', color='#1a237e')
            if fp: t.set_fontproperties(fp)

            # --- ① 左上: 生理学的データ ---
            ax1 = fig.add_axes([0.05, 0.55, 0.40, 0.35])
            ax1.set_axis_off()
            t1 = ax1.set_title("① 生理学的データ評価", loc='left', color='#0d47a1', fontsize=16)
            if fp: t1.set_fontproperties(fp)
            
            stats_text = (
                f"走行距離: {engine.total_dist}m\n"
                f"タイム: {int(m)}分{int(s):02d}秒\n"
                f"平均ラップ: {engine.avg_pace:.1f}秒\n\n"
                f"【推定VO2 Max】\n"
                f"{vo2:.1f} ml/kg/min\n"
                f"※心肺機能の目安となる数値です。"
            )
            txt1 = ax1.text(0.05, 0.8, stats_text, fontsize=12, va='top', linespacing=1.8)
            if fp: txt1.set_fontproperties(fp)

            # --- ② 右上: ラップ詳細テーブル ---
            ax2 = fig.add_axes([0.50, 0.55, 0.45, 0.35])
            ax2.set_axis_off()
            t2 = ax2.set_title("② 周回データ", loc='left', color='#0d47a1', fontsize=16)
            if fp: t2.set_fontproperties(fp)
            
            col_labels = ["周回", "ラップ", "変動"]
            table_data = []
            for i, lap in enumerate(engine.laps):
                diff = lap - engine.laps[i-1] if i > 0 else 0
                mark = "▼" if diff >= 3 else ("▲" if diff <= -2 else "-")
                table_data.append([f"{i+1}", f"{lap:.1f}", mark])
            
            if len(table_data) > 12: table_data = table_data[:12] # はみ出し防止
            
            the_table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            the_table.scale(1, 1.4)
            
            if fp:
                for cell in the_table.get_celld().values():
                    cell.set_text_props(fontproperties=fp)

            # --- ③ 左下: 目標タイム ---
            ax3 = fig.add_axes([0.05, 0.10, 0.40, 0.35])
            ax3.set_axis_off()
            t3 = ax3.set_title("③ 目標ペース配分", loc='left', color='#0d47a1', fontsize=16)
            if fp: t3.set_fontproperties(fp)
            
            target_lap = engine.avg_pace * 0.98 # 2%短縮
            target_data = [
                ["目標", "1周設定"],
                ["現状維持", f"{engine.avg_pace:.1f}"],
                ["自己ベスト更新", f"{target_lap:.1f}"],
                ["限界突破", f"{target_lap*0.98:.1f}"]
            ]
            t_table = ax3.table(cellText=target_data, loc='center', cellLoc='center')
            t_table.scale(1, 2)
            t_table.auto_set_font_size(False)
            t_table.set_fontsize(11)
            if fp:
                for cell in t_table.get_celld().values():
                    cell.set_text_props(fontproperties=fp)

            # --- ④ 右下: アドバイス ---
            ax4 = fig.add_axes([0.50, 0.10, 0.45, 0.35])
            ax4.set_axis_off()
            t4 = ax4.set_title("④ 戦略アドバイス", loc='left', color='#0d47a1', fontsize=16)
            if fp: t4.set_fontproperties(fp)
            
            # AT値判定
            at_point = next((i+1 for i in range(1, len(engine.laps)) if engine.laps[i] - engine.laps[i-1] >= 3.0), None)
            
            advice = "【レース分析】\n"
            if at_point:
                advice += f"スタミナの分岐点(AT値)は\n『{at_point}周目』に見られます。\n"
            else:
                advice += "非常に安定したペース配分です。\n"
            advice += f"\n次回は1周『{target_lap:.1f}秒』を\n目指してみましょう。"
            
            rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=1, transform=ax4.transAxes)
            ax4.add_patch(rect)
            txt4 = ax4.text(0.05, 0.9, advice, va='top', linespacing=1.8, fontsize=11)
            if fp: txt4.set_fontproperties(fp)

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return buf

        except Exception as e:
            st.error(f"グラフ描画中にエラーが発生しました: {e}")
            return None

# ==========================================
# 4. アプリメイン
# ==========================================
def main():
    st.set_page_config(page_title="Performance Analytics", layout="wide")
    st.title("🏃‍♂️ 持久走データ・サイエンス分析")
    
    uploaded_file = st.file_uploader("記録用紙を撮影/アップロード", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        with st.spinner("AIが解析中..."):
            try:
                image = Image.open(uploaded_file)
                image = ImageOps.exif_transpose(image)
                st.image(image, caption="Uploaded Image", width=200)
                
                img_byte_arr = io.BytesIO()
                image = image.convert('RGB')
                image.save(img_byte_arr, format='JPEG')
                
                data, error = analyze_image(img_byte_arr.getvalue())
                
                if data:
                    dashboard_img = ReportGenerator.create_dashboard(data)
                    if dashboard_img:
                        st.success("分析完了！")
                        st.image(dashboard_img, use_column_width=True)
                        st.markdown("画像を長押しして保存してください。")
                    else:
                        st.error("レポートの描画に失敗しました。")
                else:
                    st.error(f"解析エラー: {error}")
            except Exception as e:
                st.error(f"システムエラー: {e}")

if __name__ == "__main__":
    main()
