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
API_KEY = "AIzaSyC-OhXt-YtEKB7bVv7l3jlIvYOFKZ_ZF6I" 

# ==========================================
# 0. 日本語フォント設定
# ==========================================
def get_japanese_font_prop():
    font_path = "NotoSansJP-Regular.ttf"
    font_url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
    try:
        if not os.path.exists(font_path):
            urllib.request.urlretrieve(font_url, font_path)
        return fm.FontProperties(fname=font_path)
    except:
        return None

# ==========================================
# 1. AI読み取りエンジン (安定版固定・制限回避)
# ==========================================
def analyze_image(img_bytes):
    base64_data = base64.b64encode(img_bytes).decode('utf-8')
    
    # ★修正点: 実験版(exp)やPro版は制限にかかりやすいので除外。
    # 「gemini-1.5-flash」のみに絞ることでQuotaエラーを回避します。
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-001"
    ]
    
    prompt = """
    持久走記録用紙を読み取り、以下のJSON形式で返答せよ。
    
    【構造】
    用紙は「上段：15分間走(男子)or12分間走(女子)」と「下段：3000m(男子)or2100m(女子)」に分かれている場合がある。
    
    【必須抽出項目】
    1. name: 名前 (読めなければ"選手")
    2. long_run_dist: 上段の合計距離(m)。(例: 4050) ※記載がなければ0
    3. time_trial_laps: 下段の各周回のタイム(秒)のリスト。(例: [65, 68...])
       ※分秒表記は秒に変換。累積タイムは区間タイムに直す。
       ※下段が空欄なら空リスト[]にする。
    
    Example Output:
    {"name": "Yamada", "long_run_dist": 4050, "time_trial_laps": [65, 66, 67]}
    """
    
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_data}}]}]}
    
    last_error = ""
    
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=30)
            result_json = res.json()
            
            # エラーチェック
            if "error" in result_json:
                error_msg = result_json['error']['message'].lower()
                # 429エラー(Resource exhausted/Quota)なら少し待って次へ
                if "quota" in error_msg or "exhausted" in error_msg:
                    time.sleep(2) # 2秒待機
                    last_error = f"{model_name} (制限超過): {error_msg}"
                    continue
                # モデルが見つからない場合も次へ
                if "not found" in error_msg:
                    continue
                    
                return None, f"APIエラー: {result_json['error']['message']}"
            
            if 'candidates' in result_json:
                text = result_json['candidates'][0]['content']['parts'][0]['text']
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    return json.loads(match.group(0)), None
            
        except Exception as e:
            last_error = str(e)
            continue

    return None, f"解析に失敗しました。しばらく時間を空けて試してください。(詳細: {last_error})"

# ==========================================
# 2. 科学的分析ロジック
# ==========================================
class ScienceEngine:
    def __init__(self, data):
        self.name = data.get("name", "選手")
        self.long_run_dist = data.get("long_run_dist", 0) 
        self.tt_laps = np.array(data.get("time_trial_laps", []))
        
        # 3000mか2100mか判定
        self.is_male = True if self.long_run_dist > 3200 else False 
        self.target_dist = 3000 if self.is_male else 2100
        self.long_run_min = 15 if self.is_male else 12

    def get_potential_time(self):
        """リーゲルの公式で予測タイム算出"""
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
# 3. レポート描画 (4分割レイアウト)
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_dashboard(data):
        plt.close('all')
        fp = get_japanese_font_prop()
        if not fp: return None

        engine = ScienceEngine(data)
        potential_sec = engine.get_potential_time()
        actual_sec = sum(engine.tt_laps) if len(engine.tt_laps) > 0 else 0
        
        # A4横向き
        fig = plt.figure(figsize=(11.69, 8.27), dpi=100, facecolor='white')
        
        # ヘッダー
        fig.text(0.05, 0.95, f"科学的分析レポート: {engine.name} 選手", fontproperties=fp, fontsize=22, weight='bold', color='#1a237e')
        fig.text(0.05, 0.92, f"基準: {engine.long_run_min}分間走 {engine.long_run_dist}m", fontproperties=fp, fontsize=12, color='gray')

        # ① 左上: ポテンシャル評価
        ax1 = fig.add_axes([0.05, 0.60, 0.40, 0.25])
        ax1.set_axis_off()
        ax1.set_title("① 基礎走力からのポテンシャル推計", fontproperties=fp, loc='left', color='#0d47a1', fontsize=14, weight='bold')
        
        vo2 = engine.get_vo2_max()
        p_m, p_s = divmod(potential_sec, 60) if potential_sec else (0,0)
        
        eval_txt = (
            f"● VO2 Max(最大酸素摂取量): {vo2:.1f} ml/kg/min\n"
            f"● {engine.target_dist}m 推定限界タイム: {int(p_m)}分{int(p_s):02d}秒\n\n"
            f"【評価】\n"
            f"あなたの心肺機能があれば、{engine.target_dist}mを\n"
            f"『{int(p_m)}分{int(p_s):02d}秒』で走る力があります。"
        )
        ax1.text(0, 0.8, eval_txt, fontproperties=fp, fontsize=11, va='top', linespacing=1.6)

        # ② 右上: 実走ラップ
        ax2 = fig.add_axes([0.50, 0.60, 0.45, 0.25])
        ax2.set_axis_off()
        ax2.set_title(f"② {engine.target_dist}m 実走データ", fontproperties=fp, loc='left', color='#0d47a1', fontsize=14, weight='bold')
        
        if len(engine.tt_laps) > 0:
            col_labels = ["周回", "ラップ", "変動"]
            table_data = []
            for i, lap in enumerate(engine.tt_laps):
                diff = lap - engine.tt_laps[i-1] if i > 0 else 0
                mark = "▼DOWN" if diff >= 3 else ("▲UP" if diff <= -2 else "-")
                table_data.append([f"{i+1}", f"{lap:.1f}", mark])
            if len(table_data) > 10: table_data = table_data[:10]
            
            the_table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', colColours=['#e3f2fd']*3)
            the_table.scale(1, 1.3)
            for cell in the_table.get_celld().values(): cell.set_text_props(fontproperties=fp)
        else:
            ax2.text(0.1, 0.5, "実走データなし", fontproperties=fp)

        # ③ 左下: 目標タイム表
        ax3 = fig.add_axes([0.05, 0.10, 0.40, 0.40])
        ax3.set_axis_off()
        ax3.set_title("③ 能力別：目標ラップ表", fontproperties=fp, loc='left', color='#0d47a1', fontsize=14, weight='bold')
        
        if potential_sec:
            base_pace = potential_sec / (engine.target_dist / 300) 
            headers = ["レベル", "目標(3000/2100)", "1周(300m)"]
            rows = []
            levels = [("安全圏", 1.10), ("挑戦圏", 1.05), ("理論値", 1.00), ("限界突破", 0.98)]
            
            for label, ratio in levels:
                t_sec = potential_sec * ratio
                l_pace = base_pace * ratio
                m, s = divmod(t_sec, 60)
                rows.append([label, f"{int(m)}:{int(s):02d}", f"{l_pace:.1f}秒"])
                
            t_table = ax3.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center', colColours=['#fff9c4']*3)
            t_table.scale(1, 2)
            t_table.set_fontsize(11)
            for cell in t_table.get_celld().values(): cell.set_text_props(fontproperties=fp)
        else:
            ax3.text(0.1, 0.5, "15分走データ不足", fontproperties=fp)

        # ④ 右下: アドバイス
        ax4 = fig.add_axes([0.50, 0.10, 0.45, 0.40])
        ax4.set_axis_off()
        ax4.set_title("④ 戦術アドバイス", fontproperties=fp, loc='left', color='#0d47a1', fontsize=14, weight='bold')
        
        advice = ""
        if len(engine.tt_laps) > 0 and potential_sec:
            theory_lap = potential_sec / len(engine.tt_laps)
            advice += f"あなたの心肺機能なら、1周{theory_lap:.1f}秒で押していけます。\n\n"
            
            # AT値
            at_point = next((i+1 for i in range(1, len(engine.tt_laps)) if engine.tt_laps[i] - engine.tt_laps[i-1] >= 3.0), None)
            if at_point: advice += f"⚠️ {at_point}周目にペースダウンしています。\nここがスタミナの壁です。\n"
            
            advice += "\n👉 左の表の『理論値』または『挑戦圏』の\nラップを守って走ってみましょう。"
        elif potential_sec:
            advice += "実走データがありませんが、左の表があなたの基準です。\nまずは『安全圏』のペースで完走を目指してください。"
        else:
            advice += "データ不足のため分析できません。"

        rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#333', linewidth=1, transform=ax4.transAxes)
        ax4.add_patch(rect)
        ax4.text(0.05, 0.9, advice, fontproperties=fp, fontsize=11, va='top', linespacing=1.6)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

# ==========================================
# 4. アプリUI
# ==========================================
def main():
    st.set_page_config(page_title="Running Analysis", layout="wide")
    st.title("🏃‍♂️ 持久走データ・サイエンス分析")
    st.markdown("15分間走(または12分間走)の記録と、3000m(2100m)の記録が書かれた用紙をアップロードしてください。")
    
    uploaded_file = st.file_uploader("記録用紙をアップロード", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        with st.spinner("AI解析中..."):
            try:
                image = Image.open(uploaded_file)
                image = ImageOps.exif_transpose(image)
                st.image(image, caption="入力画像", width=200)
                
                img_byte_arr = io.BytesIO()
                image = image.convert('RGB')
                image.save(img_byte_arr, format='JPEG')
                
                data, error = analyze_image(img_byte_arr.getvalue())
                
                if data:
                    dashboard_img = ReportGenerator.create_dashboard(data)
                    if dashboard_img:
                        st.success("分析完了！")
                        st.image(dashboard_img, use_column_width=True)
                    else:
                        st.error("レポート描画失敗")
                else:
                    st.error(f"解析エラー: {error}")
            except Exception as e:
                st.error(f"エラー: {e}")

if __name__ == "__main__":
    main()
