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
# 1. AI読み取りエンジン (2つの記録を同時に読む)
# ==========================================
def analyze_image(img_bytes):
    model_name = "gemini-1.5-flash"
    base64_data = base64.b64encode(img_bytes).decode('utf-8')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    # プロンプト：上段(15分/12分走)と下段(3000m/2100m)の両方を読む
    prompt = """
    持久走記録用紙を読み取り、以下のJSON形式で返答せよ。
    
    【構造】
    用紙は「上段：15分間走(男子)or12分間走(女子)」と「下段：3000m(男子)or2100m(女子)」に分かれている場合がある。
    
    【必須抽出項目】
    1. name: 名前
    2. long_run_dist: 上段の合計距離(m)。(例: 4050) ※記載がなければ0
    3. time_trial_laps: 下段の各周回のタイム(秒)のリスト。(例: [65, 68...])
       ※分秒表記は秒に変換。累積タイムは区間タイムに直す。
       ※下段が空欄なら空リスト[]にする。
    
    Example Output:
    {"name": "Yamada", "long_run_dist": 4050, "time_trial_laps": [65, 66, 67]}
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_data}}]}]}
    
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        result_json = res.json()
        if "error" in result_json: return None, result_json['error']['message']
        if 'candidates' not in result_json: return None, "AI応答なし"

        text = result_json['candidates'][0]['content']['parts'][0]['text']
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0)), None
        else:
            return None, "解析失敗"
    except Exception as e:
        return None, f"エラー: {e}"

# ==========================================
# 2. 科学的分析ロジック (リーゲルの公式実装)
# ==========================================
class ScienceEngine:
    def __init__(self, data):
        self.name = data.get("name", "選手")
        self.long_run_dist = data.get("long_run_dist", 0) # 15分走の距離
        self.tt_laps = np.array(data.get("time_trial_laps", [])) # 3000mの実走データ
        
        # 性別判定（距離から推測）
        # 15分走で3500m以上なら男子(3000m基準)、それ以下なら女子(2100m基準)と仮定
        self.is_male = True if self.long_run_dist > 3200 else False 
        self.target_dist = 3000 if self.is_male else 2100
        self.long_run_min = 15 if self.is_male else 12

    def get_potential_time(self):
        """15分走/12分走の距離から3000m/2100mの予測タイムを算出 (Riegel's formula)"""
        if self.long_run_dist == 0: return None # データなし
        
        # T2 = T1 * (D2 / D1)^1.06
        t1 = self.long_run_min * 60 # 秒
        d1 = self.long_run_dist
        d2 = self.target_dist
        
        predicted_seconds = t1 * (d2 / d1)**1.06
        return predicted_seconds

    def get_vo2_max(self):
        if self.long_run_dist == 0: return 0
        # 12分走相当に換算して計算
        dist_12min = self.long_run_dist * (12 / self.long_run_min)
        vo2 = (dist_12min - 504.9) / 44.73
        return max(vo2, 0)

# ==========================================
# 3. レポート描画 (先生の指定レイアウト)
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_dashboard(data):
        plt.close('all')
        fp = get_japanese_font_prop()
        if not fp: return None

        engine = ScienceEngine(data)
        
        # 予測タイム（ポテンシャル）の計算
        potential_sec = engine.get_potential_time()
        
        # 実走タイム（もしあれば）
        actual_sec = sum(engine.tt_laps) if len(engine.tt_laps) > 0 else 0
        
        # A4横
        fig = plt.figure(figsize=(11.69, 8.27), dpi=100, facecolor='white')
        
        # ヘッダー
        title = f"科学的分析レポート: {engine.name} 選手"
        sub = f"基準データ: {engine.long_run_min}分間走 {engine.long_run_dist}m"
        fig.text(0.05, 0.95, title, fontproperties=fp, fontsize=22, weight='bold', color='#1a237e')
        fig.text(0.05, 0.92, sub, fontproperties=fp, fontsize=12, color='gray')

        # ------------------------------------------------
        # ① 左上: ポテンシャル評価 (VO2Max & 予測)
        # ------------------------------------------------
        ax1 = fig.add_axes([0.05, 0.60, 0.40, 0.25])
        ax1.set_axis_off()
        ax1.set_title("① 基礎走力からのポテンシャル分析", fontproperties=fp, loc='left', color='#0d47a1', fontsize=14, weight='bold')
        
        vo2 = engine.get_vo2_max()
        p_min, p_sec = divmod(potential_sec, 60) if potential_sec else (0, 0)
        
        eval_text = (
            f"【{engine.long_run_min}分間走に基づく推計】\n"
            f"● VO2 Max(最大酸素摂取量): {vo2:.1f} ml/kg/min\n"
            f"● {engine.target_dist}m 推定限界タイム: {int(p_min)}分{int(p_sec):02d}秒\n\n"
            f"【評価】\n"
            f"あなたの心肺機能(VO2Max {vo2:.1f})があれば、\n"
            f"{engine.target_dist}mを『{int(p_min)}分{int(p_sec):02d}秒』で走る能力があります。\n"
        )
        if actual_sec > 0 and potential_sec:
            diff = actual_sec - potential_sec
            if diff > 15:
                eval_text += f"しかし実際は理論値より{int(diff)}秒遅れています。\nペース配分に改善の余地があります。"
            elif diff < -5:
                eval_text += f"理論値を上回る素晴らしい走りです！"
            else:
                eval_text += f"理論値通りの実力を発揮できています。"
                
        ax1.text(0.0, 0.85, eval_text, fontproperties=fp, fontsize=11, va='top', linespacing=1.6)

        # ------------------------------------------------
        # ② 右上: 実走ラップデータ (あれば表示)
        # ------------------------------------------------
        ax2 = fig.add_axes([0.50, 0.60, 0.45, 0.25])
        ax2.set_axis_off()
        ax2.set_title(f"② {engine.target_dist}m 実走データ", fontproperties=fp, loc='left', color='#0d47a1', fontsize=14, weight='bold')
        
        if len(engine.tt_laps) > 0:
            col_labels = ["周回", "ラップ", "変動"]
            table_data = []
            for i, lap in enumerate(engine.tt_laps):
                diff = lap - engine.tt_laps[i-1] if i > 0 else 0
                mark = "▼DOWN" if diff >= 3 else ("▲UP" if diff <= -2 else "―")
                table_data.append([f"{i+1}", f"{lap:.1f}", mark])
            
            if len(table_data) > 10: table_data = table_data[:10]
            the_table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', colColours=['#e3f2fd']*3)
            the_table.scale(1, 1.3)
            for cell in the_table.get_celld().values(): cell.set_text_props(fontproperties=fp)
        else:
            ax2.text(0.1, 0.5, "※3000m/2100mの実走データが\n読み取れませんでした。", fontproperties=fp, fontsize=12)

        # ------------------------------------------------
        # ③ 左下: 目標通過タイム表 (ポテンシャルから算出)
        # ------------------------------------------------
        ax3 = fig.add_axes([0.05, 0.10, 0.40, 0.40])
        ax3.set_axis_off()
        ax3.set_title("③ 能力別：目標ラップ表", fontproperties=fp, loc='left', color='#0d47a1', fontsize=14, weight='bold')
        
        if potential_sec:
            # 基準ペース（理論限界）
            base_pace = potential_sec / (engine.target_dist / 300) # 300mトラック換算の1周
            
            # 4段階設定
            # Level 1: 安全圏 (理論値の90%強度)
            # Level 2: 挑戦圏 (理論値の95%強度)
            # Level 3: 理論限界 (100%)
            # Level 4: 限界突破 (102%)
            
            headers = ["レベル", "目標タイム", "1周(300m)ペース"]
            rows = []
            levels = [
                ("安全圏", 1.10), 
                ("挑戦圏", 1.05), 
                ("理論値", 1.00), 
                ("限界突破", 0.98)
            ]
            
            for label, ratio in levels:
                target_sec = potential_sec * ratio
                lap_pace = base_pace * ratio
                rows.append([label, ReportGenerator.fmt_time(target_sec), f"{lap_pace:.1f}秒"])
            
            t_table = ax3.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center', colColours=['#fff9c4']*3)
            t_table.scale(1, 2)
            t_table.set_fontsize(11)
            for cell in t_table.get_celld().values(): cell.set_text_props(fontproperties=fp)
        else:
            ax3.text(0.1, 0.5, "15分走データがないため算出不能", fontproperties=fp)

        # ------------------------------------------------
        # ④ 右下: 実戦アドバイス (実走データとの比較)
        # ------------------------------------------------
        ax4 = fig.add_axes([0.50, 0.10, 0.45, 0.40])
        ax4.set_axis_off()
        ax4.set_title("④ コーチからの戦術アドバイス", fontproperties=fp, loc='left', color='#0d47a1', fontsize=14, weight='bold')
        
        advice = ""
        if len(engine.tt_laps) > 0 and potential_sec:
            # AT値判定
            at_point = next((i+1 for i in range(1, len(engine.tt_laps)) if engine.tt_laps[i] - engine.tt_laps[i-1] >= 3.0), None)
            
            # 理論ラップ
            theory_lap = potential_sec / len(engine.tt_laps)
            
            advice += f"【現状の課題】\n"
            if at_point:
                advice += f"{at_point}周目にAT値（スタミナ切れ）が来ています。\n"
            advice += f"あなたの心肺機能なら、1周{theory_lap:.1f}秒で押していけるはずです。\n"
            
            # 前半と後半の比較
            half = len(engine.tt_laps) // 2
            first_half = np.mean(engine.tt_laps[:half])
            second_half = np.mean(engine.tt_laps[half:])
            
            advice += "\n【次回の戦術】\n"
            if first_half < theory_lap - 2:
                advice += "今回は「入り」が速すぎました。\n最初の3周を意識的に抑えれば、後半の失速を防げます。\n"
            else:
                advice += "イーブンペースを意識して、中盤の粘りを強化しましょう。\n"
            
            advice += f"\n👉 左の表の『理論値』のラップを参考に\nペースメイクしてください。"
        elif potential_sec:
            advice += "実走データがありませんが、左の表があなたの目安です。\nまずは『安全圏』のペースで完走を目指しましょう。"
        else:
            advice += "データ不足のためアドバイスを作成できません。"

        # 枠線
        rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#333', linewidth=1, transform=ax4.transAxes)
        ax4.add_patch(rect)
        ax4.text(0.05, 0.9, advice, fontproperties=fp, fontsize=11, va='top', linespacing=1.6)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

    @staticmethod
    def fmt_time(seconds):
        m, s = divmod(seconds, 60)
        return f"{int(m)}:{int(s):02d}"

# ==========================================
# 4. アプリUI
# ==========================================
def main():
    st.set_page_config(page_title="Running Analysis", layout="wide")
    st.title("🏃‍♂️ 持久走データ・サイエンス分析")
    st.markdown("""
    **【使い方】**
    15分間走(または12分間走)の記録と、3000m(2100m)の記録が書かれた用紙をアップロードしてください。
    基礎体力(15分走)から、3000mの目標タイムを算出します。
    """)
    
    uploaded_file = st.file_uploader("記録用紙を撮影/アップロード", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        with st.spinner("AIが用紙全体(15分走＆3000m)を解析中..."):
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
