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
API_KEY = "AIzaSyB1chpD8a-KlJj81rhuWwRoCmZ2DiR2zeU"

# ==========================================
# 0. 日本語フォント設定 (キャッシュ機能付き)
# ==========================================
@st.cache_resource
def get_japanese_font_prop():
    """日本語フォントをダウンロードし、プロパティを返す"""
    font_filename = "NotoSansJP-Regular.ttf"
    font_url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
    
    try:
        if not os.path.exists(font_filename):
            urllib.request.urlretrieve(font_url, font_filename)
        
        # フォントマネージャーに追加
        fm.fontManager.addfont(font_filename)
        
        # Matplotlibのデフォルトフォントに設定（全体適用）
        plt.rcParams['font.family'] = 'Noto Sans JP'
        
        return fm.FontProperties(fname=font_filename)
    except Exception as e:
        st.warning(f"フォント設定警告: {e}")
        return None

# ==========================================
# 1. AI読み取りエンジン (デバッグ強化版)
# ==========================================
def analyze_image(img_bytes):
    base64_data = base64.b64encode(img_bytes).decode('utf-8')
    
    # 制限にかかりにくいFlashモデルを優先
    models_to_try = [
        "gemini-1.5-flash", 
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro"
    ]
    
    prompt = """
    Please extract data from the running record sheet image.
    If the image is blurry or unreadable, do your best to guess.
    
    Return JSON ONLY. No markdown. No explanations.
    
    JSON Structure:
    {
      "name": "Student Name (or '選手')",
      "long_run_dist": 4050,  // Integer (meters) from top section (15min/12min run). 0 if not found.
      "time_trial_laps": [65, 68, 70] // Array of numbers (seconds) from bottom section (3000m/2100m).
    }
    """
    
    headers = {'Content-Type': 'application/json'}
    
    # 安全フィルター解除
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_data}}]}],
        "safetySettings": safety_settings
    }
    
    last_error_detail = "未実行"
    
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=30)
            result_json = res.json()
            
            # エラーチェック
            if "error" in result_json:
                error_msg = result_json['error']['message']
                if "quota" in error_msg.lower():
                    last_error_detail = f"{model_name}: 制限超過 (Quota exceeded)"
                    time.sleep(1)
                    continue
                last_error_detail = f"{model_name}: APIエラー ({error_msg})"
                continue
            
            # 応答チェック
            if 'candidates' in result_json and len(result_json['candidates']) > 0:
                candidate = result_json['candidates'][0]
                
                # ブロック理由チェック
                if candidate.get('finishReason') not in ['STOP', 'MAX_TOKENS', None]:
                    last_error_detail = f"{model_name}: ブロックされました (理由: {candidate.get('finishReason')})"
                    continue

                if 'content' in candidate:
                    text = candidate['content']['parts'][0]['text']
                    
                    # JSON抽出
                    match = re.search(r'\{.*\}', text, re.DOTALL)
                    if match:
                        try:
                            return json.loads(match.group(0)), None
                        except json.JSONDecodeError:
                            last_error_detail = f"{model_name}: JSON変換失敗 (内容: {text[:50]}...)"
                    else:
                        last_error_detail = f"{model_name}: JSONが見つかりません (応答: {text[:100]}...)"
            else:
                last_error_detail = f"{model_name}: 応答が空でした"
            
        except Exception as e:
            last_error_detail = f"{model_name}: 通信エラー ({str(e)})"
            continue

    return None, f"全てのモデルで失敗しました。\n詳細: {last_error_detail}"

# ==========================================
# 2. 科学的分析ロジック
# ==========================================
class ScienceEngine:
    def __init__(self, data):
        self.name = data.get("name", "選手")
        # 数値変換の安全策
        try:
            val = data.get("long_run_dist", 0)
            self.long_run_dist = float(val) if val is not None else 0
        except: self.long_run_dist = 0
        
        laps = data.get("time_trial_laps", [])
        if not isinstance(laps, list): laps = []
        clean_laps = []
        for x in laps:
            try: clean_laps.append(float(x))
            except: pass
        self.tt_laps = np.array(clean_laps)
        
        self.is_male = True if self.long_run_dist > 3200 else False 
        self.target_dist = 3000 if self.is_male else 2100
        self.long_run_min = 15 if self.is_male else 12

    def get_potential_time(self):
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
# 3. レポート描画 (文字化け修正版)
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_dashboard(data):
        plt.close('all')
        fp = get_japanese_font_prop() # フォント読み込み
        
        # フォントが見つからない場合のバックアップ（英語）
        if not fp:
            st.warning("日本語フォントのロードに失敗しました。英語で表示します。")
            title_font = None
        else:
            title_font = fp

        engine = ScienceEngine(data)
        potential_sec = engine.get_potential_time()
        
        # A4横向き
        fig = plt.figure(figsize=(11.69, 8.27), dpi=100, facecolor='white')
        
        # ヘッダー
        # フォントプロパティを明示的に指定
        fig.text(0.05, 0.95, f"科学的分析レポート: {engine.name} 選手", fontproperties=title_font, fontsize=22, weight='bold', color='#1a237e')
        fig.text(0.05, 0.92, f"基準: {engine.long_run_min}分間走 {int(engine.long_run_dist)}m", fontproperties=title_font, fontsize=12, color='gray')

        # --- ① 左上 ---
        ax1 = fig.add_axes([0.05, 0.60, 0.40, 0.25])
        ax1.set_axis_off()
        ax1.set_title("① 基礎走力からのポテンシャル推計", fontproperties=title_font, loc='left', color='#0d47a1', fontsize=14, weight='bold')
        
        vo2 = engine.get_vo2_max()
        if potential_sec:
            p_m, p_s = divmod(potential_sec, 60)
            eval_txt = (
                f"● VO2 Max(最大酸素摂取量): {vo2:.1f} ml/kg/min\n"
                f"● {engine.target_dist}m 推定限界タイム: {int(p_m)}分{int(p_s):02d}秒\n\n"
                f"【評価】\n"
                f"あなたの心肺機能があれば、{engine.target_dist}mを\n"
                f"『{int(p_m)}分{int(p_s):02d}秒』で走る力があります。"
            )
        else:
            eval_txt = "※15分間走の距離が読み取れませんでした。"
        
        ax1.text(0, 0.8, eval_txt, fontproperties=title_font, fontsize=11, va='top', linespacing=1.6)

        # --- ② 右上 ---
        ax2 = fig.add_axes([0.50, 0.60, 0.45, 0.25])
        ax2.set_axis_off()
        ax2.set_title(f"② {engine.target_dist}m 実走データ", fontproperties=title_font, loc='left', color='#0d47a1', fontsize=14, weight='bold')
        
        if len(engine.tt_laps) > 0:
            col_labels = ["周回", "ラップ", "変動"]
            table_data = []
            for i, lap in enumerate(engine.tt_laps):
                diff = lap - engine.tt_laps[i-1] if i > 0 else 0
                mark = "▼" if diff >= 3 else ("▲" if diff <= -2 else "-")
                table_data.append([f"{i+1}", f"{lap:.1f}", mark])
            if len(table_data) > 10: table_data = table_data[:10]
            
            the_table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', colColours=['#e3f2fd']*3)
            the_table.scale(1, 1.3)
            # テーブル内のフォント適用
            if title_font:
                for cell in the_table.get_celld().values():
                    cell.set_text_props(fontproperties=title_font)
        else:
            ax2.text(0.1, 0.5, "実走データなし", fontproperties=title_font)

        # --- ③ 左下 ---
        ax3 = fig.add_axes([0.05, 0.10, 0.40, 0.40])
        ax3.set_axis_off()
        ax3.set_title("③ 能力別：目標ラップ表", fontproperties=title_font, loc='left', color='#0d47a1', fontsize=14, weight='bold')
        
        if potential_sec:
            base_pace = potential_sec / (engine.target_dist / 300) 
            headers = ["レベル", "目標タイム", "1周(300m)"]
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
            if title_font:
                for cell in t_table.get_celld().values():
                    cell.set_text_props(fontproperties=title_font)
        else:
            ax3.text(0.1, 0.5, "データ不足のため算出不能", fontproperties=title_font)

        # --- ④ 右下 ---
        ax4 = fig.add_axes([0.50, 0.10, 0.45, 0.40])
        ax4.set_axis_off()
        ax4.set_title("④ 戦術アドバイス", fontproperties=title_font, loc='left', color='#0d47a1', fontsize=14, weight='bold')
        
        advice = ""
        if len(engine.tt_laps) > 0 and potential_sec:
            theory_lap = potential_sec / len(engine.tt_laps)
            advice += f"あなたの心肺機能なら、1周{theory_lap:.1f}秒で押せます。\n\n"
            at_point = next((i+1 for i in range(1, len(engine.tt_laps)) if engine.tt_laps[i] - engine.tt_laps[i-1] >= 3.0), None)
            if at_point: advice += f"⚠️ {at_point}周目にペースダウンしています。\nここがスタミナの壁です。\n"
            advice += "\n👉 左の表の『理論値』または『挑戦圏』の\nラップを守って走ってみましょう。"
        elif potential_sec:
            advice += "実走データがありませんが、左の表が基準です。\n『安全圏』のペースで完走を目指してください。"
        else:
            advice += "データ不足のため分析できません。"

        rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#333', linewidth=1, transform=ax4.transAxes)
        ax4.add_patch(rect)
        ax4.text(0.05, 0.9, advice, fontproperties=title_font, fontsize=11, va='top', linespacing=1.6)

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
    
    st.markdown("""
    **【使い方】**
    15分間走(または12分間走)の記録と、3000m(2100m)の記録が書かれた用紙をアップロードしてください。
    """)
    
    uploaded_file = st.file_uploader("記録用紙をアップロード", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        with st.spinner("AIが解析中..."):
            try:
                # 画像の準備
                image = Image.open(uploaded_file)
                image = ImageOps.exif_transpose(image)
                st.image(image, caption="入力画像", width=200)
                
                img_byte_arr = io.BytesIO()
                image = image.convert('RGB')
                image.save(img_byte_arr, format='JPEG')
                
                # 解析実行
                data, error = analyze_image(img_byte_arr.getvalue())
                
                if data:
                    dashboard_img = ReportGenerator.create_dashboard(data)
                    if dashboard_img:
                        st.success("分析完了！")
                        st.image(dashboard_img, use_column_width=True)
                    else:
                        st.error("レポート描画失敗 (フォントエラーの可能性があります)")
                else:
                    st.error(f"解析失敗: {error}")
            except Exception as e:
                st.error(f"システムエラー: {e}")

if __name__ == "__main__":
    main()
