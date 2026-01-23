import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, json, os, re
import matplotlib.font_manager as fm
from PIL import Image, ImageOps
import google.generativeai as genai

# ==========================================
# 1. システム設定・準備（堅牢性確保）
# ==========================================
st.set_page_config(page_title="持久走データサイエンス", layout="wide")

# APIキーのクリーニング（事故防止）
raw_key = st.secrets.get("GEMINI_API_KEY", "")
API_KEY = str(raw_key).replace("\n", "").replace(" ", "").replace("　", "").replace('"', "").replace("'", "").strip()

if not API_KEY:
    st.error("【緊急】SecretsにAPIキーが設定されていません。")
    st.stop()

genai.configure(api_key=API_KEY)

# 日本語フォント（絶対リンク・固定住所）
@st.cache_resource
def load_japanese_font():
    import requests
    font_path = "NotoSansJP-Regular.ttf"
    # Google Fontsのコミットハッシュ指定（リンク切れ防止）
    url = "https://raw.githubusercontent.com/google/fonts/e3082f4d6d660086395b8d23e5959146522c7a52/ofl/notosansjp/NotoSansJP-Regular.ttf"
    try:
        if not os.path.exists(font_path):
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            with open(font_path, "wb") as f:
                f.write(response.content)
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Noto Sans JP'
        return fm.FontProperties(fname=font_path)
    except Exception as e:
        st.warning(f"フォント読み込み警告: {e}")
        return None

# ==========================================
# 2. AIエンジン（プロコーチの頭脳）
# ==========================================
def run_ai_analysis(image_obj):
    # モデル自動探索（利用可能なモデルからベストを選ぶ）
    try:
        models = list(genai.list_models())
        valid_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        
        # 優先順位: 1.5-flash (高速) -> 1.5-pro (高精度) -> その他
        target_model = next((m for m in valid_models if "1.5-flash" in m.lower()), None)
        if not target_model:
            target_model = next((m for m in valid_models if "1.5-pro" in m.lower()), None)
        if not target_model and valid_models:
            target_model = valid_models[0]
            
        if not target_model:
             return None, "利用可能なAIモデルが見つかりませんでした。"

        model = genai.GenerativeModel(target_model)

    except Exception as e:
        return None, f"モデル選択エラー: {e}"

    # ★ここが魂のプロンプト：AIを「記録員」ではなく「鬼コーチ」にする
    prompt = """
    あなたはオリンピック選手を育てる「陸上長距離の専門分析官」です。
    アップロードされた「持久走記録用紙」の画像を読み取り、以下の厳密なJSONデータを作成してください。

    【分析対象】
    用紙には「15分間走(または12分間走)」と「3000m(または2100m)のラップタイム」が書かれています。

    【出力JSONフォーマット】
    {
      "name": "選手名（読めなければ'選手'）",
      "long_run_min": 15または12（上段の分数。不明なら15とする）,
      "long_run_dist": 上段の距離(m)。数値のみ。(例: 4050),
      "target_dist": 下段の種目距離(m)。男子は3000、女子は2100が多い。(例: 3000),
      "tt_laps": [ラップタイム(秒)の数値リスト],
      "coach_comment": "ここには、ラップタイムの変動（中盤の落ち込み、ラストスパートの有無など）を具体的に指摘し、
                        生理学的な観点（AT値、乳酸の蓄積）と、次回のレースに向けた具体的な戦略（例：前半〇秒抑えるネガティブスプリット）を
                        150文字程度の『熱い』アドバイスとして書いてください。"
    }

    【注意】
    - 余計な解説は不要。JSONのみ出力すること。
    - 数字は半角。
    """

    try:
        response = model.generate_content(
            [prompt, image_obj],
            generation_config={"response_mime_type": "application/json"}
        )
        
        # JSONクリーニング
        text = response.text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0)), None
        else:
            return json.loads(text), None

    except Exception as e:
        return None, f"解析エラー: {e}"

# ==========================================
# 3. レポート描画（過去最高傑作を超えるデザイン）
# ==========================================
def create_report_image(data):
    fp = load_japanese_font()
    font_bold = {'fontproperties': fp, 'weight': 'bold'} if fp else {}
    font_reg = {'fontproperties': fp} if fp else {}
    
    # --- データ展開 ---
    name = data.get("name", "選手")
    l_min = int(data.get("long_run_min", 15))
    l_dist = float(data.get("long_run_dist", 0))
    t_dist = float(data.get("target_dist", 3000))
    laps = np.array([float(x) for x in data.get("tt_laps", [])])
    comment = data.get("coach_comment", "データ不足のため分析不能")

    # --- 科学的計算 ---
    # VO2Max推定 (クーパーテストの変形式)
    dist_12min = l_dist * (12 / l_min) if l_min > 0 else 0
    vo2_max = (dist_12min - 504.9) / 44.73 if dist_12min > 504.9 else 0
    
    # ターゲット距離の理論タイム (リーゲルの公式)
    t1_sec = l_min * 60
    pred_sec = t1_sec * (t_dist / l_dist)**1.06 if l_dist > 0 else 0
    
    # --- 描画開始 ---
    fig = plt.figure(figsize=(11.69, 8.27), facecolor='#f0f2f5', dpi=150) # 背景色を少しグレーに
    
    # タイトルエリア
    fig.text(0.05, 0.95, f"DATA SCIENCE ATHLETE REPORT", fontsize=20, color='#7f8c8d', **font_bold)
    fig.text(0.05, 0.90, f"{name} 選手｜持久走能力徹底分析", fontsize=28, color='#2c3e50', **font_bold)
    
    # ==========================================
    # エリア①：左上「科学的ポテンシャル評価」
    # ==========================================
    ax1 = fig.add_axes([0.05, 0.60, 0.42, 0.25])
    ax1.set_axis_off()
    
    # カードデザイン
    rect = plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#bdc3c7', linewidth=2, transform=ax1.transAxes)
    ax1.add_patch(rect)
    
    ax1.text(0.05, 0.9, "【生理学的エンジン性能】", fontsize=16, color='#2980b9', **font_bold)
    
    info_text = f"● 推定VO2 Max : {vo2_max:.1f} ml/kg/min\n"
    avg_pace = l_dist/l_min if l_min>0 else 0
    info_text += f"● {l_min}分間走 平均ペース : {int(avg_pace)} m/分\n"
    pace_1k = 1000 / avg_pace if avg_pace > 0 else 0
    info_text += f"● 1000m換算ペース : {int(pace_1k)}分{int((pace_1k%1)*60):02d}秒\n\n"
    
    if pred_sec > 0:
        pm, ps = divmod(pred_sec, 60)
        info_text += "【到達可能ポテンシャル】\n"
        info_text += f"★ {int(t_dist)}m 理論値 : {int(pm)}分{int(ps):02d}秒\n"
        info_text += "現在の心肺機能は、このタイムを出すための\n出力を既に備えています。"
    else:
        info_text += "※基準データ不足のため算出不可"
        
    ax1.text(0.05, 0.8, info_text, fontsize=13, va='top', linespacing=1.8, **font_reg)

    # ==========================================
    # エリア②：右上「精密ラップ解析」
    # ==========================================
    ax2 = fig.add_axes([0.50, 0.60, 0.45, 0.25])
    ax2.set_axis_off()
    ax2.text(0, 1.02, "【実戦ラップ推移】", fontsize=16, color='#2980b9', **font_bold)

    if len(laps) > 0:
        col_labels = ["周", "LAP(秒)", "通過", "評価"]
        cell_data = []
        cum_time = 0
        for i, l in enumerate(laps[:15]): # 最大15周
            cum_time += l
            cm, cs = divmod(cum_time, 60)
            
            if i == 0: eval_mark = "―"
            else:
                diff = l - laps[i-1]
                if diff > 2.0: eval_mark = "▼DOWN"
                elif diff < -1.0: eval_mark = "▲UP"
                else: eval_mark = "KEEP"
            
            cell_data.append([f"{i+1}", f"{l:.1f}", f"{int(cm)}:{int(cs):02d}", eval_mark])
        
        table = ax2.table(cellText=cell_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.4)
        
        # デザイン調整
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(color='white', weight='bold')
                if fp: cell.set_text_props(fontproperties=fp, color='white')
            elif col == 3:
                if "DOWN" in cell_data[row-1][3]: cell.set_text_props(color='#e74c3c', weight='bold')
                elif "UP" in cell_data[row-1][3]: cell.set_text_props(color='#27ae60', weight='bold')
            
            if fp and row > 0: cell.set_text_props(fontproperties=fp)

    # ==========================================
    # エリア③：左下「目標設定マトリクス」
    # ==========================================
    ax3 = fig.add_axes([0.05, 0.05, 0.42, 0.50])
    ax3.set_axis_off()
    ax3.text(0, 1.02, "【目標達成ペース配分表】", fontsize=16, color='#2980b9', **font_bold)

    if pred_sec > 0:
        levels = [
            ("現状維持", 1.05, "#ecf0f1"),
            ("自己ベスト", 1.00, "#d6eaf8"),
            ("県大会レベル", 0.96, "#aed6f1"),
            ("限界突破", 0.93, "#85c1e9")
        ]
        
        rows_3 = []
        rows_3.append(["周回", "維持", "PB更新", "上位", "限界"])
        
        dist_per_lap = 300 # トラック長仮定
        num_laps = int(t_dist / dist_per_lap)

        target_paces = []
        for _, factor, _ in levels:
            target_time = pred_sec * factor
            pace_per_lap = target_time / num_laps if num_laps > 0 else 0
            target_paces.append(pace_per_lap)

        for lap_i in range(1, num_laps + 1):
            row = [f"{lap_i*dist_per_lap}m"]
            for p in target_paces:
                cum = p * lap_i
                cm, cs = divmod(cum, 60)
                row.append(f"{int(cm)}:{int(cs):02d}")
            rows_3.append(row)
            
        table3 = ax3.table(cellText=rows_3, loc='center', cellLoc='center')
        table3.auto_set_font_size(False)
        table3.set_fontsize(11)
        table3.scale(1, 1.8)
        
        for (row, col), cell in table3.get_celld().items():
            if row == 0:
                cell.set_facecolor('#34495e')
                cell.set_text_props(color='white')
            elif col == 0:
                cell.set_facecolor('#bdc3c7')
            
            if col > 0 and row > 0:
                cell.set_facecolor(levels[col-1][2])
                
            if fp: cell.set_text_props(fontproperties=fp)
            if row==0 and fp: cell.set_text_props(fontproperties=fp, color='white')

    # ==========================================
    # エリア④：右下「戦略的アドバイス」
    # ==========================================
    ax4 = fig.add_axes([0.50, 0.05, 0.45, 0.50])
    ax4.set_axis_off()
    
    rect4 = plt.Rectangle((0, 0), 1, 1, facecolor='#fff3e0', edgecolor='#f39c12', linewidth=3, transform=ax4.transAxes)
    ax4.add_patch(rect4)
    
    ax4.text(0.05, 0.92, "【COACH'S TACTICAL ADVICE】", fontsize=16, color='#d35400', **font_bold)
    
    formatted_comment = ""
    for line in comment.split("。"):
        if line and line.strip(): formatted_comment += "▶ " + line.strip() + "。\n\n"
        
    ax4.text(0.05, 0.85, formatted_comment, fontsize=13, va='top', linespacing=1.7, **font_reg)

    # 保存
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    return buf

# ==========================================
# 4. メインUI
# ==========================================
st.title("🏃‍♂️ DATA SCIENCE ATHLETE ANALYSIS")
st.markdown("##### 過去の自分を超えるための、プロフェッショナル分析レポート")
st.write("記録用紙をアップロードしてください。AIがポテンシャルを最大限に引き出すための戦略を提示します。")

uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner("Analyzing performance data..."):
        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image).convert('RGB')
            
            data, err = run_ai_analysis(image)
            
            if data:
                st.success("Analysis Complete.")
                st.image(create_report_image(data), caption="分析レポート（長押しで保存）", use_column_width=True)
            else:
                st.error(f"解析エラー: {err}")
                
        except Exception as e:
            st.error(f"システムエラー: {e}")
