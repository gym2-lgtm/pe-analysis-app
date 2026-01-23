import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, json, os, re
import matplotlib.font_manager as fm
from PIL import Image, ImageOps
import google.generativeai as genai
import requests

# ==========================================
# 1. システム設定
# ==========================================
st.set_page_config(page_title="持久走データサイエンス", layout="wide")

# APIキー設定
raw_key = st.secrets.get("GEMINI_API_KEY", "")
API_KEY = str(raw_key).replace("\n", "").replace(" ", "").replace("　", "").replace('"', "").replace("'", "").strip()

if not API_KEY:
    st.error("SecretsにAPIキーが設定されていません。")
    st.stop()

genai.configure(api_key=API_KEY)

# ==========================================
# 2. フォント設定
# ==========================================
@st.cache_resource
def load_japanese_font():
    font_filename = "JP_Font.ttf"
    url_list = [
        "https://moji.or.jp/wp-content/ipafont/IPAexfont/ipaexg00401.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansjp/NotoSansJP-Regular.ttf",
        "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
    ]
    if os.path.exists(font_filename):
        try:
            fm.fontManager.addfont(font_filename)
            plt.rcParams['font.family'] = 'IPAexGothic'
            return fm.FontProperties(fname=font_filename)
        except:
            pass
    for url in url_list:
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            if response.status_code == 200:
                with open(font_filename, "wb") as f:
                    f.write(response.content)
                fm.fontManager.addfont(font_filename)
                return fm.FontProperties(fname=font_filename)
        except:
            continue
    return None

# ==========================================
# 3. AI解析エンジン
# ==========================================
def run_ai_analysis(image_obj):
    try:
        models = list(genai.list_models())
        valid = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        target = next((m for m in valid if "1.5-flash" in m), next((m for m in valid if "1.5-pro" in m), valid[0]))
        model = genai.GenerativeModel(target)
    except:
        return None, "AIモデル検索失敗"

    prompt = """
    あなたは陸上長距離の専門分析官です。画像の「持久走記録用紙」からデータを抽出し、JSONで出力してください。
    
    【重要ロジック】
    ・用紙に「15分間走」とあれば、対象は『男子』であり、目標距離は『3000m』です。
    ・用紙に「12分間走」とあれば、対象は『女子』であり、目標距離は『2100m』です。
    ・複数回の記録がある場合は全て抽出してください。
    
    【JSON構造】
    {
      "name": "選手名",
      "record_type_minutes": 15 (または 12),
      "records": [
        {
          "attempt": 1,
          "distance": 4050, 
          "laps": [60, 62, 65...]
        }
      ],
      "coach_advice": "AT閾値(ラップの急激な落ち込み)の分析と、フォームや粘りについてのコメントがあれば抽出。なければデータから予測される改善点を記述。120文字程度。"
    }
    """

    try:
        response = model.generate_content([prompt, image_obj], generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text), None
    except Exception as e:
        return None, f"解析エラー: {e}"

# ==========================================
# 4. レポート描画（レイアウト微調整・完成版）
# ==========================================
def create_report_image(data):
    fp = load_japanese_font()
    font_main = fp if fp else None
    font_bold = fp if fp else None

    # --- データ整理 ---
    name = data.get("name", "選手")
    records = data.get("records", [])
    advice = data.get("coach_advice", "")
    
    # ベスト記録特定
    best_rec = {"distance": 0, "laps": []}
    if records:
        best_rec = max(records, key=lambda x: float(x.get("distance", 0)))
    
    l_dist = float(best_rec.get("distance", 0))

    # 種目判定
    base_min = int(data.get("record_type_minutes", 15))
    if l_dist > 0 and base_min == 12:
        if (12 * 60) / (l_dist / 100) < 19.5: base_min = 15

    if base_min == 15:
        target_dist = 3000
    else:
        target_dist = 2100

    # --- 科学的計算 ---
    current_pace_100m = (base_min * 60) / (l_dist / 100) if l_dist > 0 else 0
    
    dist_12min = l_dist * (12 / base_min) if base_min > 0 else 0
    vo2_max = (dist_12min - 504.9) / 44.73 if dist_12min > 504.9 else 0
    
    # ターゲットタイム
    t1_sec = base_min * 60
    theoretical_sec = t1_sec * (target_dist / l_dist)**1.06 if l_dist > 0 else 0
    target_sec = theoretical_sec * 0.99 
    target_pace_100m = target_sec / (target_dist / 100) if target_dist > 0 else 0

    # --- VO2Maxポテンシャル推定 ---
    potential_3k_sec = 0
    if vo2_max > 0:
        potential_3k_sec = (11000 / vo2_max) * 3.2 
    pm_pot, ps_pot = divmod(potential_3k_sec, 60)

    vo2_msg = ""
    if vo2_max >= 62:
        vo2_msg = f"VO2Max {vo2_max:.1f}は、本来3000mを【{int(pm_pot)}分{int(ps_pot):02d}秒】前後で走れる極めて高い心肺能力です。スピード持久力を磨けば全国レベルも視野に入ります。"
    elif vo2_max >= 56:
        vo2_msg = f"VO2Max {vo2_max:.1f}は、3000m【{int(pm_pot)}分{int(ps_pot):02d}秒】相当のエンジン性能です。今のタイムとの差は『スピードへの慣れ』だけです。自信を持って攻めましょう。"
    elif vo2_max >= 48:
        vo2_msg = f"VO2Max {vo2_max:.1f}は、長距離ランナーとしての強固な土台を示しています。まずはインターバル走などで心肺に高い負荷をかけ、エンジンの出力を上げましょう。"
    else:
        vo2_msg = f"現在のVO2Maxは{vo2_max:.1f}です。まずは長い距離をゆっくり走るLSDトレーニングで毛細血管を増やし、酸素を取り込む器を大きくすることから始めましょう。"

    # 描画開始
    fig = plt.figure(figsize=(11.69, 8.27), facecolor='white', dpi=150)
    
    # ヘッダー
    fig.text(0.05, 0.95, "ATHLETE PERFORMANCE REPORT", fontsize=16, color='#7f8c8d', fontproperties=font_bold)
    fig.text(0.05, 0.90, f"{name} 選手 ｜ 持久走能力徹底分析", fontsize=26, color='#2c3e50', fontproperties=font_bold)

    # ----------------------------------------------------
    # ① 左上：科学的ポテンシャル
    # ----------------------------------------------------
    # 位置調整：高さを確保しつつ、Area3との被りを避ける位置
    # [left, bottom, width, height]
    ax1 = fig.add_axes([0.05, 0.61, 0.35, 0.25]) 
    ax1.set_axis_off()
    
    # 背景 (長方形)
    ax1.add_patch(plt.Rectangle((0,0), 1, 1, facecolor='#f4f6f7', edgecolor='#bdc3c7', transform=ax1.transAxes))
    
    ax1.text(0.05, 0.92, "【① 科学的ポテンシャル診断 (Best)】", fontsize=14, color='#2980b9', fontproperties=font_bold)

    p1k_curr = current_pace_100m * 10
    p1k_tgt = target_pace_100m * 10
    tm, ts = divmod(target_sec, 60)
    
    lines = [
        f"● 測定記録 ({base_min}分間走)",
        f"   距離: {int(l_dist)} m",
        f"   平均ペース: {int(p1k_curr//60)}'{int(p1k_curr%60):02d}/km",
        "",
        f"● エンジン性能 (推定VO2Max)",
        f"   {vo2_max:.1f} ml/kg/min",
        "",
        f"● {target_dist}m 挑戦目標タイム",
        f"   {int(tm)}分{int(ts):02d}秒",
        f"   設定ペース: {int(p1k_tgt//60)}'{int(p1k_tgt%60):02d}/km",
        "   (強度を上げて挑む設定)"
    ]
    
    # フォントサイズを微調整し、行間を詰めて収める
    ax1.text(0.05, 0.85, "\n".join(lines), fontsize=10.5, va='top', linespacing=1.6, fontproperties=font_main)

    # ----------------------------------------------------
    # ② 右上〜中：精密ラップ解析表
    # ----------------------------------------------------
    # 位置調整：上部ヘッダーとの衝突を避けるため少し下げる (bottom 0.42 -> topが0.90付近になるよう調整)
    # [0.43, 0.36, 0.52, 0.50] -> Top is 0.86. 安全圏。
    ax2 = fig.add_axes([0.43, 0.36, 0.52, 0.50]) 
    ax2.set_axis_off()
    ax2.text(0, 1.01, f"【② {base_min}分間走 ラップ推移 & AT閾値判定】", fontsize=14, color='#2980b9', fontproperties=font_bold)

    if records:
        cols = ["周"]
        for r in records:
            idx = r.get("attempt", "?")
            cols.extend([f"#{idx} Lap", f"#{idx} Split"])
        
        max_laps = max([len(r.get("laps", [])) for r in records]) if records else 0
        cell_data = []
        AT_THRESHOLD = 2.0 

        for i in range(max_laps):
            row = [f"{i+1}"]
            for rec in records:
                laps = rec.get("laps", [])
                if i < len(laps):
                    l_val = laps[i]
                    s_val = sum(laps[:i+1])
                    sm, ss = divmod(s_val, 60)
                    row.append(f"{l_val:.1f}")
                    row.append(f"{int(sm)}:{int(ss):02d}")
                else:
                    row.extend(["-", "-"])
            cell_data.append(row)

        dist_row = ["DIST"]
        for rec in records:
            d = rec.get("distance", "-")
            dist_row.extend([f"{d}m", ""])
        cell_data.append(dist_row)

        table = ax2.table(cellText=cell_data, colLabels=cols, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.25)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#34495e')
                cell.set_text_props(color='white')
                if font_bold: cell.set_text_props(fontproperties=font_bold)
            elif r == len(cell_data):
                cell.set_facecolor('#ecf0f1')
            else:
                if c > 0 and c % 2 != 0:
                    rec_idx = (c - 1) // 2
                    laps = records[rec_idx].get("laps", [])
                    if r-1 < len(laps):
                        curr_lap = laps[r-1]
                        if r > 1:
                            prev_lap = laps[r-2]
                            if curr_lap - prev_lap >= AT_THRESHOLD:
                                cell.set_facecolor('#fadbd8')
                                cell.set_text_props(color='#c0392b', weight='bold')
            if font_main and r > 0: pass

    # ----------------------------------------------------
    # ③ 左下：目標ペース配分表
    # ----------------------------------------------------
    # 位置調整：Area1の下端(0.61)より十分に下げる。
    # [0.05, 0.05, 0.35, 0.50] -> Top is 0.55. Area1 Bottom is 0.61. Gap is 0.06. Perfect.
    ax3 = fig.add_axes([0.05, 0.05, 0.35, 0.50]) 
    ax3.set_axis_off()
    ax3.text(0, 1.01, f"【③ {target_dist}m 目標ペース】", fontsize=14, color='#2980b9', fontproperties=font_bold)

    if target_sec > 0:
        levels = [("維持", 1.05), ("PB更新", 1.00), ("限界突破", 0.94)]
        cols3 = ["周回"] + [l[0] for l in levels]
        rows3 = []
        
        lap_len = 300 
        total_laps = int(target_dist / lap_len)
        targets = [target_sec * l[1] for l in levels]
        
        for i in range(1, total_laps + 1):
            row = [f"{i*lap_len}m"]
            for tgt in targets:
                pass_time = tgt * (i / total_laps)
                pm, ps = divmod(pass_time, 60)
                row.append(f"{int(pm)}:{int(ps):02d}")
            rows3.append(row)
            
        table3 = ax3.table(cellText=rows3, colLabels=cols3, loc='center', cellLoc='center')
        table3.auto_set_font_size(False)
        table3.set_fontsize(10)
        table3.scale(1, 1.55)
        
        for (r, c), cell in table3.get_celld().items():
            if r == 0:
                cell.set_facecolor('#2980b9')
                cell.set_text_props(color='white')
                if font_bold: cell.set_text_props(fontproperties=font_bold)
            elif c == 0:
                cell.set_facecolor('#ecf0f1')
            elif c == 3:
                cell.set_facecolor('#d6eaf8')

    # ----------------------------------------------------
    # ④ 右下：専門アドバイス
    # ----------------------------------------------------
    ax4 = fig.add_axes([0.43, 0.05, 0.52, 0.28])
    ax4.set_axis_off()
    
    ax4.add_patch(plt.Rectangle((0,0), 1, 1, facecolor='#fff9c4', edgecolor='#f1c40f', transform=ax4.transAxes))
    
    ax4.text(0.02, 0.88, "【④ COACH'S EYE / 専門的アドバイス】", fontsize=13, color='#d35400', fontproperties=font_bold)
    
    clean_advice = advice.replace("。", "。\n")
    final_text = f"■ {target_dist}mへの戦略\n{clean_advice}\n\n■ 生理学的評価\n{vo2_msg}"
    
    ax4.text(0.02, 0.80, final_text, fontsize=10, va='top', linespacing=1.5, fontproperties=font_main)

    # 保存
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    return buf

# ==========================================
# 5. メインUI
# ==========================================
st.title("Data Science Athlete Report")
st.write("記録用紙をアップロードしてください。")

uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner("AI分析中..."):
        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image).convert('RGB')
            
            data, err = run_ai_analysis(image)
            
            if data:
                st.success("作成完了")
                st.image(create_report_image(data), use_column_width=True)
            else:
                st.error(f"解析エラー: {err}")
        except Exception as e:
            st.error(f"システムエラー: {e}")
