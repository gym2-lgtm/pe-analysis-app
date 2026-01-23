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

raw_key = st.secrets.get("GEMINI_API_KEY", "")
API_KEY = str(raw_key).replace("\n", "").replace(" ", "").replace("　", "").replace('"', "").replace("'", "").strip()

if not API_KEY:
    st.error("SecretsにAPIキーが設定されていません。")
    st.stop()

genai.configure(api_key=API_KEY)

# ==========================================
# 2. フォント設定（3重バックアップ）
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
# 3. AI解析エンジン（Area①の精度向上）
# ==========================================
def run_ai_analysis(image_obj):
    try:
        models = list(genai.list_models())
        valid = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        target = next((m for m in valid if "1.5-flash" in m), next((m for m in valid if "1.5-pro" in m), valid[0]))
        model = genai.GenerativeModel(target)
    except:
        return None, "AIモデル検索失敗"

    # プロンプト：Area①の数値を正確に取るため、種目（分）の判定を厳密にする
    prompt = """
    あなたは陸上競技の専門分析官です。画像の「持久走記録用紙」からデータを抽出し、JSONで出力してください。
    
    【重要】
    ・用紙には「複数回（1回目, 2回目...）」の記録がある場合があります。全て抽出してください。
    ・種目が「15分間走」か「12分間走」かを用紙から慎重に判断してください。
    ・もし3000mや2100mのTT記録があればそれも抽出してください。

    【JSON構造】
    {
      "name": "選手名",
      "record_type_minutes": 15 (または 12。数値で),
      "records": [
        {
          "attempt": 1,
          "distance": 4050, 
          "laps": [60, 62, 65...] (各周のラップ秒数)
        },
        {
          "attempt": 2,
          "distance": 4100,
          "laps": [...] 
        }
      ],
      "tt_record": {
         "distance": 3000 (または2100),
         "time": "10:30" (タイム文字列。なければ null)
      },
      "coach_advice": "15分/12分間走の推移（成長度）と、ラップタイムの落ち込み(AT閾値)に着目した専門的なアドバイス。TT記録があれば持久力とのバランスも言及。150文字程度。"
    }
    """

    try:
        response = model.generate_content([prompt, image_obj], generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text), None
    except Exception as e:
        return None, f"解析エラー: {e}"

# ==========================================
# 4. レポート描画（Area①修正版）
# ==========================================
def create_report_image(data):
    fp = load_japanese_font()
    font_main = fp if fp else None
    font_bold = fp if fp else None

    # --- データ整理 ---
    name = data.get("name", "選手")
    records = data.get("records", [])
    tt_rec = data.get("tt_record")
    advice = data.get("coach_advice", "")
    
    # 最新記録を取得
    if records:
        latest_rec = records[-1]
    else:
        latest_rec = {"distance": 0, "laps": []}
        
    l_dist = float(latest_rec.get("distance", 0))

    # ★Area①のロジック修正★
    # AIが読み取った分数
    base_min = int(data.get("record_type_minutes", 15))

    # 安全装置: 距離と時間からペースがおかしければ補正する
    # 例: 4000m走って12分だとキロ3分(100m18秒)を切る。一般生徒なら15分の間違いの可能性大。
    # 閾値を「キロ3分15秒ペース(100m 19.5秒)」に設定。これより速ければ15分とみなす（安全策）
    if l_dist > 0:
        calc_pace_12 = (12 * 60) / (l_dist / 100) # 12分と仮定した時の100mペース
        if calc_pace_12 < 19.5 and base_min == 12:
            base_min = 15 # 強制補正

    # ターゲット距離（男子3000, 女子2100）
    if base_min == 15:
        target_dist = 3000
    else:
        target_dist = 2100

    # 計算
    # 1. 100mペース
    pace_100m = (base_min * 60) / (l_dist / 100) if l_dist > 0 else 0
    # 2. 1000mペース（分:秒）
    pace_1k_sec = pace_100m * 10
    p1k_m, p1k_s = divmod(pace_1k_sec, 60)
    
    # 3. VO2Max (12分間走換算距離から算出)
    dist_12min = l_dist * (12 / base_min) if base_min > 0 else 0
    vo2_max = (dist_12min - 504.9) / 44.73 if dist_12min > 504.9 else 0
    
    # 4. ターゲット予測
    t1_sec = base_min * 60
    pred_sec = t1_sec * (target_dist / l_dist)**1.06 if l_dist > 0 else 0

    # --- 描画 ---
    fig = plt.figure(figsize=(11.69, 8.27), facecolor='white', dpi=150)
    
    # タイトル
    fig.text(0.05, 0.95, "ATHLETE PERFORMANCE REPORT", fontsize=16, color='#666', fontproperties=font_bold)
    fig.text(0.05, 0.90, f"{name} 選手 ｜ 持久走能力徹底分析", fontsize=24, color='#000', fontproperties=font_bold)

    # ------------------------------------------------
    # ① 左上：科学的ポテンシャル（整理・修正版）
    # ------------------------------------------------
    ax1 = fig.add_axes([0.05, 0.60, 0.35, 0.25])
    ax1.set_axis_off()
    
    # 枠線
    ax1.add_patch(plt.Rectangle((0,0), 1, 1, boxstyle='round,pad=0.02', facecolor='#f8f9fa', edgecolor='#bbb', transform=ax1.transAxes))

    # 見出し
    ax1.text(0.05, 0.92, "【① 科学的ポテンシャル診断】", fontsize=14, color='#1565c0', fontproperties=font_bold)

    # コンテンツ整理
    info_text = ""
    # 1. 測定結果（根拠）
    info_text += f"● 測定記録 ({base_min}分間走)\n"
    info_text += f"   距離: {int(l_dist)} m\n"
    
    # 2. ペース能力（1kmを主、100mを従に）
    info_text += f"● 平均巡航ペース\n"
    info_text += f"   1km換算 : {int(p1k_m)}分{int(p1k_s):02d}秒 /km\n"
    info_text += f"   (100m換算 : {pace_100m:.1f} 秒)\n"

    # 3. VO2Max
    info_text += f"● エンジン性能 (推定VO2Max)\n"
    info_text += f"   {vo2_max:.1f} ml/kg/min\n"

    # 4. ターゲット予測
    info_text += f"● {target_dist}m 到達目標タイム\n"
    if pred_sec > 0:
        pm, ps = divmod(pred_sec, 60)
        info_text += f"   {int(pm)}分{int(ps):02d}秒"
    else:
        info_text += "   算出不能"

    ax1.text(0.08, 0.82, info_text, fontsize=11, va='top', linespacing=1.6, fontproperties=font_main)


    # ------------------------------------------------
    # ② 右上〜右下：精密ラップ解析表（3回分比較）
    # ------------------------------------------------
    ax2 = fig.add_axes([0.42, 0.25, 0.55, 0.60]) 
    ax2.set_axis_off()
    ax2.text(0, 1.01, f"【② {base_min}分間走 ラップ推移 & AT閾値】", fontsize=14, color='#0d47a1', fontproperties=font_bold)

    if records:
        # 1回目、2回目... の列作成
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

        # 距離行
        dist_row = ["DIST"]
        for rec in records:
            d = rec.get("distance", "-")
            dist_row.extend([f"{d}m", ""])
        cell_data.append(dist_row)

        # テーブル
        table = ax2.table(cellText=cell_data, colLabels=cols, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.25)

        # デザイン & AT判定
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#263238')
                cell.set_text_props(color='white')
                if font_bold: cell.set_text_props(fontproperties=font_bold)
            elif r == len(cell_data): # 距離行
                cell.set_facecolor('#eceff1')
                if font_bold: cell.set_text_props(fontproperties=font_bold)
            else:
                if c > 0 and c % 2 != 0: # Lap列
                    rec_idx = (c - 1) // 2
                    laps = records[rec_idx].get("laps", [])
                    if r-1 < len(laps):
                        curr_lap = laps[r-1]
                        if r > 1:
                            prev_lap = laps[r-2]
                            if curr_lap - prev_lap >= AT_THRESHOLD:
                                cell.set_facecolor('#ffebee') # AT警告色
                                cell.set_text_props(color='#c62828', weight='bold')
            
            if font_main and r > 0:
                # 属性保持のため再設定は最小限に
                pass

    # ------------------------------------------------
    # ③ 左下：目標ペース配分表
    # ------------------------------------------------
    ax3 = fig.add_axes([0.05, 0.05, 0.35, 0.50])
    ax3.set_axis_off()
    ax3.text(0, 1.01, f"【③ {target_dist}m 目標ペース】", fontsize=14, color='#0d47a1', fontproperties=font_bold)

    if pred_sec > 0:
        levels = [("維持", 1.05), ("PB更新", 1.00), ("限界突破", 0.94)]
        cols3 = ["周回"] + [l[0] for l in levels]
        rows3 = []
        
        lap_len = 300 
        total_laps = int(target_dist / lap_len)
        targets = [pred_sec * l[1] for l in levels]
        
        for i in range(1, total_laps + 1):
            row = [f"{i*lap_len}m"]
            for tgt in targets:
                pass_time = tgt * (i / total_laps)
                pm, ps = divmod(pass_time, 60)
                row.append(f"{int(pm)}:{int(ps):02d}")
            rows3.append(row)
            
        table3 = ax3.table(cellText=rows3, colLabels=cols3, loc='center', cellLoc='center')
        table3.auto_set_font_size(False)
        table3.set_fontsize(9)
        table3.scale(1, 1.5)
        
        for (r, c), cell in table3.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1565c0')
                cell.set_text_props(color='white')
                if font_bold: cell.set_text_props(fontproperties=font_bold)
            elif c == 0:
                cell.set_facecolor('#eceff1')
            elif c == 3:
                cell.set_facecolor('#e3f2fd')

    # ------------------------------------------------
    # ④ 右下：専門アドバイス
    # ------------------------------------------------
    ax4 = fig.add_axes([0.42, 0.05, 0.55, 0.15])
    ax4.set_axis_off()
    
    rect = plt.Rectangle((0,0), 1, 1, facecolor='#fffde7', edgecolor='#fbc02d', linewidth=2, transform=ax4.transAxes)
    ax4.add_patch(rect)
    
    ax4.text(0.02, 0.85, "【④ COACH'S EYE / 専門的アドバイス】", fontsize=12, color='#ef6c00', fontproperties=font_bold)
    
    adv_text = advice.replace("。", "。\n")
    ax4.text(0.02, 0.65, adv_text, fontsize=10, va='top', linespacing=1.5, fontproperties=font_
