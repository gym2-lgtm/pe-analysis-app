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

# APIキーのクリーニング
raw_key = st.secrets.get("GEMINI_API_KEY", "")
API_KEY = str(raw_key).replace("\n", "").replace(" ", "").replace("　", "").replace('"', "").replace("'", "").strip()

if not API_KEY:
    st.error("SecretsにAPIキーが設定されていません。")
    st.stop()

genai.configure(api_key=API_KEY)

# ==========================================
# 2. フォント設定（3段構えのバックアップ）
# ==========================================
@st.cache_resource
def load_japanese_font():
    font_filename = "JP_Font.ttf"
    
    # 試行するURLリスト（上から順に試す）
    # 1. Google Fonts (Variable版 - 現在の主流)
    # 2. Google Fonts (Static版 - バックアップ)
    # 3. IPAexゴシック (ド定番の予備)
    url_list = [
        "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansjp/NotoSansJP-Regular.ttf",
        "https://ipafont.ipa.go.jp/IPAexfont/ipaexg00401.ttf"
    ]
    
    if os.path.exists(font_filename):
        try:
            fm.fontManager.addfont(font_filename)
            plt.rcParams['font.family'] = 'Noto Sans JP'
            return fm.FontProperties(fname=font_filename)
        except:
            pass # 壊れていたら再ダウンロード

    # ダウンロード試行ループ
    for url in url_list:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                with open(font_filename, "wb") as f:
                    f.write(response.content)
                fm.fontManager.addfont(font_filename)
                # フォント名を特定せずにファイルパスからロードさせる
                return fm.FontProperties(fname=font_filename)
        except Exception:
            continue # 次のURLへ

    return None # 全滅した場合（英語になるがエラーでは止まらない）

# ==========================================
# 3. AI解析エンジン
# ==========================================
def run_ai_analysis(image_obj):
    # モデル自動探索
    try:
        models = list(genai.list_models())
        valid_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        
        # 優先順位: 1.5-flash -> 1.5-pro
        target_model = next((m for m in valid_models if "1.5-flash" in m), None)
        if not target_model:
            target_model = next((m for m in valid_models if "1.5-pro" in m), valid_models[0])
            
        model = genai.GenerativeModel(target_model)
    except:
        return None, "AIモデルの検索に失敗しました。"

    # プロンプト（IMG_5066のクオリティを目指す）
    prompt = """
    あなたはプロの陸上競技分析官です。画像の「持久走記録用紙」からデータを読み取り、JSONのみ出力してください。

    【JSON構造】
    {
      "name": "選手名（不明なら'選手'）",
      "long_run_min": 15または12（上段の分数。不明なら15）,
      "long_run_dist": 上段の距離(m)。数値のみ,
      "target_dist": 下段の種目距離(m)。3000または2100,
      "tt_laps": [ラップタイム(秒)の数値リスト],
      "coach_comment": "ここには、ラップの変化（中盤のタレ、ラストスパート等）を指摘し、AT値(無酸素性作業閾値)の観点から次回のペース配分を具体的に提案する150文字程度のアドバイス。"
    }
    """

    try:
        response = model.generate_content(
            [prompt, image_obj],
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text), None
    except Exception as e:
        return None, f"解析エラー: {e}"

# ==========================================
# 4. レポート描画（IMG_5066完全再現）
# ==========================================
def create_report_image(data):
    fp = load_japanese_font()
    # フォントプロパティの設定
    if fp:
        font_main = fp
        font_bold = fp # 太字用ファイルがない場合は同じものを使う
    else:
        font_main = None
        font_bold = None

    # データ展開
    name = data.get("name", "選手")
    l_min = int(data.get("long_run_min", 15))
    l_dist = float(data.get("long_run_dist", 0))
    t_dist = float(data.get("target_dist", 3000))
    laps = np.array([float(x) for x in data.get("tt_laps", [])])
    comment = data.get("coach_comment", "")

    # 計算
    dist_12min = l_dist * (12 / l_min) if l_min > 0 else 0
    vo2_max = (dist_12min - 504.9) / 44.73 if dist_12min > 504.9 else 0
    t1_sec = l_min * 60
    pred_sec = t1_sec * (t_dist / l_dist)**1.06 if l_dist > 0 else 0

    # 描画キャンバス
    fig = plt.figure(figsize=(11.69, 8.27), facecolor='white', dpi=150)
    
    # --- タイトル ---
    fig.text(0.05, 0.95, "SCIENTIFIC RUNNING ANALYSIS", fontsize=18, color='#555', fontproperties=font_bold)
    fig.text(0.05, 0.90, f"{name} 選手｜持久走能力分析レポート", fontsize=26, color='#000', fontproperties=font_bold)

    # --- 左上：科学的データ評価 ---
    ax1 = fig.add_axes([0.05, 0.55, 0.42, 0.30])
    ax1.set_axis_off()
    
    # 評価テキスト
    info = "【生理学的データ】\n"
    info += f"● 推定VO2Max : {vo2_max:.1f} ml/kg/min\n"
    avg_pace_sec = (l_min * 60) / (l_dist/100) if l_dist > 0 else 0
    info += f"● 100m平均ペース : {avg_pace_sec:.1f} 秒\n"
    if pred_sec > 0:
        pm, ps = divmod(pred_sec, 60)
        info += f"● {int(t_dist)}m 予測タイム : {int(pm)}分{int(ps):02d}秒\n"
    
    info += "\n【専門的評価】\n"
    if vo2_max > 60: info += "心肺機能は極めて高い水準にあります。\n"
    elif vo2_max > 50: info += "上位レベルを目指せる十分な心肺機能です。\n"
    else: info += "基礎的な持久力強化がタイム短縮の鍵です。\n"
    
    info += f"予測タイム({int(pm)}:{int(ps):02d})をターゲットに、\nイーブンペースを刻む練習が有効です。"

    ax1.text(0.02, 0.95, info, fontsize=12, va='top', linespacing=1.8, fontproperties=font_main)
    # 枠線
    ax1.add_patch(plt.Rectangle((0,0),1,1, fill=False, edgecolor='#ccc', transform=ax1.transAxes))


    # --- 右上：精密ラップ表 ---
    ax2 = fig.add_axes([0.50, 0.55, 0.45, 0.30])
    ax2.set_axis_off()
    ax2.text(0, 1.02, f"【{int(t_dist)}m ラップ分析】", fontsize=14, color='#0d47a1', fontproperties=font_bold)

    if len(laps) > 0:
        col_labels = ["周", "LAP", "通過", "評価"]
        cell_data = []
        cum = 0
        for i, l in enumerate(laps[:12]):
            cum += l
            cm, cs = divmod(cum, 60)
            
            eval_s = "―"
            if i > 0:
                diff = l - laps[i-1]
                if diff > 2: eval_s = "▼遅"
                elif diff < -1: eval_s = "▲速"
                else: eval_s = "OK"
            
            cell_data.append([f"{i+1}", f"{l:.1f}", f"{int(cm)}:{int(cs):02d}", eval_s])
            
        table = ax2.table(cellText=cell_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.3)
        
        # 色付け（IMG_5066風の黒ヘッダー）
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#222')
                cell.set_text_props(color='white', weight='bold')
                if font_bold: cell.set_text_props(fontproperties=font_bold, color='white')
            elif c == 3:
                if "遅" in cell_data[r-1][3]: cell.set_text_props(color='red', weight='bold')
                elif "速" in cell_data[r-1][3]: cell.set_text_props(color='blue', weight='bold')
            
            if font_main and r > 0: cell.set_text_props(fontproperties=font_main)

    # --- 左下：目標ペース表（青い表） ---
    ax3 = fig.add_axes([0.05, 0.05, 0.42, 0.45])
    ax3.set_axis_off()
    ax3.text(0, 1.02, "【目標通過タイム表】", fontsize=14, color='#0d47a1', fontproperties=font_bold)

    if pred_sec > 0:
        # レベル設定
        levels = [("維持", 1.05), ("PB更新", 1.00), ("大幅更新", 0.96), ("限界", 0.92)]
        cols = ["周回"] + [l[0] for l in levels]
        
        lap_d = 300 # トラック長
        rows_3 = []
        
        # ヘッダーごとのターゲットタイム(合計)
        targets = [pred_sec * l[1] for l in levels]
        
        for i in range(1, 11): # 10周分
            row = [f"{i*lap_d}m"]
            for tgt in targets:
                # この周回での通過タイム
                pass_time = tgt * (i*lap_d / t_dist)
                pm, ps = divmod(pass_time, 60)
                row.append(f"{int(pm)}:{int(ps):02d}")
            rows_3.append(row)
            
        table3 = ax3.table(cellText=rows_3, colLabels=cols, loc='center', cellLoc='center')
        table3.auto_set_font_size(False)
        table3.set_fontsize(10)
        table3.scale(1, 1.6)
        
        # 青系デザイン
        colors = ["#fff", "#cfd8dc", "#90caf9", "#42a5f5", "#1e88e5"]
        for (r, c), cell in table3.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1565c0')
                cell.set_text_props(color='white')
                if font_bold: cell.set_text_props(fontproperties=font_bold, color='white')
            elif c == 0:
                cell.set_facecolor('#eceff1')
            else:
                # 列ごとに色を変えるか、シンプルにするか
                pass
            if font_main and r>0: cell.set_text_props(fontproperties=font_main)


    # --- 右下：戦術アドバイス ---
    ax4 = fig.add_axes([0.50, 0.05, 0.45, 0.45])
    ax4.set_axis_off()
    ax4.text(0, 1.02, "【科学的分析と実戦戦術】", fontsize=14, color='#0d47a1', fontproperties=font_bold)
    
    # 枠
    ax4.add_patch(plt.Rectangle((0,0), 1, 1, facecolor='white', edgecolor='#333', transform=ax4.transAxes))
    
    clean_comment = comment.replace("。", "。\n\n")
    ax4.text(0.05, 0.90, clean_comment, fontsize=11, va='top', linespacing=1.6, fontproperties=font_main)

    # 保存
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    return buf

# ==========================================
# 5. メインUI
# ==========================================
st.title("Data Science Athlete Report")
st.markdown("記録用紙をアップロードしてください。")

uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner("AI分析中..."):
        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image).convert('RGB')
            
            data, err = run_ai_analysis(image)
            
            if data:
                st.success("分析完了")
                st.image(create_report_image(data), use_column_width=True)
            else:
                st.error(f"エラー: {err}")
        except Exception as e:
            st.error(f"システムエラー: {e}")
