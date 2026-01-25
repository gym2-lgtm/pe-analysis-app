import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io, json, os, requests
import matplotlib.font_manager as fm
from PIL import Image, ImageOps
import google.generativeai as genai

# ==========================================
# 1. システム設定 & フォント準備
# ==========================================
st.set_page_config(page_title="持久走データサイエンス", layout="wide")

# APIキー設定
raw_key = st.secrets.get("GEMINI_API_KEY", "")
API_KEY = str(raw_key).replace("\n", "").replace(" ", "").replace("　", "").replace('"', "").replace("'", "").strip()

if not API_KEY:
    st.error("SecretsにAPIキーが設定されていません。")
    st.stop()

genai.configure(api_key=API_KEY)

# 日本語フォントの確保（Streamlit Cloud対策・最強版）
@st.cache_resource
def load_japanese_font():
    font_filename = "JP_Font.ttf"
    url = "https://moji.or.jp/wp-content/ipafont/IPAexfont/ipaexg00401.ttf"
    
    # フォントファイルがなければダウンロード
    if not os.path.exists(font_filename):
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                with open(font_filename, "wb") as f:
                    f.write(response.content)
        except:
            pass
    
    # フォントプロパティを作成して返す
    if os.path.exists(font_filename):
        return fm.FontProperties(fname=font_filename)
    return None

# ==========================================
# 2. AI解析エンジン（エラー耐性・最強版）
# ==========================================
def get_safe_model_name():
    try:
        models = list(genai.list_models())
        valid_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        
        for m in valid_models:
            if "1.5-flash" in m: return m
        for m in valid_models:
            if "1.5-pro" in m: return m
        
        return valid_models[0] if valid_models else "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

def run_ai_analysis(image_obj):
    target_model = get_safe_model_name()
    
    try:
        model = genai.GenerativeModel(target_model)
        
        prompt = """
        あなたは陸上長距離のデータ分析官です。画像の「持久走記録用紙」を解析してください。

        【重要：データ抽出と論理チェック】
        1. 種目の判定
           - 基本は記述に従うが、記述がない場合は「ラップタイムの合計時間」や「距離」で判断する。
           - 距離記述があれば `total_dist` に数値を入れる。
           - 時間記録(3000m走など)なら `total_dist` は 3000 に固定。
        
        2. データ抽出 (records)
           - 「ラップタイム」は必ず全て抽出すること。これが種目判定の鍵になる。

        3. アドバイス (coach_advice)
           - 結果に基づき、次につながる前向きなアドバイスを140文字程度で。

        【JSON出力形式】
        {
          "name": "選手名",
          "record_type_minutes": 15,
          "race_category": "time", 
          "records": [
            {
              "attempt": 1, 
              "total_dist": 4050, 
              "total_time_str": "14:45",
              "laps": [91, 87, 89...]
            }
          ],
          "coach_advice": "アドバイステキスト"
        }
        """

        response = model.generate_content(
            [prompt, image_obj], 
            generation_config={"response_mime_type": "application/json"}
        )
        
        # ★エラー対策：JSONクリーニング
        raw_text = response.text
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        try:
            data = json.loads(clean_text)
        except:
            # 失敗したら元のテキストでトライ
            try:
                data = json.loads(raw_text)
            except:
                return None, "AIの応答を解析できませんでした。"

        # ★エラー対策：リスト型が返ってきた場合の強制変換
        if isinstance(data, list):
            # AIがリストだけ返してきた場合、それをrecordsとして包み込む
            data = {
                "records": data,
                "name": "選手",
                "record_type_minutes": 15,
                "race_category": "time",
                "coach_advice": "データからアドバイスを生成できませんでした。"
            }

        # タイムキーパー機能（自動補正）
        max_elapsed_sec = 0
        records = data.get("records", [])
        if not isinstance(records, list): 
            records = []
            data["records"] = []

        for rec in records:
            laps = rec.get("laps", [])
            if laps:
                total_lap_sec = sum(laps)
                if total_lap_sec > max_elapsed_sec: max_elapsed_sec = total_lap_sec
            
            if "total_time_str" in rec:
                try:
                    t_parts = str(rec["total_time_str"]).replace("分",":").replace("秒","").split(":")
                    if len(t_parts) >= 2:
                        t_sec = int(t_parts[0])*60 + int(t_parts[1])
                        if t_sec > max_elapsed_sec: max_elapsed_sec = t_sec
                except: pass
        
        if max_elapsed_sec > 750:
            if data.get("record_type_minutes") == 12:
                st.toast(f"⏱️ 補正発動: {int(max_elapsed_sec//60)}分台のため『15分間走』に変更しました")
                data["record_type_minutes"] = 15
        
        dist_check = 0
        if records:
            try:
                dist_check = float(str(records[0].get("total_dist", 0)).replace("m",""))
            except: pass
            
        if dist_check > 3200 and data.get("record_type_minutes") == 12:
             st.toast(f"📏 補正発動: {int(dist_check)}mのため『15分間走』に変更しました")
             data["record_type_minutes"] = 15

        return data, None

    except Exception as e:
        return None, f"解析エラー: {e}"

# ==========================================
# 3. レポート描画（文字化け・エラー対策済み）
# ==========================================
def create_report_image(data):
    # フォントプロパティの取得
    fp = load_japanese_font()
    
    def insert_newlines(text, length=30):
        if not text: return ""
        return '\n'.join([line[i:i+length] for line in text.split('\n') for i in range(0, len(line), length)])

    name = data.get("name", "選手")
    records = data.get("records", [])
    advice = data.get("coach_advice", "")
    race_cat = data.get("race_category", "time")
    base_min = int(data.get("record_type_minutes", 15))
    target_dist = 3000 if base_min == 15 else 2100

    # ベスト記録特定
    best_rec = {}
    best_l_dist = 0
    best_total_sec = 0
    
    if records:
        if race_cat == "distance":
            def get_sec(r):
                if "total_time_str" in r:
                    try:
                        p = str(r["total_time_str"]).replace("分",":").replace("秒","").split(":")
                        if len(p) >= 2:
                            return int(p[0])*60 + int(p[1])
                    except: pass
                return sum(r.get("laps", []))
            
            # リストが空でないか確認してからmin実行
            try:
                best_rec = min(records, key=lambda x: get_sec(x) if get_sec(x) > 0 else 9999)
                best_total_sec = get_sec(best_rec)
                best_l_dist = target_dist
            except:
                best_rec = records[0]
                best_total_sec = 0
                best_l_dist = target_dist
        else:
            try:
                best_rec = max(records, key=lambda x: float(str(x.get("total_dist", 0)).replace("m","")))
                best_l_dist = float(str(best_rec.get("total_dist", 0)).replace("m",""))
                best_total_sec = base_min * 60
            except:
                best_rec = records[0]
                best_l_dist = 0
                best_total_sec = base_min * 60

    # 計算
    if best_total_sec > 0 and best_l_dist > 0:
        pace_sec_per_km = best_total_sec / (best_l_dist / 1000)
    else:
        pace_sec_per_km = 0
    avg_pace_str = f"{int(pace_sec_per_km//60)}'{int(pace_sec_per_km%60):02d}/km"
    
    if race_cat == "distance":
        if best_total_sec > 0:
            equiv_dist_12min = (best_l_dist / best_total_sec) * (12 * 60)
            vo2_max = (equiv_dist_12min - 504.9) / 44.73
        else: vo2_max = 0
        ref_sec = best_total_sec
    else:
        dist_12min = best_l_dist * (12 / base_min) if base_min > 0 else 0
        vo2_max = (dist_12min - 504.9) / 44.73 if dist_12min > 504.9 else 0
        ref_sec = best_total_sec * (target_dist / best_l_dist)**1.06 if best_l_dist > 0 else 0

    rm, rs = divmod(ref_sec, 60)
    ref_time_str = f"{int(rm)}分{int(rs):02d}秒"
    ref_pace = ref_sec / (target_dist / 1000) if target_dist > 0 else 0
    ref_pace_str = f"{int(ref_pace//60)}'{int(ref_pace%60):02d}/km"

    potential_3k = (11000 / vo2_max) * 3.2 if vo2_max > 0 else 0
    pm_pot, ps_pot = divmod(potential_3k, 60)
    
    if vo2_max >= 62: vo2_msg = f"VO2Max {vo2_max:.1f}。高い心肺機能です。スピード持久力の強化でさらに伸びます。"
    elif vo2_max >= 56: vo2_msg = f"VO2Max {vo2_max:.1f}。3000m換算で{int(pm_pot)}分{int(ps_pot):02d}秒の実力があります。"
    elif vo2_max >= 48: vo2_msg = f"VO2Max {vo2_max:.1f}。土台はできています。中盤の粘りを強化しましょう。"
    else: vo2_msg = f"VO2Max {vo2_max:.1f}。まずはペースを一定に保つ練習から始めましょう。"

    # 描画
    fig = plt.figure(figsize=(11.69, 8.27), facecolor='white', dpi=150)
    
    title_mode = f"{target_dist}m走 (実戦)" if race_cat == "distance" else f"{base_min}分間走 (測定)"
    # ★修正：すべてのテキストに fontproperties=fp を明示的に指定
    fig.text(0.05, 0.96, "ATHLETE PERFORMANCE REPORT", fontsize=16, color='#7f8c8d', fontproperties=fp)
    fig.text(0.05, 0.91, f"{name} 選手 ｜ {title_mode} 能力分析", fontsize=26, color='#2c3e50', weight='bold', fontproperties=fp)

    # エリア1
    ax1 = fig.add_axes([0.05, 0.62, 0.35, 0.25]) 
    ax1.set_axis_off()
    ax1.add_patch(patches.Rectangle((0,0), 1, 1, facecolor='#f4f6f7', edgecolor='#bdc3c7', transform=ax1.transAxes))
    ax1.text(0.05, 0.90, "【① RESULT / 最高記録(Best)】", fontsize=14, color='#2980b9', weight='bold', fontproperties=fp)
    
    rec_val = f"{int(best_l_dist)} m" if race_cat=="time" else f"{int(best_total_sec//60)}'{int(best_total_sec%60):02d}"
    
    lines = [
        f"● 自己最高記録", 
        f"   記録: {rec_val}", 
        f"   平均ペース: {avg_pace_str}", 
        "",
        f"● エンジン性能 (推定VO2Max)", 
        f"   {vo2_max:.1f} ml/kg/min", 
        "",
        f"● {target_dist}m換算 参考記録", 
        f"   {ref_time_str}",
        f"   想定ペース: {ref_pace_str}"
    ]
    ax1.text(0.05, 0.82, "\n".join(lines), fontsize=10.5, va='top', linespacing=1.5, fontproperties=fp)

    # エリア2
    ax2 = fig.add_axes([0.45, 0.38, 0.50, 0.45])
    ax2.set_axis_off()
    ax2.text(0, 1.02, f"【② ラップ推移 & AT閾値判定】", fontsize=14, color='#2980b9', weight='bold', fontproperties=fp)

    if records:
        cols = ["周"]; cell_data = []; AT_THRESHOLD = 3.0
        for r in records: cols.extend([f"#{r.get('attempt',1)} Lap", f"Split"])
        max_laps = max([len(r.get("laps", [])) for r in records]) if records else 0
        
        for i in range(max_laps):
            row = [f"{i+1}"]
            for rec in records:
                laps = rec.get("laps", [])
                if i < len(laps):
                    sm, ss = divmod(sum(laps[:i+1]), 60)
                    row.extend([f"{laps[i]:.1f}", f"{int(sm)}:{int(ss):02d}"])
                else: row.extend(["-", "-"])
            cell_data.append(row)

        dist_row = ["DIST"]
        for rec in records:
            d = rec.get("total_dist", "-")
            if race_cat == "distance": d = target_dist
            dist_row.extend([f"{d}m", ""])
        cell_data.append(dist_row)

        table = ax2.table(cellText=cell_data, colLabels=cols, loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.25)
        
        # ★修正：テーブル内のセルにも確実にフォント適用
        for (r, c), cell in table.get_celld().items():
            cell.set_text_props(fontproperties=fp)
            if r == 0: 
                cell.set_facecolor('#34495e')
                cell.set_text_props(color='white', fontproperties=fp)
            elif r == len(cell_data): 
                cell.set_facecolor('#ecf0f1')
                cell.set_text_props(weight='bold', fontproperties=fp)
            elif c > 0 and c % 2 != 0: 
                rec_idx = (c - 1) // 2
                laps = records[rec_idx].get("laps", [])
                if r > 1 and r-1 < len(laps):
                    curr = laps[r-1]; prev = laps[r-2]
                    if curr - prev >= AT_THRESHOLD:
                         cell.set_facecolor('#fadbd8')
                         cell.set_text_props(color='#c0392b', weight='bold', fontproperties=fp)

    # エリア3
    ax3 = fig.add_axes([0.05, 0.05, 0.35, 0.45]) 
    ax3.set_axis_off()
    ax3.text(0, 1.01, f"【③ {target_dist}m 目標ラップ表】", fontsize=14, color='#2980b9', weight='bold', fontproperties=fp)
    
    levels = [("維持", 1.05), ("目標", 1.00), ("突破", 0.94)]
    cols3 = ["周回"] + [l[0] for l in levels]
    rows3 = []
    lap_len = 300
    total_laps = int(target_dist / lap_len)
    
    tgt_sec = ref_sec 
    
    for i in range(1, total_laps + 1):
        row = [f"{i*lap_len}m"]
        for _, factor in levels:
            t = tgt_sec * factor * (i / total_laps)
            pm, ps = divmod(t, 60)
            row.append(f"{int(pm)}:{int(ps):02d}")
        rows3.append(row)
        
    table3 = ax3.table(cellText=rows3, colLabels=cols3, loc='upper center', cellLoc='center')
    table3.auto_set_font_size(False); table3.set_fontsize(10); table3.scale(1, 1.55)
    
    # ★修正：テーブル3にもフォント適用
    for (r, c), cell in table3.get_celld().items():
        cell.set_text_props(fontproperties=fp)
        if r == 0: 
            cell.set_facecolor('#2980b9')
            cell.set_text_props(color='white', fontproperties=fp)
        elif c == 3: cell.set_facecolor('#d6eaf8')

    # エリア4
    ax4 = fig.add_axes([0.43, 0.05, 0.52, 0.30])
    ax4.set_axis_off()
    ax4.add_patch(patches.Rectangle((0,0), 1, 1, facecolor='#fff9c4', edgecolor='#f1c40f', transform=ax4.transAxes))
    ax4.text(0.02, 0.88, "【④ COACH'S EYE / レース講評】", fontsize=13, color='#d35400', weight='bold', fontproperties=fp)
    
    # ★修正：SyntaxErrorの原因となったf-string内の改行処理を分離
    formatted_advice = advice.replace('。', '。\n')
    final_text_raw = f"■ アドバイス\n{formatted_advice}\n\n■ 生理学的評価\n{vo2_msg}"
    
    final_text_ready = insert_newlines(final_text_raw, 30)
    
    ax4.text(0.02, 0.82, final_text_ready, fontsize=10, va='top', linespacing=1.5, fontproperties=fp)

    # 保存
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    return buf

# ==========================================
# 4. メインUI
# ==========================================
st.title("Data Science Athlete Report")
st.write("記録用紙をアップロードしてください。（自動補正機能搭載）")

uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner("AI解析中（タイムキーパー作動中）..."):
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
