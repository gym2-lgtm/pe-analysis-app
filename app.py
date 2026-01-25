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
# 2. 日本語フォント準備（Google Fonts版・鉄壁）
# ==========================================
@st.cache_resource
def get_font_prop():
    # Google Fonts (Noto Sans JP) をGitHubから直接ダウンロード
    font_filename = "NotoSansJP-Regular.ttf"
    url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
    
    # ファイルが存在しない、またはサイズが小さすぎる（失敗）場合は再ダウンロード
    if not os.path.exists(font_filename) or os.path.getsize(font_filename) < 1000:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(font_filename, "wb") as f:
                    f.write(response.content)
        except:
            pass
            
    # フォントプロパティを生成
    if os.path.exists(font_filename):
        return fm.FontProperties(fname=font_filename)
    return None

# ==========================================
# 3. AI解析エンジン
# ==========================================
def get_safe_model_name():
    try:
        models = list(genai.list_models())
        valid_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        for m in valid_models:
            if "1.5-flash" in m: return m
        return valid_models[0] if valid_models else "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

def run_ai_analysis(image_obj):
    target_model = get_safe_model_name()
    try:
        model = genai.GenerativeModel(target_model)
        prompt = """
        あなたは陸上長距離のデータ分析官です。画像の「持久走記録用紙」を解析してください。
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
        response = model.generate_content([prompt, image_obj], generation_config={"response_mime_type": "application/json"})
        
        raw_text = response.text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(raw_text)
        except:
            return None, "データ解析失敗"

        # リスト型ガード
        if isinstance(data, list):
            data = {"records": data, "name": "選手", "record_type_minutes": 15, "race_category": "time", "coach_advice": ""}

        # タイムキーパー（自動補正）
        max_elapsed_sec = 0
        records = data.get("records", [])
        if not isinstance(records, list): records = []
        
        for rec in records:
            laps = rec.get("laps", [])
            if laps:
                val = sum(laps)
                if val > max_elapsed_sec: max_elapsed_sec = val
            if "total_time_str" in rec:
                try:
                    parts = str(rec["total_time_str"]).replace("分",":").replace("秒","").split(":")
                    if len(parts)>=2:
                        val = int(parts[0])*60 + int(parts[1])
                        if val > max_elapsed_sec: max_elapsed_sec = val
                except: pass
        
        if max_elapsed_sec > 750 and data.get("record_type_minutes") == 12:
            st.toast(f"⏱️ 補正: {int(max_elapsed_sec//60)}分台のため『15分間走』に変更")
            data["record_type_minutes"] = 15
            
        return data, None
    except Exception as e:
        return None, f"エラー: {e}"

# ==========================================
# 4. レポート描画（フォント強制適用・完全版）
# ==========================================
def create_report_image(data):
    # 日本語フォントプロパティを取得
    fp = get_font_prop()
    
    def insert_newlines(text, length=30):
        if not text: return ""
        return '\n'.join([line[i:i+length] for line in text.split('\n') for i in range(0, len(line), length)])

    name = data.get("name", "選手")
    records = data.get("records", [])
    
    # ★修正：アドバイスがNoneの場合の安全策
    raw_advice = data.get("coach_advice")
    if raw_advice is None:
        advice = "データから十分な情報が得られませんでした。"
    else:
        advice = str(raw_advice)

    race_cat = data.get("race_category", "time")
    base_min = int(data.get("record_type_minutes", 15))
    target_dist = 3000 if base_min == 15 else 2100

    # ベスト記録計算
    best_rec = {}
    best_l_dist = 0
    best_total_sec = 0
    if records:
        if race_cat == "distance":
            def get_sec(r):
                try:
                    p = str(r.get("total_time_str","")).replace("分",":").replace("秒","").split(":")
                    if len(p)>=2: return int(p[0])*60 + int(p[1])
                except: pass
                return sum(r.get("laps", []))
            try:
                best_rec = min(records, key=lambda x: get_sec(x) if get_sec(x)>0 else 9999)
                best_total_sec = get_sec(best_rec)
                best_l_dist = target_dist
            except: pass
        else:
            try:
                def get_d(r): return float(str(r.get("total_dist",0)).replace("m","").replace(",",""))
                best_rec = max(records, key=get_d)
                best_l_dist = get_d(best_rec)
                best_total_sec = base_min * 60
            except: pass

    pace_sec = best_total_sec / (best_l_dist/1000) if best_l_dist>0 else 0
    avg_pace = f"{int(pace_sec//60)}'{int(pace_sec%60):02d}/km"
    
    vo2_max = 0
    if race_cat == "distance":
        if best_total_sec>0:
            equiv = (best_l_dist/best_total_sec)*(12*60)
            vo2_max = (equiv - 504.9)/44.73
        ref_sec = best_total_sec
    else:
        d12 = best_l_dist*(12/base_min) if base_min>0 else 0
        vo2_max = (d12 - 504.9)/44.73
        ref_sec = best_total_sec * (target_dist/best_l_dist)**1.06 if best_l_dist>0 else 0

    rm, rs = divmod(ref_sec, 60)
    ref_str = f"{int(rm)}分{int(rs):02d}秒"
    rp = ref_sec/(target_dist/1000) if target_dist>0 else 0
    rp_str = f"{int(rp//60)}'{int(rp%60):02d}/km"
    
    pot_3k = (11000/vo2_max)*3.2 if vo2_max>0 else 0
    pm, ps = divmod(pot_3k, 60)
    vo2_msg = f"VO2Max {vo2_max:.1f}。3000m換算{int(pm)}分{int(ps):02d}秒相当。" if vo2_max>0 else "計測不能"

    # --- 描画開始 ---
    fig = plt.figure(figsize=(11.69, 8.27), facecolor='white', dpi=150)
    
    # ★重要：すべてのテキストに fontproperties=fp を渡す
    t_mode = f"{target_dist}m走" if race_cat=="distance" else f"{base_min}分間走"
    fig.text(0.05, 0.96, "ATHLETE PERFORMANCE REPORT", fontsize=16, color='#7f8c8d', fontproperties=fp)
    fig.text(0.05, 0.91, f"{name} 選手 ｜ {t_mode} 能力分析", fontsize=26, color='#2c3e50', weight='bold', fontproperties=fp)

    # Area 1
    ax1 = fig.add_axes([0.05, 0.62, 0.35, 0.25])
    ax1.set_axis_off()
    ax1.add_patch(patches.Rectangle((0,0), 1, 1, facecolor='#f4f6f7', edgecolor='#bdc3c7', transform=ax1.transAxes))
    ax1.text(0.05, 0.90, "【① RESULT / 最高記録(Best)】", fontsize=14, color='#2980b9', weight='bold', fontproperties=fp)
    
    rec_txt = f"{int(best_l_dist)} m" if race_cat=="time" else f"{int(best_total_sec//60)}'{int(best_total_sec%60):02d}"
    lines = [
        f"● 自己最高記録",
        f"   記録: {rec_txt}",
        f"   平均ペース: {avg_pace}",
        "",
        f"● エンジン性能 (推定VO2Max)",
        f"   {vo2_max:.1f} ml/kg/min",
        "",
        f"● {target_dist}m換算 参考記録",
        f"   {ref_str}",
        f"   想定ペース: {rp_str}"
    ]
    ax1.text(0.05, 0.82, "\n".join(lines), fontsize=10.5, va='top', linespacing=1.5, fontproperties=fp)

    # Area 2
    ax2 = fig.add_axes([0.45, 0.38, 0.50, 0.45])
    ax2.set_axis_off()
    ax2.text(0, 1.02, "【② ラップ推移 & AT閾値判定】", fontsize=14, color='#2980b9', weight='bold', fontproperties=fp)
    
    if records:
        cols = ["周"]
        cell_data = []
        for r in records: 
            idx = r.get('attempt','?')
            cols.extend([f"#{idx} Lap", "Split"])
        
        max_len = max([len(r.get("laps",[])) for r in records]) if records else 0
        for i in range(max_len):
            row = [f"{i+1}"]
            for rec in records:
                laps = rec.get("laps", [])
                if i < len(laps):
                    sm, ss = divmod(sum(laps[:i+1]), 60)
                    row.extend([f"{laps[i]:.1f}", f"{int(sm)}:{int(ss):02d}"])
                else: row.extend(["-","-"])
            cell_data.append(row)
            
        dist_row = ["DIST"]
        for rec in records:
            d = rec.get("total_dist", "-")
            if race_cat=="distance": d=target_dist
            dist_row.extend([f"{d}m", ""])
        cell_data.append(dist_row)

        table = ax2.table(cellText=cell_data, colLabels=cols, loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.25)
        
        # ★重要：テーブルセルへのフォント適用
        for (r,c), cell in table.get_celld().items():
            cell.set_text_props(fontproperties=fp)
            if r==0:
                cell.set_facecolor('#34495e')
                cell.set_text_props(color='white', fontproperties=fp)
            elif r==len(cell_data):
                cell.set_facecolor('#ecf0f1')
                cell.set_text_props(weight='bold', fontproperties=fp)
            elif c>0 and c%2!=0:
                rec_idx = (c-1)//2
                laps = records[rec_idx].get("laps",[])
                if r>1 and r-1 < len(laps):
                    curr=laps[r-1]; prev=laps[r-2]
                    if curr-prev >= 3.0:
                        cell.set_facecolor('#fadbd8')
                        cell.set_text_props(color='#c0392b', weight='bold', fontproperties=fp)

    # Area 3
    ax3 = fig.add_axes([0.05, 0.05, 0.35, 0.45])
    ax3.set_axis_off()
    ax3.text(0, 1.01, f"【③ {target_dist}m 目標ラップ表】", fontsize=14, color='#2980b9', weight='bold', fontproperties=fp)
    
    levels = [("維持", 1.05), ("目標", 1.00), ("突破", 0.94)]
    cols3 = ["周回"] + [l[0] for l in levels]
    rows3 = []
    lap_len = 300
    total_laps = int(target_dist/lap_len)
    
    for i in range(1, total_laps+1):
        row = [f"{i*lap_len}m"]
        for _, fac in levels:
            t = ref_sec * fac * (i/total_laps)
            pm, ps = divmod(t, 60)
            row.append(f"{int(pm)}:{int(ps):02d}")
        rows3.append(row)
    
    table3 = ax3.table(cellText=rows3, colLabels=cols3, loc='upper center', cellLoc='center')
    table3.auto_set_font_size(False); table3.set_fontsize(10); table3.scale(1, 1.55)
    for (r,c), cell in table3.get_celld().items():
        cell.set_text_props(fontproperties=fp)
        if r==0: 
            cell.set_facecolor('#2980b9')
            cell.set_text_props(color='white', fontproperties=fp)
        elif c==3: cell.set_facecolor('#d6eaf8')

    # Area 4
    ax4 = fig.add_axes([0.43, 0.05, 0.52, 0.30])
    ax4.set_axis_off()
    ax4.add_patch(patches.Rectangle((0,0), 1, 1, facecolor='#fff9c4', edgecolor='#f1c40f', transform=ax4.transAxes))
    ax4.text(0.02, 0.88, "【④ COACH'S EYE / レース講評】", fontsize=13, color='#d35400', weight='bold', fontproperties=fp)
    
    clean_adv = advice.replace("。", "。\n")
    final_txt = f"■ アドバイス\n{clean_adv}\n\n■ 生理学的評価\n{vo2_msg}"
    ax4.text(0.02, 0.82, insert_newlines(final_txt, 30), fontsize=10, va='top', linespacing=1.5, fontproperties=fp)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    return buf

# ==========================================
# 5. UI
# ==========================================
st.title("Data Science Athlete Report")
st.write("記録用紙をアップロードしてください。（自動補正機能搭載・日本語対応）")
uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner("AI解析中..."):
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
