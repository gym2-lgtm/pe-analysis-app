import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, json, os, requests, base64
import matplotlib.font_manager as fm
from PIL import Image, ImageOps
import textwrap

# ==========================================
# 1. システム設定
# ==========================================
st.set_page_config(page_title="持久走能力徹底分析", layout="wide")

# APIキー取得
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = os.environ.get("GEMINI_API_KEY", "")

if not api_key:
    st.error("【エラー】APIキーが設定されていません。Secretsを確認してください。")
    st.stop()

# ==========================================
# 2. 日本語フォント確保 (文字化け防止)
# ==========================================
@st.cache_resource
def get_jp_font():
    font_dir = "fonts"
    font_name = "NotoSansJP-Regular.ttf"
    font_path = os.path.join(font_dir, font_name)
    if not os.path.exists(font_dir): os.makedirs(font_dir)
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(font_path, "wb") as f: f.write(r.content)
        except: pass
    try:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        return prop
    except: return None

jp_font = get_jp_font()

# ==========================================
# 3. AI解析エンジン (★ここが最終奥義★)
# ==========================================
# ライブラリを使わず、直接URLを叩く関数
def call_gemini_direct(image_bytes, prompt, api_key):
    # 最新モデルのエンドポイントURL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    # 画像をBase64文字列に変換
    base64_data = base64.b64encode(image_bytes).decode('utf-8')
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {
                    "mime_type": "image/jpeg", # PNGでもjpeg扱いで通ることが多いが念のため汎用的に
                    "data": base64_data
                }}
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            return None, f"API Error {response.status_code}: {response.text}"
        
        result_json = response.json()
        # レスポンスからテキスト抽出
        text_content = result_json['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text_content), None
    except Exception as e:
        return None, f"通信エラー: {e}"

@st.cache_data(show_spinner=False)
def analyze_image_safe(image_bytes):
    prompt = """
    あなたは陸上長距離の専門分析官です。画像の「持久走記録用紙」からデータを抽出しJSONを出力してください。
    
    【重要ルール】
    1. 用紙に「15分間走」とあれば `record_type_minutes` は 15 (男子/3000m)。
    2. 用紙に「12分間走」とあれば `record_type_minutes` は 12 (女子/2100m)。
    3. 全ての記録回を `records` に抽出。
    4. `coach_advice` に、AT閾値（落ち込み）の指摘や生理学的アドバイスを150文字程度で記述。

    【JSON構造】
    {
      "name": "選手名",
      "record_type_minutes": 15,
      "records": [
        { "attempt": 1, "distance": 3200, "laps": [60, 62, 65] }
      ],
      "coach_advice": "アドバイス文"
    }
    """
    return call_gemini_direct(image_bytes, prompt, api_key)

# ==========================================
# 4. レポート描画 (変更なし)
# ==========================================
def create_report(data):
    name = data.get("name", "選手")
    base_min = int(data.get("record_type_minutes", 15))
    records = data.get("records", [])
    advice = data.get("coach_advice", "No Advice")
    
    # --- 計算 ---
    best_dist = 0
    laps = []
    if records:
        # 距離を数値化して最大を探す
        for r in records:
            d_raw = r.get("distance", 0)
            try: d_val = float(str(d_raw).replace("m",""))
            except: d_val = 0
            if d_val > best_dist:
                best_dist = d_val
                laps = r.get("laps", [])
    
    target_dist = 3000 if base_min == 15 else 2100
    
    # ペース計算
    run_sec = base_min * 60
    pace_str = "-'--/km"
    vo2max = 0
    target_str = "--:--"
    target_sec = 0

    if best_dist > 0:
        sec_per_km = run_sec / (best_dist/1000)
        pace_str = f"{int(sec_per_km//60)}'{int(sec_per_km%60):02d}/km"
        
        # VO2Max (簡易)
        dist_12 = best_dist * (12/base_min)
        vo2max = (dist_12 - 504.9) / 44.73
        
        # Target (Riegel)
        pred = run_sec * (target_dist/best_dist)**1.06
        target_sec = pred * 0.99
        target_str = f"{int(target_sec//60)}分{int(target_sec%60):02d}秒"

    # --- 描画 ---
    fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
    
    # Header
    fig.text(0.05, 0.92, "ATHLETE PERFORMANCE REPORT", fontsize=14, color='gray', fontproperties=jp_font)
    fig.text(0.05, 0.86, f"{name} 選手 ({base_min}分間走)", fontsize=24, weight='bold', fontproperties=jp_font)
    
    # Area 1: Diagnosis
    ax1 = fig.add_axes([0.05, 0.60, 0.35, 0.20])
    ax1.axis('off')
    ax1.add_patch(plt.Rectangle((0,0),1,1, color='#f5f5f5', transform=ax1.transAxes))
    info = (f"自己ベスト: {best_dist}m\n"
            f"平均ペース: {pace_str}\n"
            f"VO2Max: {vo2max:.1f}\n"
            f"----------------\n"
            f"目標({target_dist}m): {target_str}")
    ax1.text(0.05, 0.85, "■ Scientific Diagnosis", weight='bold', fontproperties=jp_font)
    ax1.text(0.05, 0.15, info, fontsize=12, linespacing=1.8, fontproperties=jp_font)
    
    # Area 2: Laps
    ax2 = fig.add_axes([0.45, 0.40, 0.50, 0.40])
    ax2.axis('off')
    ax2.set_title("■ Lap Analysis", loc='left', fontproperties=jp_font)
    
    if records:
        max_l = max([len(r.get('laps',[])) for r in records]) if records else 0
        cols = ["No"]
        for i in range(len(records)): cols.extend([f"#{i+1} Lap", f"Split"])
        rows = []
        for i in range(max_l):
            row = [str(i+1)]
            for r in records:
                ll = r.get('laps',[])
                if i < len(ll):
                    row.extend([f"{ll[i]:.1f}", f"{int(sum(ll[:i+1])//60)}:{int(sum(ll[:i+1])%60):02d}"])
                else: row.extend(["-","-"])
            rows.append(row)
        
        t = ax2.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.2)
        
        # AT Check
        for (r,c), cell in t.get_celld().items():
            cell.set_text_props(fontproperties=jp_font)
            if r>0 and c>0 and c%2!=0: # Lap col
                try:
                    val = float(rows[r-1][c])
                    if r > 1:
                        prev = float(rows[r-2][c])
                        if val - prev >= 2.0:
                            cell.set_facecolor('#ffcdd2')
                except: pass

    # Area 3: Pace
    ax3 = fig.add_axes([0.05, 0.10, 0.35, 0.45])
    ax3.axis('off')
    if target_sec > 0:
        ax3.text(0, 1.0, f"■ Target Pace ({target_dist}m)", weight='bold', fontproperties=jp_font)
        p_rows = []
        check_d = [1000, 2000, 3000] if target_dist==3000 else [1000, 2000]
        for d in check_d:
            tm = target_sec * (d/target_dist)
            p_rows.append([f"{d}m", f"{int(tm//60)}:{int(tm%60):02d}"])
        t3 = ax3.table(cellText=p_rows, colLabels=["Dist", "Time"], loc='top')
        t3.scale(1, 1.5)
        for key, cell in t3.get_celld().items(): cell.set_text_props(fontproperties=jp_font)

    # Area 4: Advice
    ax4 = fig.add_axes([0.45, 0.05, 0.50, 0.30])
    ax4.axis('off')
    ax4.add_patch(plt.Rectangle((0,0),1,1, color='#fff9c4', transform=ax4.transAxes))
    ax4.text(0.02, 0.90, "■ Coach Advice", color='#e65100', weight='bold', fontproperties=jp_font)
    wrap_adv = "\n".join(textwrap.wrap(advice, 30))
    ax4.text(0.02, 0.80, wrap_adv, va='top', fontsize=10, fontproperties=jp_font)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

# ==========================================
# 5. UI
# ==========================================
st.title("🏃 Data Science Athlete Report")
up_file = st.file_uploader("画像アップロード", type=['jpg','png'])

if up_file and st.button("解析開始"):
    with st.spinner("AI解析中 (Direct API Mode)..."):
        data, err = analyze_image_safe(up_file.getvalue())
        if err: st.error(err)
        else:
            st.success("完了")
            st.image(create_report(data))
