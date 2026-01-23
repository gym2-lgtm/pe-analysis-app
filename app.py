import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, requests, json, re, os, base64, time
import matplotlib.font_manager as fm
import urllib.request
from PIL import Image, ImageOps

# --- 1. APIキーの取得 (Secretsから読み込む) ---
API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 2. 日本語フォント設定 ---
@st.cache_resource
def load_fp():
    fpath = "NotoSansJP-Regular.ttf"
    if not os.path.exists(fpath):
        url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
        urllib.request.urlretrieve(url, fpath)
    fm.fontManager.addfont(fpath)
    plt.rcParams['font.family'] = 'Noto Sans JP'
    return fm.FontProperties(fname=fpath)

# --- 3. AIエンジン ---
def run_ai(img_bytes):
    if not API_KEY: return None, "SecretsにGEMINI_API_KEYが設定されていません。"
    b64 = base64.b64encode(img_bytes).decode()
    
    try:
        m_list = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}").json()
        models = [m['name'].split('/')[-1] for m in m_list.get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
        target = next((m for m in models if "1.5-flash" in m), models[0])
    except: return None, "APIキーが無効、または通信エラーです。"

    prompt = """
    Extract running data to JSON. Format:
    {
      "name": "Student Name",
      "long_run_dist": 4050, 
      "tt_laps": [65.2, 68.0, 70.5]
    }
    Note: long_run_dist is from the 15/12min section. tt_laps are split seconds from the 3000/2100m section.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
        "generationConfig": {"response_mime_type": "application/json"},
        "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    }
    try:
        res = requests.post(url, json=payload, timeout=30).json()
        raw_text = res['candidates'][0]['content']['parts'][0]['text']
        return json.loads(raw_text), None
    except Exception as e: return None, f"解析失敗: AIがデータを読み取れませんでした。"

# --- 4. レポート描画 ---
def draw_report(data):
    fp = load_fp()
    laps = np.array([float(l) for l in data.get("tt_laps", [])])
    dist = float(data.get("long_run_dist", 0))
    target_d = 3000 if dist > 3200 else 2100
    long_min = 15 if target_d == 3000 else 12
    
    # ポテンシャル推計
    p_sec = long_min * 60 * (target_d / dist)**1.06 if dist > 0 else None
    
    fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
    fig.text(0.05, 0.94, f"持久走 科学的分析レポート: {data.get('name','選手')} 選手", fontsize=24, weight='bold', fontproperties=fp, color='#1a237e')

    # ① 左上: 生理学的評価
    ax1 = fig.add_axes([0.05, 0.55, 0.4, 0.3]); ax1.set_axis_off()
    ax1.set_title("① 生理学的評価", fontproperties=fp, fontsize=16, loc='left', color='#0d47a1')
    vo2 = max((dist * (12/long_min) - 504.9) / 44.73, 0) if dist > 0 else 0
    txt = f"■推定VO2 Max: {vo2:.1f} ml/kg/min\n"
    if p_sec:
        txt += f"■{target_d}m 理論限界タイム: {int(p_sec//60)}分{int(p_sec%60):02d}秒\n\n"
        txt += f"君の心肺機能（エンジン）は非常に強力です。\n今の能力なら{target_d}mをこのタイムで走る\nポテンシャルが十分にあります。"
    else:
        txt += "※15分間走の記録が読み取れませんでした。"
    ax1.text(0, 0.8, txt, fontproperties=fp, fontsize=12, va='top', linespacing=1.8)

    # ② 右上: 周回精密データ
    ax2 = fig.add_axes([0.5, 0.55, 0.45, 0.3]); ax2.set_axis_off()
    ax2.set_title("② 周回精密データ (ラップ表)", fontproperties=fp, fontsize=16, loc='left', color='#0d47a1')
    if len(laps) > 0:
        rows = [[f"{i+1}周", f"{l:.1f}s", "▲UP" if i>0 and l<laps[i-1]-1.5 else ("▼DN" if i>0 and l>laps[i-1]+2 else "―")] for i, l in enumerate(laps[:12])]
        tab = ax2.table(cellText=rows, colLabels=["周回", "ラップ", "傾向"], loc='center', cellLoc='center')
        tab.scale(1, 1.5)
        for c in tab.get_celld().values(): c.set_text_props(fontproperties=fp)

    # ③ 左下: 目標設定
    ax3 = fig.add_axes([0.05, 0.1, 0.4, 0.35]); ax3.set_axis_off()
    ax3.set_title("③ 次回の目標設定ペース", fontproperties=fp, fontsize=16, loc='left', color='#0d47a1')
    if p_sec:
        base_l = p_sec / (target_d/300)
        rows = [
            ["現状維持", f"{int(p_sec*1.05//60)}:{int(p_sec*1.05%60):02d}", f"{base_l*1.05:.1f}s"],
            ["挑戦(PB)", f"{int(p_sec//60)}:{int(p_sec%60):02d}", f"{base_l:.1f}s"],
            ["限界突破", f"{int(p_sec*0.97//60)}:{int(p_sec*0.97%60):02d}", f"{base_l*0.97:.1f}s"]
        ]
        tab2 = ax3.table(cellText=rows, colLabels=["レベル", "ゴール", "ラップ"], loc='center', cellLoc='center', colColours=['#fff9c4']*3)
        tab2.scale(1, 2.5)
        for c in tab2.get_celld().values(): c.set_text_props(fontproperties=fp)
    else:
        ax3.text(0.1, 0.5, "算出用データ不足", fontproperties=fp)

    # ④ 右下: アドバイス（★ここを安全修正しました）
    ax4 = fig.add_axes([0.5, 0.1, 0.45, 0.35]); ax4.set_axis_off()
    ax4.set_title("④ 戦術アドバイス", fontproperties=fp, fontsize=16, loc='left', color='#0d47a1')
    ax4.add_patch(plt.Rectangle((0,0), 1, 1, fill=False, edgecolor='#333'))
    
    adv = "分析結果：\n"
    if p_sec: # データが揃っている場合のみアドバイス生成
        base_l = p_sec / (target_d/300)
        at_p = next((i+1 for i in range(1, len(laps)) if laps[i] > laps[i-1]+3), None)
        if at_p: adv += f"● 第{at_p}周付近でスタミナの限界（AT値）に達しています。\n"
        adv += "● 序盤の2周を「あえて」1秒抑えて入ることで、\n   後半の失速を防ぐ『ネガティブ・スプリット』が有効です。\n"
        adv += f"● 次回は1周 {base_l:.1f}秒の刻みを意識しましょう。"
    else:
        adv += "15分間走のデータがないため、詳細な戦術分析ができません。\n上段の距離が読み取れているか確認してください。"
        
    ax4.text(0.05, 0.85, adv, fontproperties=fp, fontsize=11, va='top', linespacing=1.8)

    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches='tight'); return buf

# --- 5. メインUI ---
st.set_page_config(layout="wide")
st.title("🏃‍♂️ 持久走データ・サイエンス分析")
st.write("15分間(12分間)走と3000m(2100m)の記録から、君のポテンシャルを可視化します。")

file = st.file_uploader("記録用紙の写真をアップロード", type=['jpg','jpeg','png'])
if file:
    with st.spinner("AIが科学的データを抽出中..."):
        img = ImageOps.exif_transpose(Image.open(file)).convert('RGB')
        buf_img = io.BytesIO(); img.save(buf_img, format='JPEG')
        data, err = run_ai(buf_img.getvalue())
        if data:
            st.success("解析成功！レポートを作成します。")
            st.image(draw_report(data), use_column_width=True)
            st.markdown("※画像を長押し、または右クリックで保存して配布できます。")
        else:
            st.error(err)
