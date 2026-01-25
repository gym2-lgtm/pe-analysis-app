import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from PIL import Image, ImageOps
import json
import time

# ==========================================
# 1. システム設定 & デザイン調整
# ==========================================
st.set_page_config(page_title="持久走データサイエンス", layout="wide")

# CSSでレポート風の見た目に整える
st.markdown("""
    <style>
    /* 全体のフォントを読みやすく */
    html, body, [class*="css"] {
        font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
    }
    /* エリア1: 指標ボックスのデザイン */
    .metric-container {
        background-color: #f8f9fa;
        border-left: 5px solid #2980b9;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 14px; color: #7f8c8d; font-weight: bold; }
    .metric-value { font-size: 28px; color: #2c3e50; font-weight: bold; }
    .metric-sub { font-size: 12px; color: #95a5a6; }
    
    /* エリア4: アドバイスボックスのデザイン */
    .advice-box {
        background-color: #fff9c4;
        border: 2px solid #f1c40f;
        border-radius: 10px;
        padding: 20px;
        color: #5d4037;
        line-height: 1.6;
    }
    .advice-title { font-weight: bold; color: #d35400; font-size: 18px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# APIキー設定
raw_key = st.secrets.get("GEMINI_API_KEY", "")
API_KEY = str(raw_key).replace("\n", "").replace(" ", "").replace("　", "").replace('"', "").replace("'", "").strip()

if not API_KEY:
    st.error("SecretsにAPIキーが設定されていません。")
    st.stop()

genai.configure(api_key=API_KEY)

# ==========================================
# 2. 賢いモデル選択ロジック (エラー回避)
# ==========================================
def get_best_model():
    """
    404エラーや429エラーを避けるため、使えるモデルを上から順に探して返す。
    """
    candidates = [
        "models/gemini-1.5-flash",        # 本命
        "models/gemini-1.5-flash-latest", # 表記ゆれ
        "models/gemini-1.5-pro",          # 高性能
        "models/gemini-pro-vision",       # 最後の砦
    ]
    try:
        # APIキーで利用可能なモデル一覧を取得
        my_models = [m.name for m in genai.list_models()]
        for cand in candidates:
            if cand in my_models:
                return cand
        # リストになくても画像対応なら使う
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and 'vision' in m.name:
                return m.name
    except:
        pass
    return "models/gemini-1.5-flash" # 最終手段

# ==========================================
# 3. AI解析 & データ処理ロジック
# ==========================================
def run_analysis(image):
    target_model = get_best_model()
    model = genai.GenerativeModel(target_model)
    
    prompt = """
    あなたは陸上長距離の専門分析官です。画像の「持久走記録用紙」からデータを抽出してください。
    
    【ルール】
    1. 用紙の「15分間走」または「12分間走」の記述を読み取ってください。
    2. 全ての周回のラップタイムを正確に抽出してください。
    3. アドバイスは、選手のモチベーションを上げる具体的で前向きな内容を記述してください。
    
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
      "coach_advice": "アドバイス文章"
    }
    """
    
    try:
        response = model.generate_content(
            [prompt, image], 
            generation_config={"response_mime_type": "application/json"}
        )
        data = json.loads(response.text.replace("```json", "").replace("```", "").strip())
        
        # リストで返ってきた場合の補正
        if isinstance(data, list):
            data = {"records": data, "name": "選手", "record_type_minutes": 15, "coach_advice": ""}
            
        # --- タイムキーパー & 距離チェック (自動補正) ---
        max_elapsed_sec = 0
        records = data.get("records", [])
        if not isinstance(records, list): records = []
        
        for rec in records:
            laps = rec.get("laps", [])
            if laps:
                val = sum(laps)
                if val > max_elapsed_sec: max_elapsed_sec = val
            # 文字列タイムも確認
            if "total_time_str" in rec:
                try:
                    parts = str(rec["total_time_str"]).replace("分",":").replace("秒","").split(":")
                    if len(parts)>=2:
                        val = int(parts[0])*60 + int(parts[1])
                        if val > max_elapsed_sec: max_elapsed_sec = val
                except: pass
        
        # 12分30秒超え or 3200m超えなら強制的に15分走扱い
        current_type = data.get("record_type_minutes", 15)
        
        # 距離チェック
        dist_check = 0
        if records:
            try:
                dist_check = float(str(records[0].get("total_dist", 0)).replace("m","").replace(",",""))
            except: pass

        if (max_elapsed_sec > 750 or dist_check > 3200) and current_type == 12:
            st.toast(f"⏱️ 自動補正: 記録内容から『15分間走(男子)』と判定しました。")
            data["record_type_minutes"] = 15
            
        return data, None
    except Exception as e:
        return None, str(e)

# ==========================================
# 4. レポート表示機能 (ここがメイン)
# ==========================================
def display_report(data):
    name = data.get("name", "選手")
    records = data.get("records", [])
    raw_advice = data.get("coach_advice")
    advice = str(raw_advice) if raw_advice else "データから十分な情報が得られませんでした。"
    
    base_min = int(data.get("record_type_minutes", 15))
    target_dist = 3000 if base_min == 15 else 2100 # 男子3000m / 女子2100m設定

    # --- 計算処理 ---
    best_rec = {}
    best_l_dist = 0
    best_total_sec = 0
    
    if records:
        # ベスト記録（最長距離）を探す
        try:
            def get_d(r): return float(str(r.get("total_dist",0)).replace("m","").replace(",",""))
            best_rec = max(records, key=get_d)
            best_l_dist = get_d(best_rec)
            best_total_sec = base_min * 60
        except: pass

    # 各種指標
    pace_sec = best_total_sec / (best_l_dist/1000) if best_l_dist>0 else 0
    avg_pace = f"{int(pace_sec//60)}'{int(pace_sec%60):02d}/km"
    
    # VO2Max (12分間走換算距離から推定)
    d12 = best_l_dist * (12 / base_min) if base_min > 0 else 0
    vo2_max = (d12 - 504.9) / 44.73 if d12 > 504.9 else 0
    
    # ターゲット距離(3000m/2100m)の予想タイム (リーゲルの公式)
    ref_sec = best_total_sec * (target_dist / best_l_dist)**1.06 if best_l_dist > 0 else 0
    rm, rs = divmod(ref_sec, 60)
    ref_str = f"{int(rm)}分{int(rs):02d}秒"

    # --- 画面構築 ---
    st.markdown(f"# 🏃‍♂️ {name} 選手｜能力分析レポート")
    st.caption(f"種目判定: {base_min}分間走 (ターゲット: {target_dist}m)")
    
    # ------------------------------------------------
    # ① 左上：科学的ポテンシャル診断
    # ------------------------------------------------
    st.markdown("### ① 科学的ポテンシャル診断 (Best)")
    col1, col2, col3, col4 = st.columns(4)
    
    def metric_card(label, value, sub):
        return f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """
    
    with col1: st.markdown(metric_card("自己ベスト", f"{int(best_l_dist)}m", f"{base_min}分間走"), unsafe_allow_html=True)
    with col2: st.markdown(metric_card("平均ペース", avg_pace, "/km"), unsafe_allow_html=True)
    with col3: st.markdown(metric_card("VO2Max", f"{vo2_max:.1f}", "ml/kg/min"), unsafe_allow_html=True)
    with col4: st.markdown(metric_card(f"{target_dist}m換算", ref_str, "予想タイム"), unsafe_allow_html=True)

    st.divider()

    # ------------------------------------------------
    # ② 右上：ラップ推移 & AT閾値判定 (赤色強調)
    # ------------------------------------------------
    st.markdown("### ② ラップ推移 & AT閾値判定")
    st.caption("※前回より3.0秒以上落ちたラップは「赤色背景」で警告表示されます（AT閾値超過の可能性）")

    if records:
        # ベスト記録のデータを採用
        laps = best_rec.get("laps", [])
        
        # データフレーム作成
        df_data = []
        for i, lap in enumerate(laps):
            split = sum(laps[:i+1])
            sm, ss = divmod(split, 60)
            
            # AT判定ロジック
            is_drop = False
            diff = 0
            if i > 0:
                diff = lap - laps[i-1]
                if diff >= 3.0: # 3秒落ちルール
                    is_drop = True
            
            df_data.append({
                "周回": f"{i+1}周",
                "ラップ": lap,
                "差": f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}",
                "スプリット": f"{int(sm)}:{int(ss):02d}",
                "AT判定": is_drop # スタイリング用のフラグ
            })
            
        df = pd.DataFrame(df_data)
        
        # ★Pandas Stylerで「条件付き書式」を適用（ここがポイント！）
        def highlight_drops(row):
            if row['AT判定']:
                return ['background-color: #ffcccc; color: #b71c1c; font-weight: bold'] * len(row)
            return [''] * len(row)

        # 表示用カラムのみ選択してスタイル適用
        display_cols = ["周回", "ラップ", "差", "スプリット"]
        styled_df = df.style.apply(highlight_drops, axis=1).format({"ラップ": "{:.1f}"})
        
        # 表示
        col_table, col_graph = st.columns([1, 1.5])
        with col_table:
            st.dataframe(styled_df, use_container_width=True, column_order=display_cols, hide_index=True)
        
        with col_graph:
            # 折れ線グラフも添える
            chart_data = pd.DataFrame({"周回": range(1, len(laps)+1), "ラップタイム": laps})
            st.line_chart(chart_data, x="周回", y="ラップタイム")

    else:
        st.info("ラップデータがありません")

    st.divider()

    col_L, col_R = st.columns([1, 1])
    
    # ------------------------------------------------
    # ③ 左下：目標ペース配分表
    # ------------------------------------------------
    with col_L:
        st.markdown(f"### ③ {target_dist}m 目標ペース表")
        levels = [("維持", 1.05), ("PB更新", 1.00), ("限界突破", 0.94)]
        p_rows = []
        lap_len = 300 # トラック換算
        total_laps = int(target_dist/lap_len)
        
        for i in range(1, total_laps+1):
            r = {"距離": f"{i*lap_len}m"}
            for lbl, fac in levels:
                t = ref_sec * fac * (i/total_laps)
                pm_t, ps_t = divmod(t, 60)
                r[lbl] = f"{int(pm_t)}:{int(ps_t):02d}"
            p_rows.append(r)
        
        st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------
    # ④ 右下：AIコーチのアドバイス
    # ------------------------------------------------
    with col_R:
        st.markdown("### ④ AIコーチのアドバイス")
        
        # VO2Max評価コメント
        if vo2_max >= 60: v_cmt = "県大会上位レベルの心肺機能です。"
        elif vo2_max >= 50: v_cmt = "長距離に適した強い心臓を持っています。"
        else: v_cmt = "基礎体力はついています。ここからの伸びしろが楽しみです。"

        st.markdown(f"""
        <div class="advice-box">
            <div class="advice-title">🤖 COACH'S EYE</div>
            {advice.replace("。", "。<br>")}
            <hr style="border-top: 1px dashed #f1c40f;">
            <div class="advice-title">🫀 生理学的評価</div>
            <b>VO2Max: {vo2_max:.1f}</b><br>
            {v_cmt}<br>
            この数値は、3000mを<b>{int(rm)}分{int(rs):02d}秒</b>前後で走れる潜在能力を示しています。自信を持ってください！
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 5. メインUI
# ==========================================
uploaded_file = st.file_uploader("記録用紙を撮影してアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert('RGB')
    st.image(image, caption='アップロード画像', width=300)
    
    with st.spinner("AI解析中..."):
        data, err = run_analysis(image)
        if data:
            st.success("解析完了！")
            display_report(data)
        else:
            st.error(f"解析エラー: {err}")
