import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from PIL import Image, ImageOps
import google.generativeai as genai

# ==========================================
# 1. システム設定
# ==========================================
st.set_page_config(page_title="持久走データサイエンス", layout="wide")

# スタイリング（Web表示用CSS）
st.markdown("""
    <style>
    .metric-box { background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #2980b9; }
    .advice-box { background-color:#fff9c4; padding:15px; border-radius:10px; border: 1px solid #f1c40f; }
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
# 2. AI解析エンジン（総当たりサバイバルモード）
# ==========================================
def run_ai_analysis_with_fallback(image_obj):
    # 試行するモデルの優先順位リスト
    # 1.5系がダメなら、旧世代の安定版(vision)まで降りていく
    candidate_models = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
        "gemini-pro-vision"  # 最後の砦（旧安定版）
    ]

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

    last_error = ""
    
    # 総当たりループ開始
    for model_name in candidate_models:
        try:
            # モデル設定
            model = genai.GenerativeModel(model_name)
            
            # 生成実行
            response = model.generate_content(
                [prompt, image_obj], 
                generation_config={"response_mime_type": "application/json"}
            )
            
            # ここまで来たら成功とみなし、データを解析する
            raw_text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw_text)
            
            # データ型ガード
            if isinstance(data, list):
                data = {"records": data, "name": "選手", "record_type_minutes": 15, "race_category": "time", "coach_advice": ""}

            # 成功メッセージ（デバッグ用・本番では消しても良いが安心のため表示）
            st.toast(f"✅ 接続成功: {model_name}")
            
            # --- タイムキーパー自動補正 ---
            max_elapsed_sec = 0
            records = data.get("records", [])
            if not isinstance(records, list): 
                records = []
                data["records"] = []

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
                
            return data, None # 成功したのでループを抜けてリターン

        except Exception as e:
            # 失敗したら次へ
            last_error = str(e)
            # 429(回数制限)の場合は少し待つ
            if "429" in str(e):
                time.sleep(1)
            continue

    # 全滅した場合
    return None, f"全てのモデルで解析に失敗しました。最後の工ラー: {last_error}"

# ==========================================
# 3. ダッシュボード表示（Webネイティブ方式）
# ==========================================
def display_dashboard(data):
    name = data.get("name", "選手")
    records = data.get("records", [])
    raw_advice = data.get("coach_advice")
    advice = str(raw_advice) if raw_advice else "データから十分な情報が得られませんでした。"
    
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

    # 指標計算
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

    pot_3k = (11000/vo2_max)*3.2 if vo2_max>0 else 0
    pm, ps = divmod(pot_3k, 60)
    vo2_msg = f"VO2Max {vo2_max:.1f}" if vo2_max>0 else "計測不能"

    # --- ダッシュボード構築 ---
    st.markdown(f"# 🏃‍♂️ {name} 選手｜能力分析レポート")
    
    # エリア1: スコアカード
    st.markdown("### ① 科学的ポテンシャル診断 (Best)")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-box"><h4>自己ベスト距離</h4><h2>{int(best_l_dist)} m</h2><small>({base_min}分間走)</small></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-box"><h4>平均ペース</h4><h2>{avg_pace}</h2><small>(1kmあたり)</small></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-box"><h4>エンジン性能(VO2Max)</h4><h2>{vo2_max:.1f}</h2><small>ml/kg/min</small></div>""", unsafe_allow_html=True)

    st.divider()

    # エリア2: ラップ表
    st.markdown("### ② ラップ推移 & AT閾値判定")
    if records:
        rows = []
        max_len = max([len(r.get("laps",[])) for r in records]) if records else 0
        cols = ["周回"]
        for i, r in enumerate(records):
            cols.append(f"#{i+1} Lap")
            cols.append(f"#{i+1} Split")
            
        for i in range(max_len):
            row_data = [f"{i+1}周"]
            for rec in records:
                laps = rec.get("laps", [])
                if i < len(laps):
                    sm, ss = divmod(sum(laps[:i+1]), 60)
                    lap_val = f"{laps[i]:.1f}"
                    if i > 0 and i < len(laps) and (laps[i] - laps[i-1] >= 3.0):
                        lap_val = f"⚠️ {lap_val}"
                    
                    row_data.append(lap_val)
                    row_data.append(f"{int(sm)}:{int(ss):02d}")
                else:
                    row_data.extend(["-", "-"])
            rows.append(row_data)
            
        df = pd.DataFrame(rows, columns=cols)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("ラップデータがありません")

    st.divider()

    c_left, c_right = st.columns([1, 1])
    
    # エリア3: ペース表
    with c_left:
        st.markdown(f"### ③ {target_dist}m 目標ペース表")
        levels = [("維持", 1.05), ("PB更新", 1.00), ("限界突破", 0.94)]
        p_rows = []
        lap_len = 300
        total_laps = int(target_dist/lap_len)
        
        for i in range(1, total_laps+1):
            r = {"距離": f"{i*lap_len}m"}
            for lbl, fac in levels:
                t = ref_sec * fac * (i/total_laps)
                pm_t, ps_t = divmod(t, 60)
                r[lbl] = f"{int(pm_t)}:{int(ps_t):02d}"
            p_rows.append(r)
        st.dataframe(pd.DataFrame(p_rows), use_container_width=True)

    # エリア4: アドバイス
    with c_right:
        st.markdown("### ④ AIコーチのアドバイス")
        st.markdown(f"""
        <div class="advice-box">
        <b>🤖 戦略アドバイス:</b><br>
        {advice}<br><br>
        <b>🫀 生理学的評価:</b><br>
        {vo2_msg} (3000m換算: {int(pm)}分{int(ps):02d}秒 相当)<br>
        今のタイムとの差は『スピードへの慣れ』だけです。自信を持って攻めましょう。
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 4. メインUI
# ==========================================
uploaded_file = st.file_uploader("記録用紙を撮影してアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert('RGB')
    st.image(image, caption='アップロード画像', width=300)
    
    with st.spinner("AIモデルを総当たりで検索中..."):
        data, err = run_ai_analysis_with_fallback(image)
        if data:
            st.success("解析完了！")
            display_dashboard(data)
        else:
            st.error(f"システムエラー: {err}")
