import streamlit as st
import pandas as pd
import google.generativeai as genai
from PIL import Image, ImageOps
import json

# ==========================================
# 1. システム設定
# ==========================================
st.set_page_config(page_title="持久走データサイエンス", layout="wide")

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
# 2. 解析実行ロジック（1.5-flash 固定）
# ==========================================
def run_analysis(image):
    # 最新版ライブラリが入ったので、堂々と1.5-flashを指定します
    target_model = "models/gemini-1.5-flash"
    model = genai.GenerativeModel(target_model)
    
    prompt = """
    あなたは陸上長距離のデータ分析官です。画像の「持久走記録用紙」を解析し、以下のJSON形式で出力してください。
    
    【ルール】
    1. ラップタイムは全て抽出すること。
    2. アドバイスは、選手を励ます具体的で前向きな内容を記述すること。
    
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
      "coach_advice": "アドバイス"
    }
    """
    
    try:
        response = model.generate_content(
            [prompt, image], 
            generation_config={"response_mime_type": "application/json"}
        )
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        
        if isinstance(data, list):
            data = {"records": data, "name": "選手", "record_type_minutes": 15, "coach_advice": ""}
            
        return data, None
    except Exception as e:
        return None, f"エラー: {str(e)}"

# ==========================================
# 3. メイン画面
# ==========================================
uploaded_file = st.file_uploader("記録用紙を撮影してアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert('RGB')
    st.image(image, caption='アップロード画像', width=300)
    
    with st.spinner("AI解析中..."):
        data, err = run_analysis(image)
        
        if err:
            st.error(f"解析エラー: {err}")
        elif data:
            st.success("解析完了")
            
            name = data.get("name", "選手")
            records = data.get("records", [])
            raw_advice = data.get("coach_advice")
            advice = str(raw_advice) if raw_advice else "データから十分なアドバイスを生成できませんでした。"
            
            st.markdown(f"# 🏃‍♂️ {name} 選手｜能力分析レポート")
            
            # エリア1: ラップ表
            st.markdown("### 📊 ラップ・スプリット表")
            if records:
                rec = records[0]
                laps = rec.get("laps", [])
                
                rows = []
                for i, lap in enumerate(laps):
                    total_sec = sum(laps[:i+1])
                    m, s = divmod(total_sec, 60)
                    rows.append({
                        "周回": f"{i+1}周",
                        "ラップ": f"{lap:.1f}",
                        "スプリット": f"{int(m)}:{int(s):02d}"
                    })
                st.table(pd.DataFrame(rows))
                
                total_dist = rec.get("total_dist", 0)
                st.metric("総距離", f"{total_dist} m")

            # エリア2: アドバイス
            st.markdown("### 👟 AIコーチのアドバイス")
            st.markdown(f"""
            <div class="advice-box">
            {advice}
            </div>
            """, unsafe_allow_html=True)
