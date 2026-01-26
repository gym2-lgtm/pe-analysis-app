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

# ==========================================
# 2. APIキー設定
# ==========================================
raw_key = st.secrets.get("GEMINI_API_KEY", "")
API_KEY = str(raw_key).replace("\n", "").replace(" ", "").replace("　", "").replace('"', "").replace("'", "").strip()

if not API_KEY:
    st.error("SecretsにAPIキーが設定されていません。")
    st.stop()

genai.configure(api_key=API_KEY)

# ==========================================
# 3. JSON安全処理
# ==========================================
def safe_json_load(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
    return None

def empty_result():
    return {
        "name": "選手",
        "record_type_minutes": 15,
        "race_category": "time",
        "records": [],
        "coach_advice": "今回は記録を正確に読み取ることができませんでしたが、挑戦したこと自体が素晴らしいです。次回は用紙全体がはっきり写るように撮影してみましょう。"
    }

# ==========================================
# 4. 解析ロジック（v1beta確定対応）
# ==========================================
def run_analysis(image):
    # v1betaで画像対応・確実に存在するモデル
    model = genai.GenerativeModel("gemini-1.0-pro-vision")

    prompt = """
あなたは陸上長距離のデータ分析官です。
以下の指示は【絶対に】守ってください。

【最重要ルール】
- 出力はJSONのみ
- 説明文・前置き・後書きは禁止
- ``` や ```json は使用禁止
- JSONの外に1文字でも出力したら失敗です

【JSONスキーマ】
{
  "name": "string",
  "record_type_minutes": number,
  "race_category": "time",
  "records": [
    {
      "attempt": number,
      "total_dist": number,
      "total_time_str": "mm:ss",
      "laps": [number]
    }
  ],
  "coach_advice": "string"
}

【内容ルール】
- ラップタイムは全て抽出
- 数値は半角
- laps は秒単位
- 読み取れない項目は推測せず 0 または空配列
- coach_advice は前向きで励ます内容

【失敗時】
- 解析不能でも必ず上記形式のJSONを出力
"""

    try:
        response = model.generate_content(
            [prompt, image],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        )

        data = safe_json_load(response.text)

        if data is None:
            return empty_result(), "JSON解析に失敗しました"

        return data, None

    except Exception as e:
        return empty_result(), f"解析エラー: {str(e)}"

# ==========================================
# 5. メイン画面
# ==========================================
uploaded_file = st.file_uploader(
    "記録用紙を撮影してアップロードしてください",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    st.image(image, caption="アップロード画像", width=300)

    with st.spinner("AI解析中..."):
        data, err = run_analysis(image)

    if err:
        st.warning(err)
    else:
        st.success("解析完了")

    name = data.get("name", "選手")
    records = data.get("records", [])
    advice = data.get("coach_advice", "")

    st.markdown(f"# 🏃‍♂️ {name} 選手｜能力分析レポート")

    # ラップ表
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
                "ラップ（秒）": f"{lap:.1f}",
                "累計": f"{int(m)}:{int(s):02d}"
            })

        st.table(pd.DataFrame(rows))
        st.metric("総距離", f"{rec.get('total_dist', 0)} m")

    # アドバイス
    st.markdown("### 👟 AIコーチのアドバイス")
    st.markdown(f"""
    <div class="advice-box">
    {advice}
    </div>
    """, unsafe_allow_html=True)
