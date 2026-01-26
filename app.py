import streamlit as st
import pandas as pd
import json
import base64
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance
from openai import OpenAI

# ==========================================
# 1. システム設定
# ==========================================
st.set_page_config(page_title="持久走データサイエンス", layout="wide")

st.markdown("""
<style>
.metric-box { background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #2980b9; }
.advice-box { background-color:#fff9c4; padding:15px; border-radius:10px; border: 1px solid #f1c40f; }
.small-note { color: #666; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. APIキー設定（OpenAI）
# ==========================================
API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if not API_KEY:
    st.error("Secretsに OPENAI_API_KEY が設定されていません。")
    st.stop()

client = OpenAI(api_key=API_KEY)

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
        "coach_advice": "今回は記録を正確に読み取れませんでした。用紙全体が明るく写るように撮影して再挑戦しましょう。"
    }

# ==========================================
# 4. 画像を低コスト化して base64（JPEG）へ
# ==========================================
def optimize_image_for_cost(image: Image.Image, max_width: int = 768) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")

    w, h = image.size
    if w > max_width:
        new_h = int(h * (max_width / w))
        image = image.resize((max_width, new_h))

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.15)
    return image

def image_to_jpeg_base64(image: Image.Image, jpeg_quality: int = 65) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def base64_to_data_url_jpeg(image_b64: str) -> str:
    # ★ここが今回の修正ポイント：image_url に data URL を渡す
    return f"data:image/jpeg;base64,{image_b64}"

# ==========================================
# 5. 解析ロジック（低コスト・安定）
# ==========================================
def run_analysis(image: Image.Image):
    prompt = """
あなたは陸上長距離のデータ分析官です。
以下の指示は【絶対に】守ってください。

【最重要ルール】
- 出力はJSONのみ
- 説明文・前置き・後書きは禁止
- ``` や ```json は使用禁止
- JSONの外に1文字でも出力したら失敗です

【JSON形式】
{
  "name": "選手名",
  "record_type_minutes": 15,
  "race_category": "time",
  "records": [
    {
      "attempt": 1,
      "total_dist": 4050,
      "total_time_str": "14:45",
      "laps": [91, 87, 89]
    }
  ],
  "coach_advice": "短い励まし（2〜3文）"
}

【読み取りルール】
- 「①②③」など複数回の記録があれば records に複数入れる
- laps は各周のラップ秒（できるだけ抽出）
- total_dist は合計(m)
- total_time_str は最終の合計タイム（書かれていれば）
- 不明な項目は推測せず 0 / 空配列
- coach_advice は短く具体的に（2〜3文）
"""

    optimized = optimize_image_for_cost(image, max_width=768)
    image_b64 = image_to_jpeg_base64(optimized, jpeg_quality=65)
    image_data_url = base64_to_data_url_jpeg(image_b64)

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    # ★ image_base64 ではなく image_url（data URL）を使う
                    {"type": "input_image", "image_url": image_data_url},
                ]
            }],
            temperature=0.2,
        )

        text = response.output_text.strip()
        data = safe_json_load(text)
        if data is None:
            return empty_result(), "JSON解析に失敗しました"

        return data, None

    except Exception as e:
        return empty_result(), f"解析エラー: {str(e)}"

# ==========================================
# 6. メイン画面
# ==========================================
st.markdown("## 🏃 持久走データサイエンス（低コスト版）")
st.markdown('<div class="small-note">画像は自動で軽量化して送信します（0.1円以下狙い）</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "記録用紙を撮影してアップロードしてください",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    raw_img = Image.open(uploaded_file)
    raw_img = ImageOps.exif_transpose(raw_img).convert("RGB")
    st.image(raw_img, caption="アップロード画像（元）", width=320)

    optimized_preview = optimize_image_for_cost(raw_img, max_width=768)
    st.image(optimized_preview, caption="送信する画像（軽量化後）", width=320)

    with st.spinner("AI解析中..."):
        data, err = run_analysis(raw_img)

    if err:
        st.warning(err)
    else:
        st.success("解析完了")

    name = data.get("name", "選手")
    records = data.get("records", [])
    advice = data.get("coach_advice", "")

    st.markdown(f"# 🏃‍♂️ {name} 選手｜能力分析レポート")

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

    st.markdown("### 👟 AIコーチのアドバイス")
    st.markdown(f"""
    <div class="advice-box">
    {advice}
    </div>
    """, unsafe_allow_html=True)
