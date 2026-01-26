import streamlit as st
import pandas as pd
import json
import base64
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance
from openai import OpenAI

# ==========================================
# 学校仕様
# ==========================================
LAP_M = 300  # 1周=300m

# ==========================================
# UI
# ==========================================
st.set_page_config(page_title="持久走データサイエンス", layout="wide")

st.markdown("""
<style>
.small-note { color: #666; font-size: 0.9rem; }
.report-box { background-color:#f7f7f7; padding:16px; border-radius:12px; border: 1px solid #e6e6e6; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# OpenAI
# ==========================================
API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if not API_KEY:
    st.error("Secretsに OPENAI_API_KEY が設定されていません。")
    st.stop()

client = OpenAI(api_key=API_KEY)

# ==========================================
# utils
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
        "sheet_hints": "",
        "records": []
    }

def mmss_to_sec(s: str) -> float:
    if s is None:
        return 0.0
    s = str(s).strip().replace(" ", "")
    if ":" not in s:
        try:
            return float(s)
        except:
            return 0.0
    parts = s.split(":")
    if len(parts) != 2:
        return 0.0
    try:
        m = int(parts[0])
        sec = int(parts[1])
        return float(m * 60 + sec)
    except:
        return 0.0

def sec_to_mmss(sec: float) -> str:
    if sec <= 0:
        return "0:00"
    m = int(sec // 60)
    s = int(round(sec - m * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d}"

def splits_to_laps(splits_sec):
    laps = []
    prev = 0.0
    for s in splits_sec:
        s = float(s)
        laps.append(max(0.0, s - prev))
        prev = s
    return laps

def detect_at_alerts(laps_sec, threshold=3.0):
    alerts = []
    for i in range(1, len(laps_sec)):
        prev = float(laps_sec[i-1])
        cur = float(laps_sec[i])
        diff = cur - prev
        if diff >= threshold:
            alerts.append((i+1, prev, cur, diff))
    return alerts

def pace_per_km(dist_m, time_sec):
    if dist_m <= 0 or time_sec <= 0:
        return 0.0
    return time_sec / (dist_m / 1000)

def predict_time_by_same_speed(dist_m, time_sec, target_m):
    if dist_m <= 0 or time_sec <= 0:
        return 0.0
    v = dist_m / time_sec
    return target_m / v

def estimate_vo2max(target_m, t_sec):
    if target_m <= 0 or t_sec <= 0:
        return 0.0
    v_m_per_min = target_m / (t_sec / 60.0)
    return round(0.2 * v_m_per_min + 3.5, 1)

def build_pace_guide(target_m, target_time_sec):
    if target_m <= 0 or target_time_sec <= 0:
        return []
    plans = [("維持", 1.03), ("目標", 1.00), ("突破", 0.97)]
    full_laps = target_m // LAP_M
    rem = target_m % LAP_M

    out = []
    for label, mult in plans:
        t = target_time_sec * mult
        per_m = t / target_m
        lap_sec = per_m * LAP_M
        rem_sec = per_m * rem if rem else 0
        detail = f"{LAP_M}m:{sec_to_mmss(lap_sec)} × {full_laps}"
        if rem:
            detail += f" + {rem}m:{sec_to_mmss(rem_sec)}"
        out.append({"プラン": label, "想定タイム": sec_to_mmss(t), "目標ラップ": detail})
    return out

# ==========================================
# ★方法A：欄外メモを物理的に消す（トリミング）
# ==========================================
def crop_margin_for_ignore_notes(image: Image.Image) -> Image.Image:
    """
    欄外メモを入りにくくするために、周囲の余白を少し削る。
    """
    w, h = image.size

    # 余白カット（必要なら微調整OK）
    left = int(w * 0.06)
    right = int(w * 0.96)
    top = int(h * 0.03)
    bottom = int(h * 0.98)

    return image.crop((left, top, right, bottom))

def optimize_image_for_cost(image, max_width=768):
    image = ImageOps.exif_transpose(image).convert("RGB")

    # ★欄外をカット（最強）
    image = crop_margin_for_ignore_notes(image)

    w, h = image.size
    if w > max_width:
        new_h = int(h * (max_width / w))
        image = image.resize((max_width, new_h))

    image = ImageEnhance.Contrast(image).enhance(1.15)
    return image

def image_to_data_url(image, jpeg_quality=65):
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# ==========================================
# 推理（男子/女子）
# ==========================================
def infer_profile(rec, sheet_hints: str):
    hints = (sheet_hints or "").replace("　", " ").lower()

    # distance_race_m が取れていれば最強
    dist_race_m = int(rec.get("distance_race_m", 0) or 0)
    if dist_race_m == 3000:
        return {"gender": "male", "time_min": 15, "target_m": 3000, "reason": "distance_race_m=3000"}
    if dist_race_m == 2100:
        return {"gender": "female", "time_min": 12, "target_m": 2100, "reason": "distance_race_m=2100"}

    # キーワード最優先
    if any(k in hints for k in ["男子", "15分", "3000", "3000m", "15"]):
        return {"gender": "male", "time_min": 15, "target_m": 3000, "reason": f"keyword:{sheet_hints}"}
    if any(k in hints for k in ["女子", "12分", "2100", "2100m", "12"]):
        return {"gender": "female", "time_min": 12, "target_m": 2100, "reason": f"keyword:{sheet_hints}"}

    # 通過タイムの最終＝合計時間で判定
    splits_mmss = rec.get("splits_mmss", []) or []
    splits_sec = [mmss_to_sec(x) for x in splits_mmss if str(x).strip()]
    total_time = max(splits_sec) if splits_sec else 0.0
    if total_time > 12.5 * 60:
        return {"gender": "male", "time_min": 15, "target_m": 3000, "reason": "time>12:30"}

    # 時間走距離で推理
    time_dist = float(rec.get("time_run_dist_m", 0) or 0)
    if time_dist >= 3200:
        return {"gender": "male", "time_min": 15, "target_m": 3000, "reason": "time_run_dist>=3200"}
    if time_dist > 0 and time_dist < 2600:
        return {"gender": "female", "time_min": 12, "target_m": 2100, "reason": "time_run_dist<2600"}

    return {"gender": "male", "time_min": 15, "target_m": 3000, "reason": "fallback"}

# ==========================================
# 抽出（画像→JSON）
# ==========================================
def run_extract(image):
    prompt = f"""
あなたは帳票読取の専門家です。記録用紙から必要情報を抽出し、必ずJSONのみを出力してください。
説明文や```は禁止です。

【重要】
- 手書きのメモ（欄外・余白・枠の外）は一切無視すること。
- 読み取るのは表の枠内（記入欄）のみ。

【最優先で抽出するもの】
1) 300mごとの通過タイム（スプリットタイム）
- 形式は "m:ss" 文字列で配列（例 "2:02"）
- 見える範囲で全て抽出

2) 時間走（15分/12分）の最下段「走行距離（m）」→ time_run_dist_m

3) 距離走（3000m/2100m）の最下段「記録（分:秒）」→ distance_race_time_mmss
- あわせて distance_race_m を 3000 or 2100 にする

【JSON形式】
{{
  "name": "選手名",
  "sheet_hints": "用紙内で読み取れたキーワード（男子/女子/15分/12分/3000/2100 等）を短く列挙。無ければ空文字",
  "records": [
    {{
      "attempt": 1,
      "lap_m": {LAP_M},
      "splits_mmss": ["0:58","2:02","3:08"],
      "time_run_dist_m": 4100,
      "distance_race_m": 3000,
      "distance_race_time_mmss": "11:12"
    }}
  ]
}}

【ルール】
- 不明は推測せず 0/空配列/空文字
"""

    optimized = optimize_image_for_cost(image, max_width=768)
    url = image_to_data_url(optimized, jpeg_quality=65)

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": url},
            ]
        }],
        temperature=0.2,
    )

    data = safe_json_load(resp.output_text.strip())
    if not data:
        return empty_result(), "JSON解析に失敗しました（抽出）"
    if "sheet_hints" not in data:
        data["sheet_hints"] = ""
    return data, None

# ==========================================
# 文章レポート（画像なし）
# ==========================================
def build_report_prompt(name, profile, rec):
    time_min = profile["time_min"]
    time_sec = time_min * 60
    target_m = profile["target_m"]

    time_run_dist_m = float(rec.get("time_run_dist_m", 0) or 0)

    splits_mmss = rec.get("splits_mmss", []) or []
    splits_sec = [mmss_to_sec(x) for x in splits_mmss if str(x).strip()]
    splits_sec = [s for s in splits_sec if s > 0]
    splits_sec = sorted(splits_sec)

    laps_sec = splits_to_laps(splits_sec) if len(splits_sec) >= 2 else []
    alerts = detect_at_alerts(laps_sec, threshold=3.0)

    # 科学計算
    pace_sec_km = pace_per_km(time_run_dist_m, time_sec) if time_run_dist_m > 0 else 0.0
    target_time_pred_sec = predict_time_by_same_speed(time_run_dist_m, time_sec, target_m) if time_run_dist_m > 0 else 0.0
    vo2 = estimate_vo2max(target_m, target_time_pred_sec) if target_time_pred_sec > 0 else 0.0

    pace_guide = build_pace_guide(target_m, target_time_pred_sec)
    pace_guide_text = "\n".join([f"- {r['プラン']}: {r['想定タイム']} / {r['目標ラップ']}" for r in pace_guide]) if pace_guide else "- 作成できませんでした"

    alert_lines = "\n".join(
        [f"- {idx}本目（{idx*LAP_M}m）: {prev:.1f}→{cur:.1f}（+{diff:.1f}秒）" for idx, prev, cur, diff in alerts]
    ) if alerts else "- 目立った失速アラートなし"

    dist_race_m = int(rec.get("distance_race_m", 0) or 0)
    dist_race_time_mmss = str(rec.get("distance_race_time_mmss", "") or "").strip()

    dist_race_line = ""
    if dist_race_m in (3000, 2100) and dist_race_time_mmss:
        dist_race_line = f"- 距離走の記録：{dist_race_m}m **{dist_race_time_mmss}**（用紙記載）"
    else:
        dist_race_line = "- 距離走の記録：用紙から読み取れませんでした"

    gender_jp = "男子" if profile["gender"] == "male" else "女子"

    return f"""
あなたは陸上長距離のトップコーチ兼データ分析官です。
以下の数値だけを根拠に、指定の①〜④構成で「文章レポート」を作成してください。

【絶対条件】
- 日本語（中学生に伝わる）
- 必ず数字を根拠として入れる
- 推定は「推定」と明記（VO2Max、換算参考記録）
- 見出し①〜④をそのまま使う
- ②はAT閾値アラートを必ず言及（何本目/何m地点）
- ③は維持/目標/突破の3段階（300mラップ）
- ④は熱く前向きに140文字程度
- 距離走の記録が用紙にあれば必ず拾って言及する

【選手・種別】
選手名: {name}
推定: {gender_jp}（{time_min}分間走 / {target_m}m）

【用紙から抽出できた値（枠内のみ）】
- 時間走の距離：{int(time_run_dist_m) if time_run_dist_m else 0}m（最下段）
{dist_race_line}

【通過タイム（秒）】
{splits_sec}

【ラップ（秒）=通過差分】
{laps_sec}

【平均ペース（時間走）】
{sec_to_mmss(pace_sec_km)} /km

【換算参考記録（推定）】
{target_m}m = {sec_to_mmss(target_time_pred_sec)}

【推定VO2Max（推定）】
{vo2} ml/kg/min

【AT閾値アラート（前の本より+3秒以上）】
{alert_lines}

【③ 目標ラップ表（Pace Guide）】
{pace_guide_text}

【出力フォーマット（この順番で必ず）】
① 科学的ポテンシャル診断 (RESULT / Best)
② ラップ推移 & AT閾値判定
③ 目標ラップ表 (Pace Guide)
④ COACH'S EYE (専門的アドバイス)
"""

def generate_text_report(name, profile, rec):
    prompt = build_report_prompt(name, profile, rec)
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.4,
    )
    return resp.output_text.strip()

# ==========================================
# Main
# ==========================================
st.markdown("## 🏃 持久走データサイエンス（欄外メモ無視・方法A）")
st.markdown('<div class="small-note">欄外メモはトリミングで物理的に除外します</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("記録用紙を撮影してアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file:
    raw_img = Image.open(uploaded_file)
    raw_img = ImageOps.exif_transpose(raw_img).convert("RGB")

    st.image(raw_img, caption="アップロード画像（元）", width=320)

    preview = optimize_image_for_cost(raw_img, max_width=900)
    st.image(preview, caption="送信する画像（欄外カット＋軽量化）", width=320)

    with st.spinner("AI解析中（抽出）..."):
        data, err = run_extract(raw_img)

    if err:
        st.error(err)
        st.stop()

    st.success("抽出完了")

    name = data.get("name", "選手")
    sheet_hints = data.get("sheet_hints", "")
    records = data.get("records", []) or []
    if not records:
        st.error("recordsが空でした。撮影（明るさ・傾き・用紙全体）を改善して再試行してください。")
        st.stop()

    rec = records[0]
    profile = infer_profile(rec, sheet_hints)
    gender_jp = "男子" if profile["gender"] == "male" else "女子"

    st.markdown(f"# 🏃‍♂️ {name} 選手｜能力分析レポート（推定：{gender_jp}）")
    st.caption(f"判定理由: {profile['reason']}")

    # 通過タイム表示
    splits_mmss = rec.get("splits_mmss", []) or []
    splits_sec = [mmss_to_sec(x) for x in splits_mmss if str(x).strip()]
    splits_sec = [s for s in splits_sec if s > 0]
    splits_sec = sorted(splits_sec)

    st.markdown("### 📊 通過タイム（300mごと）")
    rows = []
    for i, s in enumerate(splits_sec):
        rows.append({"本数": f"{i+1}本目", "地点": f"{(i+1)*LAP_M}m", "通過": sec_to_mmss(s)})
    if rows:
        st.table(pd.DataFrame(rows))
    else:
        st.warning("通過タイムが抽出できませんでした。画像の写りを改善して再試行してください。")

    # 最下段値
    time_run_dist_m = int(float(rec.get("time_run_dist_m", 0) or 0))
    dist_race_m = int(rec.get("distance_race_m", 0) or 0)
    dist_race_time = str(rec.get("distance_race_time_mmss", "") or "").strip()

    c1, c2 = st.columns(2)
    c1.metric("時間走の距離（最下段）", f"{time_run_dist_m} m" if time_run_dist_m else "未取得")
    if dist_race_m in (3000, 2100) and dist_race_time:
        c2.metric("距離走の記録（最下段）", f"{dist_race_m}m {dist_race_time}")
    else:
        c2.metric("距離走の記録（最下段）", "未取得")

    # ATアラート
    laps_sec = splits_to_laps(splits_sec) if len(splits_sec) >= 2 else []
    alerts = detect_at_alerts(laps_sec, threshold=3.0)
    if alerts:
        st.warning("⚠️ AT閾値アラート（前の本より+3秒以上）: " +
                   " / ".join([f"{idx}本目(+{diff:.1f}s)" for idx, _, _, diff in alerts]))
    else:
        st.info("AT閾値アラート：目立った失速なし")

    st.markdown("### 📝 文章レポート（画像なし生成）")
    if st.button("📄 詳細レポートを生成（画像なし）"):
        with st.spinner("文章レポート生成中..."):
            try:
                report = generate_text_report(name, profile, rec)
                st.markdown(f'<div class="report-box">{report.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"レポート生成エラー: {e}")
