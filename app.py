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
.glossary-box { background-color:#ffffff; padding:16px; border-radius:12px; border: 1px solid #e6e6e6; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 用語解説（固定文）
# ==========================================
GLOSSARY_TEXT = """
## 🔍 用語解説：VO₂Max（最大酸素摂取量）とは？
**VO₂Max（最大酸素摂取量）**は、運動中に体が取り込んで使える**酸素の最大量**のことです。  
簡単に言うと、**心肺のエンジンの大きさ**を表す数値です。

### ✅ どんな能力を表す？
VO₂Maxが高い人は…
- 心臓や肺が強く、体に酸素をたくさん送れる  
- 筋肉が酸素を使ってエネルギーを作りやすい  
- **長い時間、速いペースを維持しやすい**  

つまり、持久走に必要な「**基礎体力（持久力の土台）**」が大きいです。

### ✅ どう活きてくる？
VO₂Maxが高いと…
- **後半でもペースが落ちにくい**
- 3000m / 2100mで「粘れる」
- 練習を積むほど記録が伸びやすい（伸びしろが大きい）

> ※このアプリのVO₂Maxは、時間走の結果から計算した**推定値（目安）**です。実験室で測る本来の測定とは違います。

---

## 🔍 用語解説：AT（無酸素性作業閾値）とは？
**AT（Anaerobic Threshold：無酸素性作業閾値）**は、運動強度が上がっていく中で、  
体が「**酸素だけでは足りなくなり始める境目**」のことです。

イメージで言うと…
- ここまでは「まだ余裕がある」
- ここを超えると「急に苦しくなって、ペースが落ちやすくなる」

という **限界ライン**です。

### ✅ どんな能力を表す？
ATが高い（強い）人は…
- 苦しくなる境目が遅い  
- つまり、**速いペースで長く走れる**

これはVO₂Max（エンジンの大きさ）とは少し違って、  
レースでの「**粘り・実戦力**」に直結します。

### ✅ どう活きてくる？
ATが強いと…
- 中盤で失速しにくい（タレにくい）
- 苦しい区間でもスピードを維持できる
- 3000m / 2100mで自己ベストを狙いやすい

---

## 🧠 このアプリでのATの見方（授業用の簡易判定）
このアプリでは、300mごとのラップの変化から  
「ATのサイン（失速の始まり）」を見つけます。

### ⚠️ 失速アラート（ATサイン）
前の300mより **+3秒以上遅くなった**場合、  
「ここで苦しくなってペースが落ち始めた可能性がある」と判定します。

> ※本当のATは専門的な測定が必要ですが、授業では「失速が始まる地点＝ATのサイン」として理解すると分かりやすいです。

---

## ✅ VO₂MaxとATの関係（まとめ）
- **VO₂Max＝エンジンの大きさ（基礎体力）**
- **AT＝そのエンジンをレースで使い切る力（粘り）**

つまり…
- VO₂Maxが高い → 伸びる土台がある  
- ATが高い → レースで崩れにくい  
"""

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
    return {"name": "選手", "sheet_hints": "", "records": []}

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

def estimate_vo2max_by_speed(v_m_per_min):
    if v_m_per_min <= 0:
        return 0.0
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
# ★方法A：欄外トリミング
# ==========================================
def crop_margin_for_ignore_notes(image):
    w, h = image.size
    left = int(w * 0.06)
    right = int(w * 0.96)
    top = int(h * 0.03)
    bottom = int(h * 0.98)
    return image.crop((left, top, right, bottom))

def optimize_image_for_cost(image, max_width=768):
    image = ImageOps.exif_transpose(image).convert("RGB")
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
# records から「ベスト回」を選ぶ（最重要）
# ==========================================
def pick_best_time_run(records):
    """
    時間走の3回（①②③）がある前提で、time_run_dist_m が最大の回をベストとして返す。
    取れない場合は records[0] を返す。
    """
    if not records:
        return None
    best = None
    best_dist = -1
    for r in records:
        d = float(r.get("time_run_dist_m", 0) or 0)
        if d > best_dist:
            best_dist = d
            best = r
    return best if best else records[0]

# ==========================================
# 推理（男子/女子）
# ==========================================
def infer_profile(rec, sheet_hints: str):
    hints = (sheet_hints or "").replace("　", " ").lower()

    dist_race_m = int(rec.get("distance_race_m", 0) or 0)
    if dist_race_m == 3000:
        return {"gender": "male", "time_min": 15, "target_m": 3000, "reason": "distance_race_m=3000"}
    if dist_race_m == 2100:
        return {"gender": "female", "time_min": 12, "target_m": 2100, "reason": "distance_race_m=2100"}

    if any(k in hints for k in ["男子", "15分", "3000", "3000m", "15"]):
        return {"gender": "male", "time_min": 15, "target_m": 3000, "reason": f"keyword:{sheet_hints}"}
    if any(k in hints for k in ["女子", "12分", "2100", "2100m", "12"]):
        return {"gender": "female", "time_min": 12, "target_m": 2100, "reason": f"keyword:{sheet_hints}"}

    splits_mmss = rec.get("splits_mmss", []) or []
    splits_sec = [mmss_to_sec(x) for x in splits_mmss if str(x).strip()]
    total_time = max(splits_sec) if splits_sec else 0.0
    if total_time > 12.5 * 60:
        return {"gender": "male", "time_min": 15, "target_m": 3000, "reason": "time>12:30"}

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
- 形式は "m:ss" の配列（例 "2:02"）
- 見える範囲で全て抽出
- ①②③の列がある場合は records に複数入れる

2) 時間走（15分/12分）の最下段「走行距離（m）」→ time_run_dist_m
- ①②③がある場合はそれぞれ入れる

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
def build_report_prompt(name, profile, rec, all_records):
    time_min = profile["time_min"]
    time_sec = time_min * 60
    target_m = profile["target_m"]

    # ベスト/平均/ワースト（時間走距離）
    dists = [int(float(r.get("time_run_dist_m", 0) or 0)) for r in all_records]
    dists = [d for d in dists if d > 0]
    best_dist = max(dists) if dists else int(float(rec.get("time_run_dist_m", 0) or 0))
    worst_dist = min(dists) if dists else int(float(rec.get("time_run_dist_m", 0) or 0))
    avg_dist = int(round(sum(dists) / len(dists))) if dists else int(float(rec.get("time_run_dist_m", 0) or 0))

    time_run_dist_m = float(rec.get("time_run_dist_m", 0) or 0)

    splits_mmss = rec.get("splits_mmss", []) or []
    splits_sec = [mmss_to_sec(x) for x in splits_mmss if str(x).strip()]
    splits_sec = [s for s in splits_sec if s > 0]
    splits_sec = sorted(splits_sec)

    laps_sec = splits_to_laps(splits_sec) if len(splits_sec) >= 2 else []
    alerts = detect_at_alerts(laps_sec, threshold=3.0)

    pace_sec_km = pace_per_km(time_run_dist_m, time_sec) if time_run_dist_m > 0 else 0.0
    target_time_pred_sec = predict_time_by_same_speed(time_run_dist_m, time_sec, target_m) if time_run_dist_m > 0 else 0.0

    # VO2Maxは「時間走の平均速度」から推定
    v_m_per_min = (time_run_dist_m / time_min) if (time_run_dist_m > 0 and time_min > 0) else 0.0
    vo2 = estimate_vo2max_by_speed(v_m_per_min)

    pace_guide = build_pace_guide(target_m, target_time_pred_sec)
    pace_guide_text = "\n".join([f"- {r['プラン']}: {r['想定タイム']} / {r['目標ラップ']}" for r in pace_guide]) if pace_guide else "- 作成できませんでした"

    alert_lines = "\n".join(
        [f"- {idx}本目（{idx*LAP_M}m）: {prev:.1f}→{cur:.1f}（+{diff:.1f}秒）" for idx, prev, cur, diff in alerts]
    ) if alerts else "- 目立った失速アラートなし"

    dist_race_m = int(rec.get("distance_race_m", 0) or 0)
    dist_race_time_mmss = str(rec.get("distance_race_time_mmss", "") or "").strip()

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
- ②は失速地点を必ず言及（何本目/何m地点）
- ③は維持/目標/突破の3段階（300mラップ）
- ④は熱く前向きに140文字程度
- 距離走の記録が用紙にあれば必ず拾って言及する
- 「何を評価しているか」が分かるように、冒頭で評価軸を一言で示す
- 時間走は3回（①②③）の結果がある前提で、**ベスト（最大距離）**も必ず示す

【選手・種別】
選手名: {name}
推定: {gender_jp}（{time_min}分間走 / {target_m}m）
判定理由: {profile["reason"]}

【時間走（3回）の結果】
- ベスト距離: {best_dist}m
- 平均距離: {avg_dist}m
- ワースト距離: {worst_dist}m

【今回の詳細（このレポートが解析している回）】
- 今回の時間走の距離：{int(time_run_dist_m) if time_run_dist_m else 0}m（最下段）
{dist_race_line}

【通過タイム（秒）】
{splits_sec}

【ラップ（秒）=通過差分】
{laps_sec}

【平均ペース（時間走）】
{sec_to_mmss(pace_sec_km)} /km

【換算参考記録（推定）】
{target_m}m = {sec_to_mmss(target_time_pred_sec)}

【推定VO2Max（推定：時間走の平均速度から）】
{vo2} ml/kg/min

【失速アラート（前の本より+3秒以上）】
{alert_lines}

【③ 目標ラップ表（Pace Guide）】
{pace_guide_text}

【出力フォーマット（この順番で必ず）】
① 科学的ポテンシャル診断 (RESULT / Best)
② ラップ推移 & 失速地点（ATサイン）
③ 目標ラップ表 (Pace Guide)
④ COACH'S EYE (専門的アドバイス)
"""

def generate_text_report(name, profile, rec, all_records):
    prompt = build_report_prompt(name, profile, rec, all_records)
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.4,
    )
    return resp.output_text.strip()

# ==========================================
# Main
# ==========================================
st.markdown("## 🏃 持久走データサイエンス（ベスト回対応 + 用語解説 + 欄外無視）")
st.markdown('<div class="small-note">時間走は①②③から「ベスト回（最大距離）」を自動採用できます</div>', unsafe_allow_html=True)

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

    # --- ベスト回の自動選択（最大距離） ---
    best_rec = pick_best_time_run(records)

    # --- UIで手動選択も可能にする（任意）---
    st.markdown("### ✅ どの回をレポート対象にしますか？")
    labels = []
    for i, r in enumerate(records):
        att = r.get("attempt", i + 1)
        d = int(float(r.get("time_run_dist_m", 0) or 0))
        labels.append(f"{att}回目（{d}m）")

    default_idx = 0
    # best_rec と同じものを初期値にする
    for i, r in enumerate(records):
        if r is best_rec:
            default_idx = i
            break

    idx = st.selectbox("回を選択（デフォルトはベスト回）", range(len(records)), index=default_idx,
                       format_func=lambda i: labels[i])
    rec = records[idx]

    # 種目推定は「選択した回」を基準（距離走記録が取れていれば最強）
    profile = infer_profile(rec, sheet_hints)
    gender_jp = "男子" if profile["gender"] == "male" else "女子"

    st.markdown(f"# 🏃‍♂️ {name} 選手｜能力分析レポート（推定：{gender_jp}）")
    st.caption(f"判定理由: {profile['reason']}")

    # 通過タイム
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

    # 失速アラート
    laps_sec = splits_to_laps(splits_sec) if len(splits_sec) >= 2 else []
    alerts = detect_at_alerts(laps_sec, threshold=3.0)
    if alerts:
        st.warning("⚠️ 失速アラート（前の本より+3秒以上）: " +
                   " / ".join([f"{idx2}本目(+{diff:.1f}s)" for idx2, _, _, diff in alerts]))
    else:
        st.info("失速アラート：目立った失速なし")

    st.markdown("### 📝 文章レポート（画像なし生成）")
    if st.button("📄 詳細レポートを生成（画像なし）"):
        with st.spinner("文章レポート生成中..."):
            try:
                report = generate_text_report(name, profile, rec, records)

                st.markdown("#### レポート本文")
                st.markdown(f'<div class="report-box">{report.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

                st.markdown("#### 用語解説（授業用）")
                st.markdown(f'<div class="glossary-box">{GLOSSARY_TEXT.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"レポート生成エラー: {e}")
