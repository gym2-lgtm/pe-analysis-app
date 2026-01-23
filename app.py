import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, json, os, requests
import matplotlib.font_manager as fm
from PIL import Image, ImageOps
import google.generativeai as genai
import textwrap

# ==========================================
# 1. システム設定 & APIキー取得
# ==========================================
st.set_page_config(page_title="持久走能力徹底分析", layout="wide")

# APIキーの安全な取得
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    # ローカル開発用など、secretsがない場合のフォールバック（必要なら直接入力も可）
    api_key = os.environ.get("GEMINI_API_KEY", "")

if not api_key:
    st.error("【重要】StreamlitのSecrets、または環境変数に 'GEMINI_API_KEY' を設定してください。")
    st.stop()

genai.configure(api_key=api_key)

# ==========================================
# 2. 日本語フォントの強力な確保ロジック
# ==========================================
@st.cache_resource
def get_jp_font():
    """
    Matplotlibで日本語を表示するためのフォントを確保する。
    環境になければGoogle FontsからNotoSansJPをダウンロードする。
    """
    font_dir = "fonts"
    font_name = "NotoSansJP-Regular.ttf"
    font_path = os.path.join(font_dir, font_name)
    
    # フォルダがなければ作成
    if not os.path.exists(font_dir):
        os.makedirs(font_dir)

    # フォントファイルがなければダウンロード
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(font_path, "wb") as f:
                    f.write(response.content)
            else:
                return None # DL失敗
        except:
            return None # ネットワークエラー等

    # フォントマネージャーに追加
    try:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        return prop
    except:
        return None

# フォント読み込み実行
jp_font = get_jp_font()
font_prop_bold = jp_font # 簡易的に同じフォントを使用（Weight変える場合は別途DLが必要だが今回は安定重視）

# ==========================================
# 3. AI解析エンジン (キャッシュ付き・モデル固定)
# ==========================================
@st.cache_data(show_spinner=False)
def analyze_image_with_gemini(image_bytes):
    """
    画像をGeminiに投げ、JSONデータを返す。
    キャッシュ有効化により、UI操作での再実行を防ぐ。
    """
    model_name = "gemini-1.5-flash" # 固定・安定版
    
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        return None, f"モデル初期化エラー: {str(e)}"

    prompt = """
    あなたは陸上長距離の専門分析官です。画像の「持久走記録用紙」からデータを抽出し、以下の厳密なJSON形式のみを出力してください。Markdownタグ(```json)は不要です。

    【ルール】
    1. 用紙に「15分間走」とあれば `record_type_minutes` は 15。男子・目標3000m。
    2. 用紙に「12分間走」とあれば `record_type_minutes` は 12。女子・目標2100m。
    3. 複数回の記録がある場合は全て `records` 配列に入れる。
    4. `coach_advice` は、データの数値（落ち込み等）に基づいた具体的なアドバイスと、生理学的観点のコメントを150文字程度で生成する。

    【出力JSON構造】
    {
      "name": "選手名",
      "record_type_minutes": 15,
      "records": [
        {
          "attempt": 1,
          "distance": 3200, 
          "laps": [60, 62, 65, 68]
        }
      ],
      "coach_advice": "ここにアドバイス"
    }
    """

    try:
        # 画像データをPILオブジェクトからBytesへ（入力がbytesの場合はそのまま使うが、ここではPILを想定してBytes変換済みを受け取るか、PILを渡す）
        # StreamlitのUploadFileをPILに変換してから渡す
        img = Image.open(io.BytesIO(image_bytes))
        
        response = model.generate_content(
            [prompt, img],
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text), None
    except Exception as e:
        return None, f"解析エラー: {str(e)}"

# ==========================================
# 4. レポート描画ロジック (A4横・Matplotlib)
# ==========================================
def create_athlete_report(data):
    # --- データ準備 ---
    name = data.get("name", "選手")
    records = data.get("records", [])
    advice_text = data.get("coach_advice", "データ不足のためアドバイスを生成できません。")
    base_min = int(data.get("record_type_minutes", 15))

    # 自己ベスト特定
    best_record = {"distance": 0, "laps": []}
    if records:
        # distanceが文字列の場合も考慮してfloat変換
        best_record = max(records, key=lambda x: float(str(x.get("distance", 0)).replace("m","")))
    
    best_dist = float(str(best_record.get("distance", 0)).replace("m",""))
    laps = best_record.get("laps", [])

    # 種目別設定
    if base_min == 15:
        target_dist = 3000
        gender_label = "男子"
    else:
        target_dist = 2100
        gender_label = "女子"

    # --- 科学的計算 ---
    # 1. ペース計算
    run_seconds = base_min * 60
    if best_dist > 0:
        mean_pace_sec_per_km = run_seconds / (best_dist / 1000)
        p_min = int(mean_pace_sec_per_km // 60)
        p_sec = int(mean_pace_sec_per_km % 60)
        pace_str = f"{p_min}'{p_sec:02d}/km"
        
        # 100m換算
        pace_100m = run_seconds / (best_dist / 100)
    else:
        pace_str = "-'--/km"
        pace_100m = 0

    # 2. VO2Max (クーパーテスト変法: 12分間走換算)
    # 15分の場合、12分時点の距離を推計 (単純比例)
    dist_12min = best_dist * (12 / base_min)
    vo2max = (dist_12min - 504.9) / 44.73
    if vo2max < 0: vo2max = 0

    # 3. ターゲットタイム (リーゲルの公式: T2 = T1 * (D2/D1)^1.06)
    if best_dist > 0:
        pred_sec = run_seconds * (target_dist / best_dist) ** 1.06
        # 攻めの目標 (98-99%程度に設定)
        target_sec_aggressive = pred_sec * 0.99 
        t_min = int(target_sec_aggressive // 60)
        t_sec = int(target_sec_aggressive % 60)
        target_time_str = f"{t_min}分{t_sec:02d}秒"
    else:
        target_time_str = "--分--秒"

    # --- 描画開始 ---
    fig = plt.figure(figsize=(11.69, 8.27), dpi=150, facecolor='white')
    
    # ヘッダー
    fig.text(0.05, 0.94, "ATHLETE PERFORMANCE REPORT", fontsize=14, color='gray', fontproperties=jp_font)
    fig.text(0.05, 0.88, f"{name} 選手 ｜ 持久走能力徹底分析 ({base_min}分間走)", fontsize=24, weight='bold', color='#1a237e', fontproperties=jp_font)
    fig.lines.append(plt.Line2D([0.05, 0.95], [0.86, 0.86], transform=fig.transFigure, color='#1a237e', linewidth=2))

    # ==========================
    # エリア①: 左上 (科学的ポテンシャル)
    # ==========================
    ax1 = fig.add_axes([0.05, 0.60, 0.35, 0.22]) # [left, bottom, width, height]
    ax1.axis('off')
    
    # 角丸四角形風の背景
    rect = plt.Rectangle((0, 0), 1, 1, transform=ax1.transAxes, color='#f5f5f5', zorder=0)
    ax1.add_patch(rect)
    
    ax1.text(0.05, 0.85, "■ Scientific Diagnosis (Best)", fontsize=12, color='#333', weight='bold', fontproperties=jp_font)
    
    info_text = (
        f"自己ベスト距離: {int(best_dist)} m\n"
        f"平均ペース: {pace_str} ({pace_100m:.1f}秒/100m)\n"
        f"推定VO2Max: {vo2max:.1f} ml/kg/min\n"
        f"----------------------------\n"
        f"【{target_dist}m 目標タイム】\n"
        f" >> {target_time_str}"
    )
    ax1.text(0.05, 0.70, info_text, fontsize=14, va='top', linespacing=1.6, fontproperties=jp_font)

    # ==========================
    # エリア②: 右側 (精密ラップ解析表)
    # ==========================
    ax2 = fig.add_axes([0.45, 0.40, 0.50, 0.42]) 
    ax2.axis('off')
    ax2.set_title("■ Lap Analysis & AT Threshold Check", loc='left', fontsize=12, pad=10, fontproperties=jp_font)

    if records:
        # テーブルデータ作成
        # 最大周回数取得
        max_laps = max([len(r.get("laps", [])) for r in records])
        
        col_labels = ["No."]
        for i, _ in enumerate(records):
            col_labels.extend([f"#{i+1} Lap", f"#{i+1} Split"])
            
        table_data = []
        # 行データ
        for lap_idx in range(max_laps):
            row = [f"{lap_idx+1}"]
            for r in records:
                laps_list = r.get("laps", [])
                if lap_idx < len(laps_list):
                    val = laps_list[lap_idx]
                    split = sum(laps_list[:lap_idx+1])
                    sp_m, sp_s = divmod(split, 60)
                    row.extend([f"{val:.1f}", f"{int(sp_m)}:{int(sp_s):02d}"])
                else:
                    row.extend(["-", "-"])
            table_data.append(row)
        
        # 総距離行
        row_dist = ["Dist"]
        for r in records:
            d = r.get("distance", "-")
            row_dist.extend([f"{d}m", ""])
        table_data.append(row_dist)

        # テーブル描画
        table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)

        # 装飾 & AT判定
        cells = table.get_celld()
        for (r, c), cell in cells.items():
            cell.set_text_props(fontproperties=jp_font)
            if r == 0: # Header
                cell.set_facecolor('#424242')
                cell.set_text_props(color='white', fontproperties=jp_font)
            elif r == len(table_data): # Footer (Distance)
                cell.set_facecolor('#e0e0e0')
                cell.set_text_props(weight='bold', fontproperties=jp_font)
            else:
                # AT判定ロジック (Lap列のみ)
                if c > 0 and c % 2 != 0: # Lap columns (1, 3, 5...)
                    rec_idx = (c - 1) // 2
                    laps_list = records[rec_idx].get("laps", [])
                    current_lap_idx = r - 1
                    
                    if current_lap_idx < len(laps_list) and current_lap_idx > 0:
                        prev = laps_list[current_lap_idx - 1]
                        curr = laps_list[current_lap_idx]
                        if (curr - prev) >= 2.0: # 2秒以上の落ち込み
                            cell.set_facecolor('#ffcdd2') # 薄い赤
                            cell.set_text_props(color='#b71c1c', weight='bold')

    # ==========================
    # エリア③: 左下 (目標ペース配分表)
    # ==========================
    # エリア①の下、エリア④の左
    ax3 = fig.add_axes([0.05, 0.10, 0.35, 0.45])
    ax3.axis('off')
    ax3.text(0, 1.02, f"■ {target_dist}m Target Pace", fontsize=12, weight='bold', fontproperties=jp_font)

    if target_sec_aggressive > 0:
        patterns = [
            ("維持", 1.05),
            ("PB更新", 1.00),
            ("突破", 0.94)
        ]
        
        col3 = ["地点"] + [p[0] for p in patterns]
        row3 = []
        
        check_points = [1000, 2000, 3000] if target_dist == 3000 else [1000, 2000, 2100]
        
        for cp in check_points:
            if cp > target_dist: continue
            r_dat = [f"{cp}m"]
            ratio = cp / target_dist
            for _, factor in patterns:
                tgt_s = target_sec_aggressive * factor * ratio
                tm, ts = divmod(tgt_s, 60)
                r_dat.append(f"{int(tm)}:{int(ts):02d}")
            row3.append(r_dat)
            
        t3 = ax3.table(cellText=row3, colLabels=col3, loc='top', cellLoc='center')
        t3.scale(1, 1.8)
        t3.auto_set_font_size(False)
        t3.set_fontsize(10)
        
        for (r, c), cell in t3.get_celld().items():
            cell.set_text_props(fontproperties=jp_font)
            if r == 0:
                cell.set_facecolor('#1976d2') # 青
                cell.set_text_props(color='white', fontproperties=jp_font)

    # ==========================
    # エリア④: 右下 (AIコーチのアドバイス)
    # ==========================
    ax4 = fig.add_axes([0.45, 0.05, 0.50, 0.30])
    ax4.axis('off')
    
    # 背景 (薄い黄色)
    rect4 = plt.Rectangle((0, 0), 1, 1, transform=ax4.transAxes, color='#fff9c4', zorder=0)
    ax4.add_patch(rect4)
    
    ax4.text(0.02, 0.90, "■ AI Coach's Advice", fontsize=12, color='#e65100', weight='bold', fontproperties=jp_font)
    
    # テキスト整形 (30文字折り返し)
    wrapped_lines = textwrap.wrap(advice_text, width=28)
    final_advice = "\n".join(wrapped_lines)
    
    ax4.text(0.02, 0.80, final_advice, fontsize=10, va='top', linespacing=1.5, fontproperties=jp_font)

    # 画像化して戻す
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

# ==========================================
# 5. UIメイン処理
# ==========================================
st.title("🏃 Data Science Athlete Report")
st.markdown("記録用紙をアップロードすると、**A4一枚の分析レポート**を生成します。")

uploaded_file = st.file_uploader("画像のアップロード", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    # 画像を表示（確認用）
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    
    # 解析ボタン（誤操作防止）
    if st.button("AI解析＆レポート生成"):
        with st.spinner("AIが記録用紙を解析中... (Gemini 1.5 Flash)"):
            file_bytes = uploaded_file.getvalue()
            
            # 1. 解析
            json_data, error = analyze_image_with_gemini(file_bytes)
            
            if error:
                st.error(error)
            else:
                # デバッグ用（本番では消しても良い）
                with st.expander("抽出データを確認"):
                    st.json(json_data)
                
                # 2. レポート生成
                with st.spinner("レポートを描画中..."):
                    try:
                        report_img_buf = create_athlete_report(json_data)
                        
                        st.success("レポート生成完了！")
                        st.image(report_img_buf, caption="Generated Report", use_container_width=True)
                        
                        # ダウンロードボタン
                        st.download_button(
                            label="レポートをダウンロード (PNG)",
                            data=report_img_buf,
                            file_name=f"{json_data.get('name', 'athlete')}_report.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"描画エラー: {e}")
