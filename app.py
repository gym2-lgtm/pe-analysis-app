import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import io
import zipfile
import google.generativeai as genai
from PIL import Image
import json

# ==========================================
# 設定：ここに先生の鍵を埋め込みました
# ==========================================
API_KEY = "AIzaSyATM7vIfyhj6vKsZga3fydYLHvAMRVNdzg"

# ==========================================
# 1. AI読み取りエンジン (Gemini 1.5 Flash)
# ==========================================
def analyze_image_with_gemini(image_bytes):
    """Geminiを使って画像を解析し、JSONデータを返す"""
    # 埋め込んだキーを使用
    genai.configure(api_key=API_KEY)
    
    # 高速・軽量なFlashモデルを使用
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except:
        return "画像の読み込みに失敗しました"
    
    prompt = """
    あなたは学校の先生の助手です。アップロードされた持久走の記録用紙（手書き）を読み取り、以下の情報を抽出してください。
    
    【必須抽出項目】
    1. 名前 (name): 読み取れなければ "不明"
    2. 性別 (gender): "男子" または "女子" (わからなければ男子)
    3. 距離 (distances): 完走した距離(m)のリスト (例: [3000, 3100])。複数回ある場合は全て。
    4. ラップタイム (laps): 1周ごとのタイム(秒)のリスト (例: [60, 62, 65])。
       - 分・秒で書かれている場合(例 1'05)は秒に変換(65)すること。
       - 累積タイムしか書かれていない場合は、引き算してラップを算出すること。

    出力は以下のPython辞書形式（JSON）のみを行ってください。余計な文章やマークダウン(```json等)は不要です。
    {
        "name": "増田",
        "gender": "男子",
        "distances": "3000, 3100",
        "laps": "60, 62, 65, 68"
    }
    """
    
    try:
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"エラー: {e}"

# ==========================================
# 2. 科学的分析エンジン
# ==========================================
class SuperScienceEngine:
    def __init__(self, gender="男子"):
        self.gender = gender
        self.time_limit = 900 if gender == "男子" else 720
        self.target_dist = 3000 if gender == "男子" else 2100

    def calculate_vo2_max(self, distance):
        dist_12min = distance * (12/15) if self.gender == "男子" else distance
        return (dist_12min - 504.9) / 44.73

    def generate_detailed_advice(self, laps, total_dist):
        if len(laps) < 3: return "データ不足のため分析できません。"
        
        laps_np = np.array(laps)
        avg_pace = np.mean(laps_np)
        std_dev = np.std(laps_np)
        drop_off = np.max(laps_np) - np.min(laps_np)
        
        advice = ""
        cv = (std_dev / avg_pace) * 100
        
        if cv < 3.0:
            advice += "【精密機械のようなペース管理】\nラップの変動係数が極めて低く、体内時計が正確です。\n"
        elif cv > 8.0:
            advice += f"【ペース配分の改善余地】\nラップに最大{drop_off:.0f}秒の乱高下があります。序盤のオーバーペースに注意。\n"
        else:
            advice += "【安定した走力】\n全体を通して粘り強く走れています。\n"

        first_half = np.mean(laps[:len(laps)//2])
        second_half = np.mean(laps[len(laps)//2:])
        if second_half < first_half:
            advice += "後半にペースが上がる「ネガティブ・スプリット」を達成しており、理想的です。\n"
        
        target_pace = avg_pace * 0.98
        advice += "\n【今後の強化指針】\n"
        if total_dist >= (3000 if self.gender == "男子" else 2100):
            advice += f"すでに高水準です。設定{target_pace:.0f}秒でのインターバル走が有効です。"
        else:
            advice += "まずは基礎スタミナの強化が必要です。ビルドアップ走に取り組みましょう。"

        return advice

# ==========================================
# 3. レポート描画エンジン
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_image(student_data):
        plt.close('all')
        name = student_data["name"]
        gender = student_data["gender"]
        
        # データ整形
        try:
            d_str = str(student_data["distances"]).replace(" ", "").replace("　", "").replace("[", "").replace("]", "").replace("'", "")
            l_str = str(student_data["laps"]).replace(" ", "").replace("　", "").replace("[", "").replace("]", "").replace("'", "")
            dists = [float(x) for x in d_str.split(",") if x and x.replace('.','',1).isdigit()]
            laps = [float(x) for x in l_str.split(",") if x and x.replace('.','',1).isdigit()]
        except:
            return None

        if not dists or not laps: return None

        pb_dist = max(dists)
        pb_idx = dists.index(pb_dist)
        
        engine = SuperScienceEngine(gender)
        vo2_max = engine.calculate_vo2_max(pb_dist)
        pb_vel = pb_dist / engine.time_limit
        advice_text = engine.generate_detailed_advice(laps, pb_dist)
        
        fig = plt.figure(figsize=(11.69, 8.27), dpi=100, facecolor='white')
        plt.axis('off')

        target_race = "3000m" if gender == "男子" else "2100m"
        fig.text(0.5, 0.92, f"{name} 様：持久走 科学的分析 ＆ {target_race} 予測ナビ", fontsize=20, ha='center', weight='bold', color='#1A2A3A')

        ax1 = fig.add_axes([0.05, 0.58, 0.40, 0.28])
        ax1.set_axis_off()
        ax1.add_patch(plt.Rectangle((0, 0), 1, 1, color='#F8F9FA', transform=ax1.transAxes))
        eval_text = (f"【PB時の生理学的データ】\n● 性別・種目: {gender} {int(engine.time_limit/60)}分間走\n● 推定VO2 Max: {vo2_max:.1f} ml/kg/min\n● 自己ベスト: {pb_dist}m (第{pb_idx+1}回)\n● 平均秒速: {pb_vel:.2f} m/s\n\n【専門的評価】\n算出されたVO2 Maxに基づくと、{target_race}走において高い適性を示しています。")
        ax1.text(0.05, 0.5, eval_text, fontsize=10, linespacing=1.9, va='center')
        ax1.text(0, 1.05, "① 科学的データによる走力評価", fontsize=12, weight='bold', transform=ax1.transAxes)

        ax2 = fig.add_axes([0.50, 0.50, 0.45, 0.36])
        ax2.set_axis_off()
        header = ["周回", "ラップ", "累積"]
        table_data = []
        cum_time = 0
        display_limit = min(len(laps), 13)
        for i in range(display_limit):
            cum_time += laps[i]
            m, s = divmod(cum_time, 60)
            table_data.append([f"{i+1}周", f"{laps[i]:.0f}s", f"{int(m)}:{int(s):02d}"])
        t2 = ax2.table(cellText=table_data, colLabels=header, loc='center', cellLoc='center', colColours=["#1A2A3A"]*3)
        t2.auto_set_font_size(False); t2.set_fontsize(9)
        for i in range(1, len(table_data)):
            if laps[i] - laps[i-1] >= 3.0: t2.get_celld()[(i+1, 1)].set_facecolor('#FFDADA')
        for (r, c), cell in t2.get_celld().items():
            if r == 0: cell.get_text().set_color('white'); cell.set_height(0.055)
        ax2.text(0, 1.02, "② ラップタイム精密分析", fontsize=12, weight='bold', transform=ax2.transAxes)

        ax3 = fig.add_axes([0.05, 0.05, 0.40, 0.45])
        ax3.set_axis_off()
        t_base = engine.target_dist / pb_vel
        targets = [t_base, t_base*0.98, t_base*0.96, t_base*0.94]
        header3 = ["周回", "PB維持", "PB超え", "大幅更新", "限界突破"]
        rows3 = []
        total_laps_target = int(engine.target_dist / 300)
        for lp in range(1, total_laps_target + 1):
            row = [f"{lp}周"]
            for v in targets:
                st_time = v * (lp/total_laps_target)
                row.append(f"{int(st_time//60)}:{int(st_time%60):02d}")
            rows3.append(row)
        t3 = ax3.table(cellText=rows3, colLabels=header3, loc='center', cellLoc='center', colColours=["#2980B9"]*5)
        t3.auto_set_font_size(False); t3.set_fontsize(8)
        for (r, c), cell in t3.get_celld().items():
            if r == 0: cell.get_text().set_color('white'); cell.set_height(0.12)
            else: cell.set_height(0.06)
        ax3.text(0, 1.05, f"③ {target_race}走：目標通過タイム表", fontsize=12, weight='bold', color='#2980B9', transform=ax3.transAxes)

        ax4 = fig.add_axes([0.50, 0.05, 0.45, 0.38])
        ax4.set_axis_off()
        ax4.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#1A2A3A', linewidth=1.2, transform=ax4.transAxes))
        ax4.text(0.05, 0.5, advice_text, fontsize=9, linespacing=1.8, va='center')
        ax4.text(0, 1.05, "④ 科学的分析と強化指針", fontsize=12, weight='bold', transform=ax4.transAxes)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.2)
        plt.close()
        buf.seek(0)
        return buf

# ==========================================
# 4. アプリケーション画面 (AI Vision搭載・自動化版)
# ==========================================
def main():
    st.set_page_config(page_title="持久走分析エージェント", layout="wide")
    st.title("🏃‍♂️ 持久走・科学的分析 (AI自動読み取り)")
    
    # セッション状態の初期化
    if 'student_df' not in st.session_state:
        st.session_state.student_df = pd.DataFrame(columns=["name", "gender", "distances", "laps"])

    # 1. 画像アップロード (ここに入れるだけで動く！)
    st.info("【使い方】\n記録用紙の写真をアップロードしてください。AIが自動で読み取って表に追加します。")
    uploaded_files = st.file_uploader("📸 記録用紙をアップロード", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

    # 2. 自動解析ロジック (アップロードされたら即実行)
    if uploaded_files:
        # まだ解析していないファイルだけ処理する
        for file in uploaded_files:
            # ファイル名で重複チェック（簡易的）
            if 'processed_files' not in st.session_state:
                st.session_state.processed_files = []
            
            if file.name not in st.session_state.processed_files:
                with st.spinner(f"AIが解析中... ({file.name})"):
                    try:
                        bytes_data = file.getvalue()
                        result_text = analyze_image_with_gemini(bytes_data)
                        
                        # JSON抽出
                        start = result_text.find('{')
                        end = result_text.rfind('}') + 1
                        if start != -1 and end != -1:
                            json_str = result_text[start:end]
                            data = json.loads(json_str)
                            
                            new_row = {
                                "name": data.get("name", "不明"),
                                "gender": data.get("gender", "男子"),
                                "distances": str(data.get("distances", "3000")).replace("[","").replace("]",""),
                                "laps": str(data.get("laps", "")).replace("[","").replace("]","")
                            }
                            st.session_state.student_df = pd.concat([st.session_state.student_df, pd.DataFrame([new_row])], ignore_index=True)
                            st.session_state.processed_files.append(file.name)
                            st.success(f"読み取り成功: {new_row['name']}さん")
                        else:
                            st.error(f"読み取り失敗: {file.name}")
                    except Exception as e:
                        st.error(f"エラー: {e}")

    # 3. 編集・確認エリア
    st.subheader("📝 データ確認・修正")
    edited_df = st.data_editor(
        st.session_state.student_df,
        num_rows="dynamic",
        column_config={
            "name": "氏名",
            "gender": st.column_config.SelectboxColumn("性別", options=["男子", "女子"]),
            "distances": "記録(m)",
            "laps": "ラップ(秒)"
        }
    )

    # 4. レポート生成
    if st.button("🚀 診断レポートを一括生成", type="primary"):
        if len(edited_df) > 0:
            japanize_matplotlib.japanize()
            zip_buffer = io.BytesIO()
            has_data = False
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                cols = st.columns(2)
                for idx, row in edited_df.iterrows():
                    if not row["name"]: continue
                    has_data = True
                    with cols[idx % 2]:
                        try:
                            img_buf = ReportGenerator.create_image(row)
                            if img_buf:
                                st.image(img_buf, caption=f"{row['name']}様のレポート")
                                zip_file.writestr(f"{row['name']}_report.png", img_buf.getvalue())
                        except Exception as e:
                            st.error(f"描画エラー: {e}")
            
            if has_data:
                st.download_button("📥 ダウンロード (ZIP)", data=zip_buffer.getvalue(), file_name="reports.zip", mime="application/zip")
        else:
            st.warning("データがありません。画像をアップロードするか、直接入力してください。")

if __name__ == "__main__":
    main()
