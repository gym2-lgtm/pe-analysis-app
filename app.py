import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import io
import zipfile
from PIL import Image

# ==========================================
# 1. 科学的分析エンジン
# ==========================================
class ScienceEngine:
    def __init__(self, gender="男子"):
        self.gender = gender
        self.time_limit = 900 if gender == "男子" else 720
        self.target_dist = 3000 if gender == "男子" else 2100

    def calculate_vo2_max(self, distance):
        dist_12min = distance * (12/15) if self.gender == "男子" else distance
        return (dist_12min - 504.9) / 44.73

    def generate_advice(self, laps, total_dist):
        if len(laps) < 3: return "データ不足のため分析できません"
        laps_np = np.array(laps)
        std_dev = np.std(laps_np)
        
        at_point = -1
        for i in range(1, len(laps)):
            if laps[i] - laps[i-1] > 3.0:
                at_point = i + 1
                break
        
        advice = ""
        if std_dev < 2.0: advice += "【精密機械のようなペース管理】\nラップのばらつきが極めて小さく、自分の限界値を把握できています。"
        elif std_dev > 5.0: advice += "【ペース配分の改善が必要】\n変動が大きいです。序盤のオーバーペースが後半の失速を招いています。"
        else: advice += "【標準的なペース配分】\n全体を通して粘り強く走れています。"

        if at_point != -1: advice += f"\n\n【AT値（乳酸閾値）の壁】\n{at_point}周目でガクッとペースが落ちています。ここが現在の『生理学的限界点』です。"
        elif laps[-1] < laps[0]: advice += "\n\n【見事なネガティブ・スプリット】\n後半にペースを上げる余力を残した理想的な展開です。"
        else: advice += "\n\n【高い乳酸耐性】\n大きな失速がなく、高いレベルで乳酸を処理し続けられています。"

        if self.gender == "男子" and total_dist >= 4000: advice += "\n\n【3000m戦略】\n9分台前半が狙えます。序盤から攻めの走りを。"
        elif self.gender == "女子" and total_dist >= 2300: advice += "\n\n【2100m戦略】\n9分15秒切りが見えています。"
        
        return advice

# ==========================================
# 2. レポート描画エンジン
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_image(student_data):
        name = student_data["name"]
        gender = student_data["gender"]
        # データ変換処理
        d_str = str(student_data["distances"]).replace(" ", "").replace("　", "")
        l_str = str(student_data["laps"]).replace(" ", "").replace("　", "")
        dists = [float(x) for x in d_str.split(",") if x]
        laps = [float(x) for x in l_str.split(",") if x]
        
        pb_dist = max(dists)
        pb_idx = dists.index(pb_dist)
        
        engine = ScienceEngine(gender)
        vo2_max = engine.calculate_vo2_max(pb_dist)
        pb_vel = pb_dist / engine.time_limit
        advice_text = engine.generate_advice(laps, pb_dist)
        
        # 描画設定
        fig = plt.figure(figsize=(13, 9.5), facecolor='white', dpi=100)
        target_race = "3000m" if gender == "男子" else "2100m"
        plt.text(0.5, 0.96, f"{name} 様：持久走 科学的分析 ＆ {target_race} 予測ナビ", fontsize=24, ha='center', weight='bold', color='#1A2A3A')

        # ① 科学的データ
        ax1 = fig.add_axes([0.05, 0.62, 0.40, 0.25])
        ax1.axis('off')
        ax1.add_patch(plt.Rectangle((0, 0), 1, 1, color='#F8F9FA', transform=ax1.transAxes))
        eval_text = (f"【PB時の生理学的データ】\n● 性別・種目: {gender} {int(engine.time_limit/60)}分間走\n● 推定VO2 Max: {vo2_max:.1f} ml/kg/min\n● 自己ベスト: {pb_dist}m (第{pb_idx+1}回)\n● 平均秒速: {pb_vel:.2f} m/s\n\n【専門的評価】\n心肺機能に基づくと、{target_race}走において高い適性を示しています。")
        ax1.text(0.05, 0.5, eval_text, fontsize=11, linespacing=1.8, va='center')
        ax1.text(0, 1.05, "① 科学的データによる走力評価", fontsize=15, weight='bold', transform=ax1.transAxes)

        # ② 周回データ
        ax2 = fig.add_axes([0.53, 0.52, 0.43, 0.38])
        ax2.axis('off')
        header = ["周回", "ラップ", "累積"]
        table_data = []
        cum_time = 0
        display_limit = min(len(laps), 13)
        for i in range(display_limit):
            cum_time += laps[i]
            table_data.append([f"{i+1}周", f"{laps[i]:.0f}s", f"{int(cum_time//60)}:{int(cum_time%60):02d}"])
        t2 = ax2.table(cellText=table_data, colLabels=header, loc='center', cellLoc='center', colColours=["#1A2A3A"]*3)
        t2.auto_set_font_size(False); t2.set_fontsize(9)
        for i in range(1, len(table_data)):
            if laps[i] - laps[i-1] >= 3.0: t2.get_celld()[(i+1, 1)].set_facecolor('#FFDADA')
        for (r, c), cell in t2.get_celld().items():
            if r == 0: cell.get_text().set_color('white')
            cell.set_height(0.05)
        ax2.text(0, 1.08, "② ラップタイム精密分析", fontsize=13, weight='bold', transform=ax2.transAxes)

        # ③ 目標タイム表
        ax3 = fig.add_axes([0.05, 0.05, 0.43, 0.45])
        ax3.axis('off')
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
        t3.auto_set_font_size(False); t3.set_fontsize(9)
        for (r, c), cell in t3.get_celld().items():
            if r == 0: cell.get_text().set_color('white'); cell.set_height(0.12)
            else: cell.set_height(0.08)
        ax3.text(0, 1.10, f"③ {target_race}走：目標通過タイム表", fontsize=15, weight='bold', color='#2980B9', transform=ax3.transAxes)

        # ④ 科学的アドバイス
        ax4 = fig.add_axes([0.53, 0.05, 0.43, 0.42])
        ax4.axis('off')
        ax4.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#1A2A3A', linewidth=1.2, transform=ax4.transAxes))
        ax4.text(0.05, 0.5, advice_text, fontsize=11, linespacing=2.0, va='center')
        ax4.text(0, 1.05, "④ 科学的分析と実戦戦術", fontsize=15, weight='bold', transform=ax4.transAxes)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

# ==========================================
# 3. アプリケーション画面
# ==========================================
def main():
    st.set_page_config(page_title="持久走分析エージェント", layout="wide")
    st.title("🏃‍♂️ 持久走・科学的分析エージェント")
    
    with st.sidebar:
        st.header("設定")
        st.info("手動入力、または将来的な画像読み込みに対応しています。")

    if 'student_df' not in st.session_state:
        st.session_state.student_df = pd.DataFrame(columns=["name", "gender", "distances", "laps"])

    # テストデータ追加ボタン
    if st.sidebar.button("テスト用データを追加(増田くん)"):
        new_row = {"name": "増田", "gender": "男子", "distances": "3000, 3100, 3200", "laps": "60, 62, 65, 68, 70"}
        st.session_state.student_df = pd.concat([st.session_state.student_df, pd.DataFrame([new_row])], ignore_index=True)

    st.subheader("データの編集・入力")
    edited_df = st.data_editor(
        st.session_state.student_df,
        num_rows="dynamic",
        column_config={
            "name": "氏名",
            "gender": st.column_config.SelectboxColumn("性別", options=["男子", "女子"]),
            "distances": "記録(m) カンマ区切り",
            "laps": "ラップ(秒) カンマ区切り"
        }
    )

    if st.button("🚀 診断レポートを一括生成", type="primary"):
        st.subheader("生成レポート")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            cols = st.columns(2)
            for idx, row in edited_df.iterrows():
                if not row["name"] or not row["laps"]: continue
                with cols[idx % 2]:
                    try:
                        img_buf = ReportGenerator.create_image(row)
                        st.image(img_buf, caption=f"{row['name']}様のレポート")
                        zip_file.writestr(f"{row['name']}_report.png", img_buf.getvalue())
                    except Exception as e:
                        st.error(f"{row['name']}のエラー: {e}")
        st.download_button("📥 一括ダウンロード (ZIP)", data=zip_buffer.getvalue(), file_name="reports.zip", mime="application/zip")

if __name__ == "__main__":
    main()
