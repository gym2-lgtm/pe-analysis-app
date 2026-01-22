import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import io
import zipfile

# ==========================================
# 1. スーパー・サイエンス・ロジック (アドバイス生成部)
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
        """ルールベースで長文の科学的アドバイスを生成"""
        if len(laps) < 3: return "データ不足のため分析できません。"
        
        laps_np = np.array(laps)
        avg_pace = np.mean(laps_np)
        best_lap = np.min(laps_np)
        worst_lap = np.max(laps_np)
        std_dev = np.std(laps_np)
        drop_off = (worst_lap - best_lap)
        
        advice = ""

        # ① ペース変動係数による評価
        cv = (std_dev / avg_pace) * 100 # 変動係数
        if cv < 3.0:
            advice += "【精密機械のようなペース管理】\n"
            advice += f"ラップの変動係数が{cv:.1f}%と極めて低く、体内時計が正確です。\n"
            advice += "AT値（無酸素性作業閾値）ギリギリのラインを感覚的に把握できています。\n"
        elif cv > 8.0:
            advice += "【ペース配分の最適化が必要】\n"
            advice += f"ラップの乱高下（最大{drop_off:.0f}秒差）が見られます。\n"
            advice += "序盤の乳酸蓄積が、後半の急激な失速（OBLA到達）を招いています。\n"
        else:
            advice += "【安定した走力】\n"
            advice += "全体を通して大きな崩れがなく、粘り強く走れています。\n"

        # ② スプリット分析
        first_half = np.mean(laps[:len(laps)//2])
        second_half = np.mean(laps[len(laps)//2:])
        if second_half < first_half:
            advice += "後半にペースが上がる「ネガティブ・スプリット」を達成しており、\n"
            advice += "心肺機能に余力を残した理想的な展開です。\n"
        else:
            diff = second_half - first_half
            if diff > 5.0:
                advice += f"後半に平均{diff:.1f}秒の落ち込みがあります。\n"
                advice += "筋持久力よりも、最大酸素摂取量(VO2Max)の向上が課題です。\n"

        # ③ 具体的なトレーニング提案
        target_pace = avg_pace * 0.98 # 2%向上
        advice += "\n【今後の強化指針】\n"
        if total_dist >= (3000 if self.gender == "男子" else 2100):
            advice += "すでに高水準です。スピード持久力を高めるため、\n"
            advice += f"400mインターバル走を「設定{target_pace:.0f}秒」で行いましょう。"
        else:
            advice += "まずは基礎スタミナの強化が必要です。\n"
            advice += "20分間のビルドアップ走（徐々にペースを上げる）が有効です。"

        return advice

# ==========================================
# 2. レポート描画エンジン (レイアウト完全固定版)
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_image(student_data):
        # 毎回描画領域をクリア（これが亡霊線を消す鍵）
        plt.close('all')
        
        name = student_data["name"]
        gender = student_data["gender"]
        
        # データ変換
        d_str = str(student_data["distances"]).replace(" ", "").replace("　", "")
        l_str = str(student_data["laps"]).replace(" ", "").replace("　", "")
        dists = [float(x) for x in d_str.split(",") if x]
        laps = [float(x) for x in l_str.split(",") if x]
        
        pb_dist = max(dists)
        pb_idx = dists.index(pb_dist)
        
        engine = SuperScienceEngine(gender)
        vo2_max = engine.calculate_vo2_max(pb_dist)
        pb_vel = pb_dist / engine.time_limit
        advice_text = engine.generate_detailed_advice(laps, pb_dist)
        
        # --- 描画設定 (A4横・100dpi固定) ---
        fig = plt.figure(figsize=(11.69, 8.27), dpi=100, facecolor='white')
        
        # 不要な枠線を消すための親設定
        plt.axis('off')

        # タイトル (位置調整: y=0.92)
        target_race = "3000m" if gender == "男子" else "2100m"
        fig.text(0.5, 0.92, f"{name} 様：持久走 科学的分析 ＆ {target_race} 予測ナビ", 
                 fontsize=20, ha='center', weight='bold', color='#1A2A3A')

        # ① 科学的データ (左上)
        ax1 = fig.add_axes([0.05, 0.58, 0.40, 0.28]) # 座標微調整
        ax1.set_axis_off() # ★完全に消す
        ax1.add_patch(plt.Rectangle((0, 0), 1, 1, color='#F8F9FA', transform=ax1.transAxes))
        
        eval_text = (
            f"【PB時の生理学的データ】\n"
            f"● 性別・種目: {gender} {int(engine.time_limit/60)}分間走\n"
            f"● 推定VO2 Max: {vo2_max:.1f} ml/kg/min\n"
            f"● 自己ベスト: {pb_dist}m (第{pb_idx+1}回)\n"
            f"● 平均秒速: {pb_vel:.2f} m/s\n\n"
            f"【専門的評価】\n"
            f"算出されたVO2 Maxに基づくと、{target_race}走において\n"
            f"高い適性を示しています。"
        )
        ax1.text(0.05, 0.5, eval_text, fontsize=10, linespacing=1.9, va='center')
        ax1.text(0, 1.05, "① 科学的データによる走力評価", fontsize=12, weight='bold', transform=ax1.transAxes)

        # ② 周回データ (右上)
        ax2 = fig.add_axes([0.50, 0.50, 0.45, 0.36])
        ax2.set_axis_off() # ★完全に消す
        
        header = ["周回", "ラップ", "累積"]
        table_data = []
        cum_time = 0
        display_limit = min(len(laps), 13)
        
        for i in range(display_limit):
            cum_time += laps[i]
            m, s = divmod(cum_time, 60)
            cum_str = f"{int(m)}:{int(s):02d}"
            table_data.append([f"{i+1}周", f"{laps[i]:.0f}s", cum_str])
        
        t2 = ax2.table(cellText=table_data, colLabels=header, loc='center', cellLoc='center', colColours=["#1A2A3A"]*3)
        t2.auto_set_font_size(False); t2.set_fontsize(9)
        
        # AT値ハイライト
        for i in range(1, len(table_data)):
            if laps[i] - laps[i-1] >= 3.0:
                t2.get_celld()[(i+1, 1)].set_facecolor('#FFDADA')
        
        for (r, c), cell in t2.get_celld().items():
            if r == 0: cell.get_text().set_color('white')
            cell.set_height(0.055)
        ax2.text(0, 1.02, "② ラップタイム精密分析", fontsize=12, weight='bold', transform=ax2.transAxes)

        # ③ 目標タイム表 (左下)
        ax3 = fig.add_axes([0.05, 0.05, 0.40, 0.45])
        ax3.set_axis_off() # ★完全に消す
        
        t_base = engine.target_dist / pb_vel
        targets = [t_base, t_base*0.98, t_base*0.96, t_base*0.94]
        header3 = ["周回", "PB維持", "PB超え", "大幅更新", "限界突破"]
        
        rows3 = []
        lap_unit = 300 
        total_laps_target = int(engine.target_dist / lap_unit)
        
        for lp in range(1, total_laps_target + 1):
            row = [f"{lp}周"]
            for v in targets:
                split_time = v * (lp/total_laps_target)
                row.append(f"{int(split_time//60)}:{int(split_time%60):02d}")
            rows3.append(row)
            
        t3 = ax3.table(cellText=rows3, colLabels=header3, loc='center', cellLoc='center', colColours=["#2980B9"]*5)
        t3.auto_set_font_size(False); t3.set_fontsize(8)
        for (r, c), cell in t3.get_celld().items():
            if r == 0: cell.get_text().set_color('white'); cell.set_height(0.12)
            else: cell.set_height(0.06)
        ax3.text(0, 1.05, f"③ {target_race}走：目標通過タイム表", fontsize=12, weight='bold', color='#2980B9', transform=ax3.transAxes)

        # ④ 科学的アドバイス (右下)
        ax4 = fig.add_axes([0.50, 0.05, 0.45, 0.38])
        ax4.set_axis_off() # ★完全に消す
        ax4.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#1A2A3A', linewidth=1.2, transform=ax4.transAxes))
        
        ax4.text(0.05, 0.5, advice_text, fontsize=9, linespacing=1.8, va='center')
        ax4.text(0, 1.05, "④ 科学的分析と強化指針", fontsize=12, weight='bold', transform=ax4.transAxes)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.2)
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
        st.info("データ修正・画像アップロード対応")

    if 'student_df' not in st.session_state:
        st.session_state.student_df = pd.DataFrame(columns=["name", "gender", "distances", "laps"])

    # 画像アップローダー
    uploaded_files = st.file_uploader("📸 記録用紙をアップロード", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

    # データ編集
    st.subheader("📝 データ編集")
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
    
    if st.button("増田くんデータを入力"):
        new_row = {"name": "増田", "gender": "男子", "distances": "3000, 3100, 3200", "laps": "60, 62, 65, 68, 70"}
        st.session_state.student_df = pd.concat([st.session_state.student_df, pd.DataFrame([new_row])], ignore_index=True)

    if st.button("🚀 診断レポートを一括生成", type="primary"):
        st.subheader("生成レポート")
        japanize_matplotlib.japanize() # フォント適用
        
        cols = st.columns(2)
        for idx, row in edited_df.iterrows():
            if not row["name"] or not row["laps"]: continue
            with cols[idx % 2]:
                try:
                    img_buf = ReportGenerator.create_image(row)
                    st.image(img_buf, caption=f"{row['name']}様のレポート")
                except Exception as e:
                    st.error(f"{row['name']}のエラー: {e}")

if __name__ == "__main__":
    main()
