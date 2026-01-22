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
# 設定：APIキー (埋め込み済み)
# ==========================================
API_KEY = "AIzaSyATM7vIfyhj6vKsZga3fydYLHvAMRVNdzg"

# ==========================================
# 1. AI読み取りエンジン (高速化・即時JSON)
# ==========================================
def analyze_image_with_gemini(image_bytes):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={"response_mime_type": "application/json"}
    )
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except:
        return None
    
    prompt = """
    持久走の記録用紙を読み取ってください。
    全てのラップタイムを正確に抽出することが最重要です。

    【抽出項目】
    1. "name": 名前 (不明なら"選手")
    2. "gender": "男子" or "女子"
    3. "distances": [3000, 3000] のように距離(m)のリスト
    4. "laps": [68, 70, 72...] のように1周ごとのラップ(秒)のリスト。
       - "1'05"は65に変換。
       - 累積タイムしか書かれていない場合は、必ず引き算して「その周回のラップ」を算出すること。

    Output JSON format:
    {"name": str, "gender": str, "distances": list[int], "laps": list[int]}
    """
    
    try:
        response = model.generate_content([prompt, img])
        return json.loads(response.text)
    except:
        return None

# ==========================================
# 2. スーパー・サイエンス・ロジック (3000m特化)
# ==========================================
class ScienceEngine:
    def __init__(self, gender="男子"):
        self.gender = gender
        self.target_dist = 3000 if gender == "男子" else 2000

    def analyze(self, laps, total_dist):
        laps_np = np.array(laps)
        if len(laps) == 0: return "データなし", 0, 0, 0
        
        avg_pace = np.mean(laps_np)
        max_pace = np.max(laps_np)
        min_pace = np.min(laps_np)
        
        # AT値の検出（前の周より3秒以上落ちた最初のポイント）
        at_point = None
        for i in range(1, len(laps)):
            if laps[i] - laps[i-1] >= 3.0:
                at_point = i + 1
                break
        
        # 3000m/2000m 予測タイム
        current_total_time = np.sum(laps_np)
        estimated_time = current_total_time
        if total_dist < self.target_dist:
            remaining_dist = self.target_dist - total_dist
            laps_needed = remaining_dist / (total_dist / len(laps))
            estimated_time += laps_needed * avg_pace * 1.05 # 疲労係数
            
        advice = f"【レース分析】\n"
        advice += f"平均ラップ: {avg_pace:.1f}秒 (最大差: {max_pace - min_pace:.1f}秒)\n"
        
        if at_point:
            advice += f"⚠️ AT値到達: {at_point}周目\nここでガクッとペースが落ちています。ここが現在の限界点です。\n"
        else:
            advice += "✅ AT値未到達: 最後までペースを維持できています。\n"

        advice += f"\n【{self.target_dist}m 戦略・予測】\n"
        m, s = divmod(estimated_time, 60)
        advice += f"予測タイム: {int(m)}分{int(s):02d}秒\n"
        
        target_pace = avg_pace * 0.97
        advice += f"次回の目標ラップ: {target_pace:.0f}秒\n"
        advice += "後半の落ち込みを防ぐため、序盤の入りをあと1秒抑えましょう。"

        return advice, at_point

# ==========================================
# 3. レポート描画エンジン (即時生成)
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_image(data):
        plt.close('all')
        name = data.get("name", "選手")
        gender = data.get("gender", "男子")
        laps = data.get("laps", [])
        dists = data.get("distances", [3000])
        total_dist = max(dists) if dists else 3000
        
        if not laps: return None

        engine = ScienceEngine(gender)
        advice, at_point = engine.analyze(laps, total_dist)
        
        # 描画
        fig = plt.figure(figsize=(11.69, 8.27), dpi=100, facecolor='white')
        plt.axis('off')
        
        # タイトル
        fig.text(0.5, 0.93, f"{name} 様：持久走 科学的分析レポート", fontsize=22, ha='center', weight='bold')

        # ① 分析サマリ (左上)
        ax1 = fig.add_axes([0.05, 0.60, 0.40, 0.25])
        ax1.set_axis_off(); ax1.add_patch(plt.Rectangle((0,0),1,1,color='#F0F8FF',transform=ax1.transAxes))
        
        m, s = divmod(sum(laps), 60)
        summary = f"● 距離: {total_dist}m\n● タイム: {int(m)}分{int(s):02d}秒\n● 平均ラップ: {np.mean(laps):.1f}秒"
        ax1.text(0.05, 0.5, summary, fontsize=14, linespacing=2.0, va='center')
        ax1.text(0, 1.05, "① 記録サマリ", fontsize=14, weight='bold', transform=ax1.transAxes)

        # ② ラップ＆スプリット表 (右上) - 全周回表示
        ax2 = fig.add_axes([0.50, 0.50, 0.45, 0.35])
        ax2.set_axis_off()
        
        header = ["周", "ラップ", "スプリット"]
        table_data = []
        cum = 0
        # スペースの都合上、最大15周まで表示（それ以上は省略）
        display_laps = laps[:15]
        
        for i, lap in enumerate(display_laps):
            cum += lap
            sm, ss = divmod(cum, 60)
            table_data.append([f"{i+1}", f"{lap:.0f}", f"{int(sm)}:{int(ss):02d}"])
            
        t2 = ax2.table(cellText=table_data, colLabels=header, loc='center', cellLoc='center', colColours=["#333"]*3)
        t2.auto_set_font_size(False); t2.set_fontsize(10)
        
        # ヘッダー色調整
        for (r, c), cell in t2.get_celld().items():
            if r == 0: cell.get_text().set_color('white')
            # AT値の行を赤くする
            if at_point and r == at_point:
                cell.set_facecolor('#FFCCCC')

        ax2.text(0, 1.02, "② ラップ / スプリット", fontsize=14, weight='bold', transform=ax2.transAxes)

        # ③ グラフ (左下)
        ax3 = fig.add_axes([0.05, 0.05, 0.40, 0.45])
        ax3.plot(range(1, len(laps)+1), laps, marker='o', linewidth=2, color='#2980B9')
        ax3.set_title("ペース推移グラフ", fontsize=12)
        ax3.set_xlabel("周回"); ax3.set_ylabel("タイム(秒)")
        ax3.grid(True, linestyle='--', alpha=0.6)
        if at_point:
            ax3.axvline(x=at_point, color='red', linestyle='--', label='AT Point')
            ax3.legend()

        # ④ 鬼コーチのアドバイス (右下)
        ax4 = fig.add_axes([0.50, 0.05, 0.45, 0.35])
        ax4.set_axis_off(); ax4.add_patch(plt.Rectangle((0,0),1,1,fill=False,edgecolor='#333',linewidth=2,transform=ax4.transAxes))
        ax4.text(0.05, 0.9, "④ 科学的アドバイス・予測", fontsize=14, weight='bold')
        ax4.text(0.05, 0.4, advice, fontsize=11, linespacing=1.6)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

# ==========================================
# 4. アプリUI (超速モード)
# ==========================================
def main():
    st.set_page_config(page_title="持久走即時分析", layout="wide")
    st.title("⏱️ 3000m/持久走 即時分析システム")
    st.info("画像をアップロードするだけで、AIが読み取り・分析・レポート作成まで一気に行います。")

    uploaded_files = st.file_uploader("記録用紙をアップロード (複数枚OK)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

    if uploaded_files:
        japanize_matplotlib.japanize()
        
        # ZIP作成準備
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            
            # 画像の数だけループ
            cols = st.columns(2)
            for i, file in enumerate(uploaded_files):
                with cols[i % 2]:
                    with st.spinner(f"{file.name} を解析中..."):
                        # 1. 読み取り
                        data = analyze_image_with_gemini(file.getvalue())
                        
                        if data:
                            # 2. レポート作成 (表への転記プロセスをスキップ)
                            img_buf = ReportGenerator.create_image(data)
                            
                            if img_buf:
                                # 3. 即表示
                                st.image(img_buf, caption=f"{data.get('name')}選手のレポート")
                                # ZIPに追加
                                zip_file.writestr(f"{data.get('name')}_report.png", img_buf.getvalue())
                            else:
                                st.error("データ不足で描画できませんでした")
                        else:
                            st.error(f"{file.name}: AI読み取り失敗")

        # 最後にまとめてダウンロードボタン
        st.write("---")
        st.download_button(
            label="📥 全員のレポートを一括ダウンロード (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="race_reports.zip",
            mime="application/zip",
            type="primary"
        )

if __name__ == "__main__":
    main()
