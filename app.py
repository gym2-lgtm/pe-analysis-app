import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import io
import google.generativeai as genai
from PIL import Image
import json
import re

# ==========================================
# 設定：APIキー
# ==========================================
API_KEY = "AIzaSyATM7vIfyhj6vKsZga3fydYLHvAMRVNdzg"

# ==========================================
# 1. AI読み取りエンジン (生徒の文字を解読)
# ==========================================
def analyze_image_with_gemini(image_bytes):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except:
        return None, "画像を読み込めませんでした。"
    
    # プロンプト：生徒向けに最適化
    prompt = """
    持久走の記録用紙を読み取ってください。
    
    【重要：距離設定】
    - 男子は通常 3000m
    - 女子は通常 2100m (記録が短い場合は2100と判断して)
    
    【抽出項目】
    1. 名前 (name): 読めなければ "あなた"
    2. 性別 (gender): "男子" or "女子"
    3. 距離 (distances): 完走距離のリスト
    4. ラップ (laps): 1周ごとのタイム(秒)のリスト
       - 分秒表記(1'20)は秒(80)に変換
       - 累積タイムなら引き算して計算
       
    Output JSON format:
    {"name": "名前", "gender": "男子", "distances": [3000], "laps": [70, 72]}
    """
    
    try:
        response = model.generate_content([prompt, img])
        text = response.text
        # JSON抽出
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0)), None
        else:
            return None, "データが見つかりませんでした。もう一度きれいに撮影してください。"
    except Exception as e:
        return None, f"エラーが発生しました: {e}"

# ==========================================
# 2. 分析ロジック (女子2100m対応)
# ==========================================
class ScienceEngine:
    def __init__(self, gender="男子"):
        self.gender = gender
        # ★ここを修正しました
        if self.gender == "女子":
            self.target_dist = 2100
        else:
            self.target_dist = 3000

    def analyze(self, laps, total_dist):
        if not laps: return "", None

        laps_np = np.array(laps)
        avg_pace = np.mean(laps_np)
        
        # AT値 (3秒落ち)
        at_point = None
        for i in range(1, len(laps)):
            if laps[i] - laps[i-1] >= 3.0:
                at_point = i + 1
                break
        
        # 完走タイム予測
        current_time = sum(laps)
        pred_time = current_time
        
        # もし途中までのデータなら、残りを予測
        if total_dist < self.target_dist:
            remaining = self.target_dist - total_dist
            lap_dist = total_dist / len(laps) # 1周の距離
            if lap_dist > 0:
                laps_needed = remaining / lap_dist
                # 後半の疲労(1.05倍)を考慮
                pred_time += laps_needed * avg_pace * 1.05

        # 生徒へのメッセージ
        advice = f"【{self.target_dist}m 分析結果】\n"
        
        m, s = divmod(pred_time, 60)
        advice += f"🏁 予測タイム: {int(m)}分{int(s):02d}秒\n"
        
        if at_point:
            advice += f"⚠️ {at_point}周目にペースダウンしています。\nここがあなたの『スタミナの壁(AT値)』です。\n"
        else:
            advice += "✅ 最後までペースを守り切れています！素晴らしい！\n"
            
        target = avg_pace * 0.98
        advice += f"\n💡 次回の目標ラップ: {target:.0f}秒\n"
        advice += "このペースを刻めば、記録更新は確実です。"

        return advice, at_point

# ==========================================
# 3. レポート描画 (スマホで見やすいレイアウト)
# ==========================================
class ReportGenerator:
    @staticmethod
    def create_image(data):
        plt.close('all')
        
        try:
            name = data.get("name", "あなた")
            gender = data.get("gender", "男子")
            
            # データ整形
            laps = data.get("laps", [])
            if isinstance(laps, str): laps = [float(x) for x in re.findall(r"[\d\.]+", laps)]
            
            dists = data.get("distances", [3000])
            if isinstance(dists, str): dists = [float(x) for x in re.findall(r"[\d\.]+", dists)]
            
            total_dist = max(dists) if dists else 3000
        except:
            return None

        if not laps: return None

        engine = ScienceEngine(gender)
        advice, at_point = engine.analyze(laps, total_dist)
        
        # A4縦に近い比率（スマホでスクロールして見やすい）
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100, facecolor='white')
        plt.axis('off')
        
        # タイトル
        fig.text(0.5, 0.95, f"{name}さんの分析レポート", fontsize=24, ha='center', weight='bold', color='#1A2A3A')

        # ① 結果サマリ
        ax1 = fig.add_axes([0.1, 0.75, 0.8, 0.15])
        ax1.set_axis_off()
        ax1.add_patch(plt.Rectangle((0,0),1,1,color='#E6F3FF',transform=ax1.transAxes, zorder=0))
        
        m, s = divmod(sum(laps), 60)
        summary = f"距離: {total_dist}m\nタイム: {int(m)}分{int(s):02d}秒\n平均ラップ: {np.mean(laps):.1f}秒"
        ax1.text(0.5, 0.5, summary, fontsize=18, ha='center', va='center', linespacing=1.8)

        # ② グラフ
        ax2 = fig.add_axes([0.1, 0.45, 0.8, 0.25])
        ax2.plot(range(1, len(laps)+1), laps, marker='o', linewidth=3, color='#FF6B6B')
        ax2.set_title("ラップタイムの推移", fontsize=16)
        ax2.set_xlabel("周回", fontsize=14)
        ax2.set_ylabel("秒数", fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.5)
        if at_point:
            ax2.axvline(x=at_point, color='blue', linestyle='--', label='スタミナ切れ')
            ax2.legend(fontsize=12)

        # ③ アドバイス
        ax3 = fig.add_axes([0.1, 0.10, 0.8, 0.30])
        ax3.set_axis_off()
        ax3.add_patch(plt.Rectangle((0,0),1,1,fill=False,edgecolor='#333',linewidth=2,transform=ax3.transAxes))
        ax3.text(0.05, 0.9, "コーチからのアドバイス", fontsize=16, weight='bold')
        ax3.text(0.05, 0.5, advice, fontsize=14, linespacing=1.8, va='center')

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

# ==========================================
# 4. アプリUI (生徒用・超シンプル)
# ==========================================
def main():
    st.set_page_config(page_title="持久走分析", layout="centered") # スマホ向けに中央寄せ
    
    st.title("🏃‍♂️ 持久走分析アプリ")
    st.write("記録用紙の写真をアップロードすると、すぐに分析結果が出ます。")
    
    # 1. 画像アップロード (カメラ起動ボタンになる)
    uploaded_file = st.file_uploader("ここをタップして写真を撮る", type=['png', 'jpg', 'jpeg'])

    # 2. アップロードされたら即実行
    if uploaded_file:
        with st.spinner("AIが分析しています...少々お待ちください"):
            # 画像を表示
            st.image(uploaded_file, caption="読み込んだ画像", width=200)
            
            # AI解析
            data, error = analyze_image_with_gemini(uploaded_file.getvalue())
            
            if data:
                # レポート作成
                japanize_matplotlib.japanize()
                img_buf = ReportGenerator.create_image(data)
                
                if img_buf:
                    st.success("分析完了！")
                    st.image(img_buf, caption="あなたの分析レポート", use_column_width=True)
                    st.markdown("長押しして画像を保存してください👆")
                else:
                    st.error("データがうまく読み取れませんでした。")
            else:
                st.error(error)

if __name__ == "__main__":
    main()
