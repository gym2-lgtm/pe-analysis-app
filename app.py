import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, requests, json, re, os, base64
import matplotlib.font_manager as fm
from PIL import Image, ImageOps

# ---------------------------------------------------------
# 1. 設定と準備
# ---------------------------------------------------------
API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# ★修正点: ブロック回避のための「ブラウザ偽装」技術を使用
@st.cache_resource
def load_japanese_font():
    font_path = "NotoSansJP-Regular.ttf"
    # 安定しているGitHubのURLを使用
    url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
    
    try:
        if not os.path.exists(font_path):
            # 【重要】ここで「私はブラウザです」と名乗ることでブロックを回避
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status() # エラーなら即座に停止して通知
            
            with open(font_path, "wb") as f:
                f.write(response.content)
        
        # フォント読み込み
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Noto Sans JP'
        return fm.FontProperties(fname=font_path)
        
    except Exception as e:
        # 万が一失敗した場合、アプリを止めずにエラーを表示
        st.error(f"フォント読み込みエラー: {e}")
        return None

# ---------------------------------------------------------
# 2. AI解析エンジン
# ---------------------------------------------------------
def run_ai_analysis(img_bytes):
    if not API_KEY:
        return None, "APIキーが設定されていません。Secretsを確認してください。"

    b64_image = base64.b64encode(img_bytes).decode()

    # モデル自動選定
    try:
        models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
        resp = requests.get(models_url)
        if resp.status_code != 200:
            return None, f"API通信エラー({resp.status_code}): キーが無効の可能性があります。"
            
        model_data = resp.json()
        # 利用可能なモデルから 'generateContent' ができるものを抽出
        available_models = [
            m['name'].split('/')[-1] 
            for m in model_data.get('models', []) 
            if 'generateContent' in m.get('supportedGenerationMethods', [])
        ]
        # flashを優先、なければリストの最初を使う
        target_model = next((m for m in available_models if "flash" in m), available_models[0])
        
    except Exception as e:
        return None, f"モデル検出エラー: {str(e)}"

    # プロンプト（命令文）
    prompt = """
    あなたは陸上競技のデータアナリストです。
    アップロードされた「持久走記録用紙」の画像を読み取り、以下のデータをJSON形式のみで出力してください。
    
    【ルール】
    - 余計な解説や挨拶は一切不要です。
    - JSONデータのみを返してください。

    【抽出項目】
    1. name: 生徒の名前（読み取れなければ "選手"）
    2. long_run_dist: 上段の15分間走(または12分間走)の記録(m)。数値のみ。空欄なら0。
    3. tt_laps: 下段の3000m(または2100m)のラップタイム(秒)のリスト。
       例: "65.0" や "1'05" は 65.0 に変換。

    【出力JSONの例】
    {
      "name": "山田 太郎",
      "long_run_dist": 4050,
      "tt_laps": [65, 68, 70, 72, 68]
    }
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}}]}],
        "generationConfig": {"response_mime_type": "application/json"},
        "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        result = response.json()
        
        if "error" in result:
            return None, f"AI解析エラー: {result['error']['message']}"
            
        raw_text = result['candidates'][0]['content']['parts'][0]['text']
        
        # 正規表現でJSON部分だけを強力に抜き出す
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0)), None
        else:
            return None, f"データを読み取れませんでした。\nAIの応答: {raw_text[:100]}..."
            
    except Exception as e:
        return None, f"システムエラー: {str(e)}"

# ---------------------------------------------------------
# 3. レポート作成（可視化）
# ---------------------------------------------------------
def create_report_image(data):
    fp = load_japanese_font()
    font_arg = {'fontproperties': fp} if fp else {}
    
    # データ取り出し（エラー防止付き）
    try: laps = np.array([float(x) for x in data.get("tt_laps", [])])
    except: laps = np.array([])
    
    try: dist = float(data.get("long_run_dist", 0))
    except: dist = 0.0
    
    name = data.get("name", "選手")

    target_dist = 3000 if dist > 3200 else 2100
    base_time_min = 15 if target_dist == 3000 else 12

    if dist > 0:
        potential_sec = (base_time_min * 60) * (target_dist / dist)**1.06
        vo2_max = (dist * (12/base_time_min) - 504.9) / 44.73
        vo2_max = max(vo2_max, 0)
    else:
        potential_sec = None
        vo2_max = 0

    # 描画キャンバス
    fig = plt.figure(figsize=(11.69, 8.27), facecolor='white', dpi=100)
    
    # ヘッダー
    fig.text(0.05, 0.94, f"持久走 科学的分析レポート", fontsize=24, weight='bold', color='#1a237e', **font_arg)
    fig.text(0.05, 0.90, f"氏名: {name}　|　基準データ: {base_time_min}分間走 {int(dist)}m", fontsize=14, color='#333', **font_arg)

    # ① 生理学的ポテンシャル
    ax1 = fig.add_axes([0.05, 0.55, 0.42, 0.30])
    ax1.set_axis_off()
    ax1.set_title("① 生理学的ポテンシャル評価", fontsize=16, loc='left', color='#0d47a1', weight='bold', **font_arg)
    
    text_content = f"■ 推定VO2Max: {vo2_max:.1f} ml/kg/min\n"
    if potential_sec:
        m, s = divmod(potential_sec, 60)
        text_content += f"■ {target_dist}m 理論限界タイム: {int(m)}分{int(s):02d}秒\n\n"
        text_content += f"【AIコーチの評価】\n君の心肺機能（エンジン）に基づくと、\n{target_dist}mを『{int(m)}分{int(s):02d}秒』で走る\n潜在能力を持っています。\n今の記録に満足せず、上を目指せます！"
    else:
        text_content += "※15分間走(12分間走)の距離が\n読み取れませんでした。\n用紙上段の記入を確認してください。"
    ax1.text(0.02, 0.85, text_content, fontsize=12, va='top', linespacing=1.8, **font_arg)
    rect1 = plt.Rectangle((0,0), 1, 1, fill=False, edgecolor='#ddd', transform=ax1.transAxes)
    ax1.add_patch(rect1)

    # ② ラップ表
    ax2 = fig.add_axes([0.52, 0.55, 0.43, 0.30])
    ax2.set_axis_off()
    ax2.set_title("② 周回精密データ", fontsize=16, loc='left', color='#0d47a1', weight='bold', **font_arg)
    if len(laps) > 0:
        table_data = []
        for i, lap in enumerate(laps[:10]):
            diff = lap - laps[i-1] if i > 0 else 0
            mark = "▼DN" if diff >= 2.0 else ("▲UP" if diff <= -1.5 else "―")
            table_data.append([f"{i+1}周", f"{lap:.1f}秒", mark])
        col_labels = ["周回", "ラップ", "傾向"]
        table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.scale(1, 1.4)
        if fp:
            for key, cell in table.get_celld().values(): cell.set_text_props(fontproperties=fp)
    else:
        ax2.text(0.1, 0.5, "ラップデータなし", **font_arg)

    # ③ 目標設定
    ax3 = fig.add_axes([0.05, 0.10, 0.42, 0.35])
    ax3.set_axis_off()
    ax3.set_title("③ 次回の目標設定", fontsize=16, loc='left', color='#0d47a1', weight='bold', **font_arg)
    if potential_sec:
        base_pace = potential_sec / (target_dist / 300)
        goals = [
            ["現状維持", f"{base_pace*1.05:.1f}秒", "今の走りを安定させる"],
            ["挑戦(PB)", f"{base_pace:.1f}秒", "理論値に挑むペース"],
            ["限界突破", f"{base_pace*0.97:.1f}秒", "VO2Maxを使い切る"]
        ]
        col_labels3 = ["レベル", "300m設定", "狙い"]
        table3 = ax3.table(cellText=goals, colLabels=col_labels3, loc='center', cellLoc='center', colColours=['#fff9c4']*3)
        table3.scale(1, 2.0)
        table3.auto_set_font_size(False)
        table3.set_fontsize(11)
        if fp:
            for key, cell in table3.get_celld().values(): cell.set_text_props(fontproperties=fp)
    else:
        ax3.text(0.1, 0.5, "算出不能", **font_arg)

    # ④ アドバイス
    ax4 = fig.add_axes([0.52, 0.10, 0.43, 0.35])
    ax4.set_axis_off()
    ax4.set_title("④ AIコーチの戦術アドバイス", fontsize=16, loc='left', color='#0d47a1', weight='bold', **font_arg)
    advice_text = "【分析結果】\n"
    if len(laps) > 0 and potential_sec:
        at_lap = next((i+1 for i in range(1, len(laps)) if laps[i] - laps[i-1] > 3.0), None)
        if at_lap: advice_text += f"● {at_lap}周目でペースが急落しています。\n   ここがスタミナの切れ目(AT値)です。\n"
        else: advice_text += "● 大きなペースダウンがなく、安定しています。\n"
        advice_text += "\n【次の戦術】\n"
        advice_text += "● 「ネガティブ・スプリット」を試そう。\n"
        advice_text += "   最初の2周をあえて1〜2秒落として入ると、\n   後半の粘りが劇的に変わります。\n"
        advice_text += f"● 左の表の『挑戦』ペースで刻む練習が有効です。"
    else:
        advice_text += "データ不足のためアドバイス生成不可。"
    ax4.text(0.02, 0.85, advice_text, fontsize=12, va='top', linespacing=1.6, **font_arg)
    rect4 = plt.Rectangle((0,0), 1, 1, fill=False, edgecolor='#333', transform=ax4.transAxes)
    ax4.add_patch(rect4)

    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches='tight'); return buf

# ---------------------------------------------------------
# 4. メイン画面 (UI)
# ---------------------------------------------------------
st.set_page_config(page_title="持久走分析", layout="wide")
st.title("🏃‍♂️ 持久走データ・サイエンス分析")
st.markdown("記録用紙を撮影してアップロードしてください。AIが君のポテンシャルを科学的に分析します。")

uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner("AIが記録を読み取り、科学的分析を行っています..."):
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        img_byte_arr = io.BytesIO(); image = image.convert('RGB'); image.save(img_byte_arr, format='JPEG')
        
        data, error_msg = run_ai_analysis(img_byte_arr.getvalue())
        
        if data:
            st.success("分析完了！レポートを作成しました。")
            st.image(create_report_image(data), caption="分析レポート（長押しで保存）", use_column_width=True)
        else:
            st.error(f"解析失敗: {error_msg}")
            st.warning("ヒント: Secrets設定でAPIキーが正しいか確認してください。")
