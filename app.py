import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, requests, json, re, os, base64, time
import matplotlib.font_manager as fm
from PIL import Image, ImageOps

# ---------------------------------------------------------
# 1. 設定と準備（世界標準の堅牢性）
# ---------------------------------------------------------
API_KEY = st.secrets.get("GEMINI_API_KEY", "")

@st.cache_resource
def load_japanese_font():
    """
    【リスク対策】
    フォント取得失敗を防ぐため、複数の確実なソース（URL）を順番に試す。
    1つ目がダメでも2つ目、3つ目で必ず成功させる「多重防御」仕様。
    """
    font_path = "NotoSansJP-Regular.ttf"
    
    # 優先順位付きのダウンロード元リスト
    # 1. Google Fontsの特定バージョン（リンク切れしない永久固定リンク）
    # 2. GitHubのミラーサイト（予備）
    urls = [
        "https://raw.githubusercontent.com/google/fonts/e3082f4d6d660086395b8d23e5959146522c7a52/ofl/notosansjp/NotoSansJP-Regular.ttf",
        "https://raw.githubusercontent.com/minoryorg/Noto-Sans-JP/master/fonts/NotoSansJP-Regular.ttf"
    ]
    
    # すでに正常なファイルがあれば即リターン
    if os.path.exists(font_path) and os.path.getsize(font_path) > 1000:
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Noto Sans JP'
        return fm.FontProperties(fname=font_path)

    # 順番にダウンロードを試行
    for url in urls:
        try:
            headers = {"User-Agent": "Mozilla/5.0"} # ブラウザのふりをする（ブロック回避）
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                with open(font_path, "wb") as f:
                    f.write(response.content)
                fm.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = 'Noto Sans JP'
                return fm.FontProperties(fname=font_path)
        except Exception:
            continue # 次のURLへ
            
    # 全滅時は警告を出して英語フォントで続行（アプリはクラッシュさせない）
    st.warning("⚠️ 日本語フォントの取得に失敗しました。")
    return None

# ---------------------------------------------------------
# 2. AI解析エンジン（自動圧縮＆リトライ機能付き）
# ---------------------------------------------------------
def run_ai_analysis(image_obj):
    # ① 鍵チェック
    if not API_KEY:
        return None, "APIキーが見つかりません。Secretsの設定を確認してください。"

    # ② 【リスク対策】画像サイズの自動最適化
    # 巨大な画像をそのまま送るとタイムアウトするため、長辺1024pxにリサイズ
    image_obj.thumbnail((1024, 1024))
    img_byte_arr = io.BytesIO()
    image_obj.save(img_byte_arr, format='JPEG', quality=85)
    b64_image = base64.b64encode(img_byte_arr.getvalue()).decode()

    # ③ モデル選定
    target_model = "gemini-1.5-flash" # 基本はこれ
    try:
        # 動的にモデルを探すが、失敗しても基本モデルを使う
        models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
        resp = requests.get(models_url, timeout=5)
        if resp.status_code == 200:
            m_data = resp.json()
            avail = [m['name'].split('/')[-1] for m in m_data.get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
            if avail: target_model = next((m for m in avail if "flash" in m), avail[0])
    except:
        pass # 通信エラーでも、とりあえずデフォルト設定で進む（止まらない設計）

    # ④ プロンプト（AIへの指示書）
    prompt = """
    あなたは陸上競技の専門アナリストです。
    画像から「15分間走(または12分間走)」と「3000m(または2100m)走」の記録を読み取り、JSONデータのみを出力してください。

    【ルール】
    - 必ずJSON形式のみで返すこと。Markdownの装飾(```jsonなど)も不要。
    - 数値は半角数字に変換すること。

    【JSON構造】
    {
      "name": "氏名(読み取れなければ'選手')",
      "long_run_dist": 15分/12分間走の距離(数値のみ, 例: 4050)。空欄なら0,
      "tt_laps": [ラップタイム(秒)の数値リスト]
    }
    """

    url = f"[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/){target_model}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    # ⑤ 【リスク対策】自動リトライ機能（最大3回）
    # 一瞬の通信エラーで諦めず、粘り強く再接続する
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            result = response.json()
            
            if "error" in result:
                # 致命的なエラーならリトライしても無駄なので即終了
                return None, f"AIエラー: {result['error']['message']}"
            
            # 正常なデータが返ってきたかチェック
            if 'candidates' in result and result['candidates']:
                raw_text = result['candidates'][0]['content']['parts'][0]['text']
                # 強力な正規表現でJSONを摘出
                match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if match:
                    return json.loads(match.group(0)), None
            
            # データが空ならリトライへ
            
        except Exception as e:
            if attempt == max_retries - 1: # 最後までダメだったら
                return None, f"システムエラー: {str(e)}"
            time.sleep(1) # 1秒待って再挑戦

    return None, "AIからの応答がありませんでした。画像が鮮明か確認してください。"

# ---------------------------------------------------------
# 3. レポート描画エンジン
# ---------------------------------------------------------
def create_report_image(data):
    fp = load_japanese_font()
    font_arg = {'fontproperties': fp} if fp else {}
    
    # データ安全読み込み
    try: laps = np.array([float(x) for x in data.get("tt_laps", [])])
    except: laps = np.array([])
    try: dist = float(data.get("long_run_dist", 0))
    except: dist = 0.0
    name = data.get("name", "選手")

    # 距離に応じたコース推定
    target_dist = 3000 if dist > 3200 else 2100
    base_time_min = 15 if target_dist == 3000 else 12

    # ポテンシャル計算
    if dist > 0:
        potential_sec = (base_time_min * 60) * (target_dist / dist)**1.06
        vo2_max = max((dist * (12/base_time_min) - 504.9) / 44.73, 0)
    else:
        potential_sec = None
        vo2_max = 0

    # A4横サイズのキャンバス
    fig = plt.figure(figsize=(11.69, 8.27), facecolor='white', dpi=100)
    
    # ヘッダー
    fig.text(0.05, 0.94, "持久走 科学的分析レポート", fontsize=24, weight='bold', color='#1a237e', **font_arg)
    fig.text(0.05, 0.90, f"氏名: {name}  |  基準: {base_time_min}分間走 {int(dist)}m", fontsize=14, color='#333', **font_arg)

    # ① 生理学的ポテンシャル
    ax1 = fig.add_axes([0.05, 0.55, 0.42, 0.30]); ax1.set_axis_off()
    ax1.set_title("① 生理学的ポテンシャル", fontsize=16, loc='left', color='#0d47a1', weight='bold', **font_arg)
    
    txt = f"■ 推定VO2Max: {vo2_max:.1f} ml/kg/min\n"
    if potential_sec:
        m, s = divmod(potential_sec, 60)
        txt += f"■ {target_dist}m 理論限界タイム: {int(m)}分{int(s):02d}秒\n\n"
        txt += "【AIコーチの評価】\nこのエンジンの性能なら、上記のタイムで走れる\n潜在能力を持っています。自信を持ちましょう！"
    else:
        txt += "※基準記録が不足しています。"
    ax1.text(0.02, 0.85, txt, fontsize=12, va='top', linespacing=1.8, **font_arg)
    ax1.add_patch(plt.Rectangle((0,0), 1, 1, fill=False, edgecolor='#ddd', transform=ax1.transAxes))

    # ② ラップ表
    ax2 = fig.add_axes([0.52, 0.55, 0.43, 0.30]); ax2.set_axis_off()
    ax2.set_title("② 周回精密データ", fontsize=16, loc='left', color='#0d47a1', weight='bold', **font_arg)
    if len(laps) > 0:
        rows = []
        for i, l in enumerate(laps[:10]):
            diff = l - laps[i-1] if i > 0 else 0
            mark = "▼DN" if diff >= 2.0 else ("▲UP" if diff <= -1.5 else "―")
            rows.append([f"{i+1}周", f"{l:.1f}s", mark])
        tab = ax2.table(cellText=rows, colLabels=["周回", "ラップ", "傾向"], loc='center', cellLoc='center')
        tab.scale(1, 1.4)
        if fp:
            for key, cell in tab.get_celld().values(): cell.set_text_props(fontproperties=fp)
    else:
        ax2.text(0.1, 0.5, "データなし", **font_arg)

    # ③ 目標設定
    ax3 = fig.add_axes([0.05, 0.10, 0.42, 0.35]); ax3.set_axis_off()
    ax3.set_title("③ 次回の目標設定", fontsize=16, loc='left', color='#0d47a1', weight='bold', **font_arg)
    if potential_sec:
        pace = potential_sec / (target_dist / 300)
        data3 = [
            ["現状維持", f"{pace*1.05:.1f}s", "今の走り"],
            ["挑戦(PB)", f"{pace:.1f}s", "理論値"],
            ["限界突破", f"{pace*0.97:.1f}s", "最大能力"]
        ]
        tab3 = ax3.table(cellText=data3, colLabels=["レベル", "300m設定", "狙い"], loc='center', cellLoc='center', colColours=['#fff9c4']*3)
        tab3.scale(1, 2.0); tab3.auto_set_font_size(False); tab3.set_fontsize(11)
        if fp:
            for key, cell in tab3.get_celld().values(): cell.set_text_props(fontproperties=fp)
    else:
        ax3.text(0.1, 0.5, "算出不能", **font_arg)

    # ④ アドバイス
    ax4 = fig.add_axes([0.52, 0.10, 0.43, 0.35]); ax4.set_axis_off()
    ax4.set_title("④ 戦術アドバイス", fontsize=16, loc='left', color='#0d47a1', weight='bold', **font_arg)
    adv = "【分析結果】\n"
    if len(laps) > 0 and potential_sec:
        at_lap = next((i+1 for i in range(1, len(laps)) if laps[i] - laps[i-1] > 3.0), None)
        if at_lap: adv += f"● {at_lap}周目でペースダウンしています。\n   ここがスタミナの切れ目(AT値)です。\n"
        else: adv += "● 全体を通して安定したペース配分です。\n"
        adv += "\n【次の戦術】\n● 「ネガティブ・スプリット」推奨。\n   前半を1〜2秒抑えて、後半に上げる走りです。"
    else:
        adv += "データ不足のため分析できません。"
    ax4.text(0.02, 0.85, adv, fontsize=12, va='top', linespacing=1.6, **font_arg)
    ax4.add_patch(plt.Rectangle((0,0), 1, 1, fill=False, edgecolor='#333', transform=ax4.transAxes))

    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches='tight'); return buf

# ---------------------------------------------------------
# 4. メインUI
# ---------------------------------------------------------
st.set_page_config(page_title="持久走分析", layout="wide")
st.title("🏃‍♂️ 持久走データ・サイエンス分析")
st.markdown("記録用紙をアップロードしてください。AIがポテンシャルを可視化します。")

uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner("AIが画像を解析中... (数秒お待ちください)"):
        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image).convert('RGB')
            
            data, error_msg = run_ai_analysis(image)
            
            if data:
                st.success("分析完了！")
                st.image(create_report_image(data), caption="長押しで保存", use_column_width=True)
            else:
                st.error(error_msg)
        except Exception as e:
            st.error(f"予期せぬエラー: {e}")
