import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, json, os, re
import matplotlib.font_manager as fm
from PIL import Image, ImageOps

# ★ここが新兵器：Google公式のAI操作ツールを読み込む
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# ---------------------------------------------------------
# 1. APIキーの設定（公式SDKに渡す）
# ---------------------------------------------------------
raw_key = st.secrets.get("GEMINI_API_KEY", "")
# キーのクリーニング（改行やゴミを削除）
API_KEY = str(raw_key).replace("\n", "").replace(" ", "").replace("　", "").replace('"', "").replace("'", "").strip()

# 公式ツールにキーを登録（これで接続設定は完了）
if API_KEY:
    genai.configure(api_key=API_KEY)

# ---------------------------------------------------------
# 2. フォント設定（変更なし）
# ---------------------------------------------------------
@st.cache_resource
def load_japanese_font():
    import requests # フォント取得のみrequestsを使用
    font_path = "NotoSansJP-Regular.ttf"
    url = "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansjp/NotoSansJP-Regular.ttf"
    try:
        if not os.path.exists(font_path):
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            with open(font_path, "wb") as f:
                f.write(response.content)
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Noto Sans JP'
        return fm.FontProperties(fname=font_path)
    except:
        return None

# ---------------------------------------------------------
# 3. AI解析エンジン（Google公式SDK使用）
# ---------------------------------------------------------
def run_ai_analysis(image_obj):
    if not API_KEY:
        return None, "APIキーが設定されていません。Secretsを確認してください。"

    # 使用するモデル：安定版の 1.5-flash を指定
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = """
    この「持久走記録用紙」の画像を読み取り、以下のデータを抽出してJSON形式のみで出力してください。
    
    【抽出項目】
    1. "name": 名前（読めなければ "選手"）
    2. "long_run_dist": 上段の距離(m)。数値のみ。
    3. "tt_laps": 下段のラップタイム(秒)の数値リスト。
    
    【厳守】
    JSONデータ以外の文字（```json や解説）は一切書かないでください。
    """

    try:
        # ★公式SDKを使ってAIに命令（URLなどは自動処理される）
        response = model.generate_content(
            [prompt, image_obj],
            generation_config={"response_mime_type": "application/json"}
        )
        
        # 結果の取り出し
        text = response.text
        
        # JSONクリーニング（念のため）
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0)), None
        else:
            # そのままパースを試みる
            return json.loads(text), None

    except google_exceptions.InvalidArgument as e:
        return None, f"API設定エラー: キーが無効か、画像サイズが大きすぎます。({e})"
    except google_exceptions.PermissionDenied:
        return None, "アクセス拒否: APIキーが間違っています。Secretsを再確認してください。"
    except Exception as e:
        return None, f"解析エラー: {str(e)}"

# ---------------------------------------------------------
# 4. レポート作成（可視化）
# ---------------------------------------------------------
def create_report_image(data):
    fp = load_japanese_font()
    font_arg = {'fontproperties': fp} if fp else {}
    
    try: laps = np.array([float(x) for x in data.get("tt_laps", [])])
    except: laps = np.array([])
    try: dist = float(data.get("long_run_dist", 0))
    except: dist = 0.0
    name = data.get("name", "選手")

    target_dist = 3000 if dist > 3200 else 2100
    base_time_min = 15 if target_dist == 3000 else 12
    
    potential_sec = None
    vo2_max = 0
    if dist > 0:
        potential_sec = (base_time_min * 60) * (target_dist / dist)**1.06
        vo2_max = max((dist * (12/base_time_min) - 504.9) / 44.73, 0)

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
        txt += "【AIコーチの評価】\nこのエンジンの性能なら、上記のタイムを出せる\nポテンシャルがあります。"
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
# 5. メイン画面 (UI)
# ---------------------------------------------------------
st.set_page_config(page_title="持久走分析", layout="wide")
st.title("🏃‍♂️ 持久走データ・サイエンス分析")
st.markdown("記録用紙をアップロードしてください。AIがポテンシャルを可視化します。")

uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner("AIが分析中..."):
        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image).convert('RGB')
            # PIL ImageオブジェクトをそのままSDKに渡せるので、BytesIO変換は不要
            
            data, error_msg = run_ai_analysis(image)
            
            if data:
                st.success("分析完了！")
                st.image(create_report_image(data), caption="分析レポート（長押しで保存）", use_column_width=True)
            else:
                st.error(error_msg)
        except Exception as e:
            st.error(f"予期せぬエラー: {e}")
