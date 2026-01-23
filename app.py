import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, json, os, re
import matplotlib.font_manager as fm
from PIL import Image, ImageOps
import google.generativeai as genai

# ---------------------------------------------------------
# 1. APIキーの設定
# ---------------------------------------------------------
st.set_page_config(page_title="持久走分析", layout="wide")
st.title("🏃‍♂️ 持久走データ・サイエンス分析")

raw_key = st.secrets.get("GEMINI_API_KEY", "")
API_KEY = str(raw_key).replace("\n", "").replace(" ", "").replace("　", "").replace('"', "").replace("'", "").strip()

if not API_KEY:
    st.error("【重要】SecretsにAPIキーが設定されていません。")
    st.stop()

# APIを設定
genai.configure(api_key=API_KEY)

# ---------------------------------------------------------
# 2. フォント設定
# ---------------------------------------------------------
@st.cache_resource
def load_japanese_font():
    import requests
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
# 3. モデル自動探索ロジック（ここが生命線）
# ---------------------------------------------------------
def find_working_model():
    """
    Googleに問い合わせて、'generateContent'（画像解析）が可能なモデルを探し出す。
    名前を決め打ちせず、リストにあるものを必ず使う。
    """
    try:
        # アカウントで利用可能な全モデルを取得
        all_models = list(genai.list_models())
        
        # 画像認識(generateContent)ができるモデルだけを抽出
        valid_models = []
        for m in all_models:
            if 'generateContent' in m.supported_generation_methods:
                valid_models.append(m)
        
        if not valid_models:
            st.error("エラー: このAPIキーで利用可能なAIモデルが1つも見つかりませんでした。Google AI Studioで有効なモデルがあるか確認してください。")
            return None

        # 優先順位: 1.5-flash -> 1.5-pro -> その他
        # リストの中からベストなものを探すが、なければリストの先頭を使う
        best_model = None
        
        # 1. Flashを探す
        for m in valid_models:
            if "flash" in m.name.lower():
                best_model = m
                break
        
        # 2. なければProを探す
        if not best_model:
            for m in valid_models:
                if "pro" in m.name.lower() and "vision" not in m.name.lower(): # vision専用を除く
                    best_model = m
                    break
        
        # 3. それもなければ、とにかくリストの最初にあるやつを使う（意地でも動かす）
        if not best_model:
            best_model = valid_models[0]
            
        return best_model.name

    except Exception as e:
        st.error(f"モデルリストの取得に失敗しました: {e}")
        return None

# ---------------------------------------------------------
# 4. メイン処理
# ---------------------------------------------------------

# ★アプリ起動時にモデルを探す
target_model_name = find_working_model()

if target_model_name:
    # 画面にこっそり「使用中のモデル」を表示（デバッグ用・安心材料）
    st.caption(f"✅ 接続成功: AIモデル `{target_model_name}` を使用して解析します")
else:
    st.stop() # モデルがない場合はここで停止

st.markdown("記録用紙をアップロードしてください。AIがポテンシャルを可視化します。")
uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner("AIが解析中..."):
        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image).convert('RGB')
            
            # モデルの準備
            model = genai.GenerativeModel(target_model_name)
            
            prompt = """
            この「持久走記録用紙」の画像を読み取り、以下のデータを抽出してJSON形式のみで出力してください。
            【抽出項目】
            1. "name": 名前（読めなければ "選手"）
            2. "long_run_dist": 上段の距離(m)。数値のみ。
            3. "tt_laps": 下段のラップタイム(秒)の数値リスト。
            【厳守】JSONデータ以外の文字は一切書かないでください。
            """

            # 実行
            response = model.generate_content(
                [prompt, image],
                generation_config={"response_mime_type": "application/json"}
            )
            
            # 結果処理
            try:
                data = json.loads(response.text)
                
                # --- 以下、レポート描画 ---
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
                fig.text(0.05, 0.94, "持久走 科学的分析レポート", fontsize=24, weight='bold', color='#1a237e', **font_arg)
                fig.text(0.05, 0.90, f"氏名: {name}  |  基準: {base_time_min}分間走 {int(dist)}m", fontsize=14, color='#333', **font_arg)

                # ①
                ax1 = fig.add_axes([0.05, 0.55, 0.42, 0.30]); ax1.set_axis_off()
                ax1.set_title("① 生理学的ポテンシャル", fontsize=16, loc='left', color='#0d47a1', weight='bold', **font_arg)
                txt = f"■ 推定VO2Max: {vo2_max:.1f} ml/kg/min\n"
                if potential_sec:
                    m, s = divmod(potential_sec, 60)
                    txt += f"■ {target_dist}m 理論限界タイム: {int(m)}分{int(s):02d}秒\n\n"
                    txt += "【AIコーチの評価】\n今のエンジン性能なら、上記のタイムを出せる\nポテンシャルがあります。"
                else:
                    txt += "※基準記録が不足しています。"
                ax1.text(0.02, 0.85, txt, fontsize=12, va='top', linespacing=1.8, **font_arg)
                ax1.add_patch(plt.Rectangle((0,0), 1, 1, fill=False, edgecolor='#ddd', transform=ax1.transAxes))

                # ②
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

                # ③
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

                # ④
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

                buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches='tight')
                
                st.success("分析完了！")
                st.image(buf, caption="分析レポート（長押しで保存）", use_column_width=True)

            except json.JSONDecodeError:
                st.error("AIからのデータを解析できませんでした。別の画像を試してください。")
        
        except Exception as e:
            st.error(f"システムエラー: {e}")
            st.warning("ヒント: SecretsのAPIキーが正しいか、Google AI Studioで有効なモデルがあるか確認してください。")
