def run_analysis(image):
    model = genai.GenerativeModel("gemini-1.0-pro-vision")

    prompt = """
あなたは陸上長距離のデータ分析官です。
以下の指示は【絶対に】守ってください。

【最重要ルール】
- 出力はJSONのみ
- 説明文・前置き・後書きは禁止
- ``` や ```json も使用禁止
- JSONの外に1文字でも出力したら失敗です

【JSONスキーマ】
{
  "name": "string",
  "record_type_minutes": number,
  "race_category": "time",
  "records": [
    {
      "attempt": number,
      "total_dist": number,
      "total_time_str": "mm:ss",
      "laps": [number]
    }
  ],
  "coach_advice": "string"
}

【内容ルール】
- ラップタイムは全て抽出
- 数値は半角
- laps は秒単位の数値配列
- 読み取れない項目は推測せず 0 または空配列
- coach_advice は前向きで励ます内容
"""

    try:
        response = model.generate_content(
            [prompt, image],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        )

        data = safe_json_load(response.text)

        if data is None:
            return empty_result(), "JSON解析に失敗しました"

        return data, None

    except Exception as e:
        return empty_result(), f"解析エラー: {str(e)}"
