# かまいたちの夜×3 ウィンドウOCR

Windows上のゲームウィンドウのクライアント領域をキャプチャし、国立国会図書館の
[NDLOCR-Lite](https://github.com/ndl-lab/ndlocr-lite) で日本語本文を抽出します。
実行環境と依存関係の管理には `uv` を使います。

## セットアップ

PowerShellでこのフォルダを開き、次を実行します。

```powershell
uv sync
```

## 実行

現在表示中の「かまいたちの夜×３」を1回キャプチャしてOCRします。

```powershell
uv run game-ocr
```

タイトルは `かまいたちの夜x3` と `かまいたちの夜×３` のどちらでも一致します。
結果は `output\日時\` に保存されます。

- `capture_raw.png`: クライアント領域の原寸キャプチャ
- `capture.png`: OCRに渡した画像（既定では2倍拡大）
- `text.txt`: 低信頼度の誤認識を除いた本文
- `capture.txt`: NDLOCR-Liteの未加工テキスト
- `capture.json`, `capture.xml`: NDLOCR-Liteの詳細結果

よく使うオプション:

```powershell
# キャプチャだけを確認
uv run game-ocr --capture-only

# 対象ウィンドウを明示
uv run game-ocr --title "かまいたちの夜x3"

# クライアント領域の一部だけをOCR（left,top,right,bottom）
uv run game-ocr --crop "0,180,640,360" --scale 3

# 認識位置を青枠で描いた確認画像も保存
uv run game-ocr --viz

# 低信頼度行も残す（既定の下限は0.5）
uv run game-ocr --min-confidence 0

# ウィンドウ一覧
uv run game-ocr --list-windows
```

画面取得方式の都合上、対象ウィンドウを前面に出してからキャプチャします。
`--no-activate` を指定すると前面化しませんが、他のウィンドウが重なっている場合は
それも写り込みます。

NDLOCR-Liteは国立国会図書館がCC BY 4.0で公開しているソフトウェアです。

## Realtime実況の試作

`OPENAI_API_KEY` を設定すると、`gpt-realtime-2.1-mini` で本文朗読と短い感想を
音声再生できます。音声は24kHz PCMを受信しながら再生し、同時にWAVへ保存します。
NDLOCRの検出・認識モデルは起動時に一度だけ初期化し、その後の全ターンで再利用します。
各画面の本文と感想はRealtimeの既定Conversationへ追加されるため、同じ実行中は
過去の展開を踏まえて感想を生成します。本文朗読だけは逐語性を高めるため履歴外で生成します。
RealtimeセッションはAPI仕様上最大60分で、プログラムを終了・再起動した場合も
サーバー側の会話状態は引き継がれません。

まずは既存のOCR結果などを使い、ゲームを操作せず音声だけ確認します。

```powershell
uv run game-commentary --text-file output\result\text.txt
```

現在のゲーム画面をOCRして1回だけ実況します。既定ではEnterを送りません。

```powershell
uv run game-commentary
```

朗読・感想の再生後にゲームへEnterを1回送る場合だけ、明示的に指定します。

```powershell
uv run game-commentary --press-enter
```

miniを音声生成に使いながら、感想文の自然さを優先して上位モデルに判断させる場合:

```powershell
uv run game-commentary --press-enter --commentary-model gpt-realtime-2.1
```

`--commentary-model` を省略すると、感想文も `--model`（既定では
`gpt-realtime-2.1-mini`）で生成します。

複数画面を進める試験:

```powershell
uv run game-commentary --press-enter --max-turns 3
```

安全策として、OCR本文が空または直前と同一の場合は停止し、Enterを送りません。
朗読音声の転写が本文と一致しない場合もEnterを送りません。試験目的でこの安全策を
外す場合だけ `--allow-narration-mismatch` を指定できます。
各ターンの本文、朗読・感想のWAV、生成音声の転写、照合結果は
`output\commentary_日時\turn_XXX\` に保存されます。

音声を再生せずファイルだけ生成するには `--no-playback`、感想なしの朗読試験には
`--narration-only` を指定します。

感想の長さは固定ではありません。通常の場面は8～35文字程度の自然な一言、伏線の回収・
新事実・重大な選択・事件の進展など重要な場面だけ最大2文・約90文字で話すよう、
Realtime側に判断させます。迷った場合は一言を選びます。
感想はまずテキストと `length_mode`（`quick` / `extended`）、`emotion`、
`intensity`、`pace` の演技情報をJSONで決め、
次の音声応答でその感想を演技付きで読み上げます。感情は `calm`、`amused`、
`excited`、`surprised`、`tense`、`sad`、`thoughtful` から毎ターン自動選択されます。
通常の一言が35文字を超えた場合や、説明口調・不自然な体言止めになった場合は、
音声化する前に最大2回まで自動で書き直します。
ゲーム画面が前の文章を残したまま1行ずつ増える場合は、追加された本文だけを検出して
朗読・感想の対象にします。感想は詩的な短文ではなく、視聴者に話す自然な実況口調で
完結した一文を基本にします。
