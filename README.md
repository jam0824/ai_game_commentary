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

`OPENAI_API_KEY` を設定すると、`gpt-realtime-2.1-mini` で本文朗読と実況音声を
再生できます。実況内容の判断には既定で `gpt-realtime-2.1` を使います。
音声は24kHz PCMを受信しながら再生し、同時にWAVへ保存します。
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

既定では、音声生成にmini、実況内容の判断に上位モデルを使います。

```powershell
uv run game-commentary --press-enter --commentary-model gpt-realtime-2.1
```

`--commentary-model` の既定値は `gpt-realtime-2.1` です。感想判断もminiで
試す場合だけ `--commentary-model gpt-realtime-2.1-mini` を指定します。

複数画面を進める試験:

```powershell
uv run game-commentary --press-enter --max-turns 3
```

安全策として、OCR本文が空または直前と同一の場合は停止し、Enterを送りません。
OCRが空白位置だけを変えた場合も同一本文として停止します。
朗読音声の転写が本文と一致しない場合もEnterを送りません。試験目的でこの安全策を
外す場合だけ `--allow-narration-mismatch` を指定できます。
各ターンの本文、朗読・感想のWAV、生成音声の転写、照合結果は
`output\commentary_日時\turn_XXX\` に保存されます。

音声を再生せずファイルだけ生成するには `--no-playback`、感想なしの朗読試験には
`--narration-only` を指定します。

実況は毎ターン必要とは限りません。Realtime側が次の4モードから判断します。

- `silent`: 感想なし。実況音声も生成しません。
- `reaction`: 「うわっ！」「えっ、待って！」など1～12文字の反射的な反応。
- `quick`: 通常は8～35文字の短い感想・ツッコミ。ページ末では2～3文に広げます。
- `extended`: 重大な展開だけ、最大2文・約90文字のしっかりした感想。

OCRの前に文字送りの三角マークまたは本マークが表示されるまで待つため、
表示途中の短い断片を読み上げません。三角マークは通常の細かい文字送りとして、
突然の展開に対する `reaction` または重大な発見への `extended` だけを許可し、
それ以外は `silent` にします。本マークはページ終端として扱い、そのページ内で
まだ一度も実況していなければ、ページ全体を踏まえた実況を必ず一度入れます。
すでに実況しているページでは、本マークでも新たに話す価値がなければ `silent` にできます。
本マークで発話するときは一言だけで切らず、ページ全体を踏まえた短めの2～3文・
合計28～90文字にします。
モードと本文に矛盾する出力や文字数超過、これらのタイミング規則への違反、
説明口調・不自然な体言止めは、音声化する前に最大3回まで自動で書き直します。
それでも条件を満たさない場合は安全のためEnterを送りません。
マークを12秒以内に検出できない場合は実況を終了せず、自動的に待機をやり直します。
再試行回数は既定で無制限です。停止したい場合は `Ctrl+C`、回数を制限したい場合は
たとえば `--marker-retries 3` を指定します。タイムアウト時の最新画面は
`turn_XXX\marker_timeout_latest.png` に保存されます。
発話する場合は `emotion`、`intensity`、`pace` も自動選択し、次の音声応答で
演技付きで読み上げます。口調は友達と一緒に遊ぶ20代くらいの女性のような、
明るく親しみやすいカジュアルさを基本にし、「へぇ」「あ、なるほど」
「〜だよね」「〜かも」「〜じゃない？」「〜よねー」などを場面に応じて使います。
「〜だな」「〜だろ」のようなぶっきらぼうな語尾は自動で書き直します。
同じ相づちや語尾は連続させず、過剰なネットスラングは使いません。
配信映えするよう、感情強度にはモード別の最低値を設けています。`reaction` は0.85、
`quick` は0.55、`extended` は0.70以上とし、声量だけでなく音程差、抑揚、間、
テンポの変化を普段の会話より大きくします。
ゲーム画面が前の文章を残したまま1行ずつ増える場合は、追加された本文だけを検出して
朗読・感想の対象にします。感想は詩的な短文ではなく、視聴者に話す自然な実況口調にします。
