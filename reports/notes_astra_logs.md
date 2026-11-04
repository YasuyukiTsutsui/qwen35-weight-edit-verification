# Astra ログ調査メモ: 実験8（Qwen3.5-0.8B の数学編集）で層・k・α・方向はどう決まったか

調査日: 2026-11-04。一次資料は github.com/RikiyaT/lm_understand-logs（コミット 4d09ba6d, 2026-10-29）。
取得先: `C:\Users\pop_c\AppData\Local\Temp\claude\18db6e3b7926fe61\b5f4f540-8f99-42bb-8519-a7164493b6b1\scratchpad\astra_logs`（以下 `LOGS/`）。
ブログ本文は別ディレクトリ `...\scratchpad\astra_blog\` に取得。取得物のコード・スクリプトは実行していない（読むだけ。JSON 集計は `python -I` の標準入力スクリプトで、取得ディレクトリを作業ディレクトリにせず実施）。
以下で「ブログの実験8の編集」とは `LOGS/exp8_math/runs/p50c`（50 パスの回）を指す。README の対応表でこの回の最終編集は「層15と19の MLP の、方向を最も強く書く 64 出力列から loop 方向を引く」と書かれている。

## 要点

- 4つの値はどれも、Astra が「系統的な探索の結果として選んだ」というより、**逐次の小さな判断の積み重ね**で決まっている。層15・19 は 1 回の層別分解表と「層19 単独→2 層」への段階的拡張から、k=64 は一度だけ作った固定値（他の k は試していない）、α=6 は 4→6 の一度の引き上げ（5 や 8 は試していない）。
- 方向 u は、Astra が自分で書いた短い合成テキスト 4 組（生産的な続き vs 同じ文の繰り返し）の、残差の平均差。実生成ではなく手書きの文から取っている。
- 最終編集は **W' = W − 6·outer(u, mask64 ⊙ (uᵀW))**。選んだ 64 列について u 方向の成分が (1−6) = −5 倍になる（符号反転した過補正。これは私の計算で、ログの文言ではない）。
- 選択用と確認用の問題集合は分離されている（確認 1,152 問は選択用 352 問と重複 0、重みは固定後に不変）。ただし選択はほぼ同じ 64 問に何度も依存し、確認は **練習問題（GSM8K/MATH の train）**であってベンチマークではない。
- 最終評価（MATH-500 7.9→48.1 など）は思考オン・上限 32,768 トークン。基底モデルの 93% が上限で打ち切られており、編集後も 33% が打ち切られる。上限を変えた最終評価は資料内に確認できなかった。

---

## 1. 層・列数・強さ・方向の決まり方

### 1.1 方向の取り方（残差出力 16/20 の意味）

- 出典の言葉: 「u is a unit repetition direction from FOUR matched productive/repetitive original contexts in fwd4, residual outputs16/20.」
  （`runs/p50c/work_notes/experiment_history.md` 146-147 行、および `work_notes/final_report.md` の "Unit repetition directions ..." の行）
- 「16/20」は hidden_states の添字。ツール仕様は「0 = 埋め込み、l = after layer l」なので、添字 16/20 は **0 始まりの層 15/19 の出力**にあたる。コードでも層 l に対して `hidden_{layer+1}` を読み、`diffH[min(l+1,23)]` を方向に使っている（`turns/t010/answer.md`、`tool_calls.jsonl` 61 行目 = `turns/t068`）。つまりユーザーが書いた「層16・20 の残差出力」は、層15・19 自身が書き込んだ直後の残差を指す（ログのコードからの読み取り）。
- 作り方（`turns/t009`〜`t010`、`t013`）: 8 本のテキスト = 4 組。各組は同じ問題文と同じ前半に対し、(a) 生産的な続き、(b) 同じ一文を繰り返す続き。テキストは Astra 自身が書いた約 100〜150 トークンの合成文（`requests/fwd_4.json.claimed`）。各テキストの末尾 36 トークンの残差を平均し、(b)−(a) を 4 組で平均して単位化したものが u。
- 最初の教師強制（fwd_2）はチャットテンプレートが空の think ブロックを足してしまい、raw テンプレートで取り直した（fwd_4）。方向は fwd_4 由来。
- 実生成との整合: 初回の測定（fwd_2、後にテンプレート不備で取り直し）では、手書き文の対比方向が、実生成の「失敗−成功」平均差とコサイン約 0.73（残差 16/20）で一致した（`turns/t009` 出力）。後の新規 6 組でも、元の方向とのコサインは両サイトで約 0.84（`turns/t067`）。ただし方向そのものは生成文からではなく手書き文から取っている点に注意。

### 1.2 層15・19 を選んだ根拠

- 最初の候補は単一ニューロン: 「Top MLP writers」の最上位が (層19, ニューロン11)（`turns/t010`〜`t011` 出力）。そこで層19 の 1 ニューロン除去 → 層19 の方向除去と進んだ。
- 層別の分解表（`turns/t011` 出力、4 組の対比方向への各層の寄与）で、MLP の寄与が最大の 2 層が層15（0.2634）と層19（0.2497）。次は層21（0.1637）、層14（0.1495）。ブロック全体の増分も層19（0.5304）と層15（0.4720）が上位。
- Astra の言葉:
  - 「test a stronger intervention at layers 15 and 19, where the recorded activations showed substantial amplification.」（`turns/t029/answer.md` 冒頭段落）
  - 「# Hypothesis: repetition is maintained by amplification at both layers 15 and 19, so removing the shared direction at those two sites should have a larger causal effect.」（`turns/t026/answer.md` コード 1 行目）
- 層15 と 19 は、タスク文で示されたフルアテンション層（3, 7, 11, 15, 19, 23）でもある（`turns/t001/prompt.txt`）。Astra がこの点を理由にしたかは確認できなかった。
- 確認の仕方: 実際に生成で比べたのは「層19 のみ α=1」（fwd_7、+6.2±4.3、32 問）と「層15+19 α=1」（fwd_8〜11 など）だけ。層15 のみ・層19 のみは教師強制プローブ（α=4）でのみ比較: 層15 のみ −0.0813、層19 のみ −0.0345 nats/トークン（反復側）で、全列 α=4 の −0.1037 に近い（`work_notes/experiment_history.md` の fwd37/38 行）。最終報告の言葉: 「L15-only and L19-only interventions supported both sites」（`work_notes/final_report.md`）。
- 他の層は生成で一度も試されていない。Astra は層12〜22 全部への適用（t015）、層12〜22 の内容一致版（t034）、層15+19+22（t039）を書いたが、t015 はカーネル再起動の NameError、t034 と t039 は「説明なし」で差し戻されて未実行（`turns/t016`, `t035`, `t040` の受信文）。

### 1.3 列数 k=64 を選んだ根拠

- 64 列の選び方: 層ごとに、score_j = (uᵀW0)_j × 〔4 組の (反復−生産) の MLP 活性差（各テキスト末尾 24 トークン平均）の平均〕_j を計算し、fwd_4 だけで降順に並べて上位 64（`tool_calls.jsonl` 61 行目、`turns/t068/answer.md` 57-58 行）。ホールドアウトの正答は使っていない（Astra の記載: "chosen on original fwd4"）。
- **64 という数を選んだ理由を述べた文は確認できなかった。** コードは `top[:64]` で、同時に出した「上位 8/32/64 が正の質量に占める割合」は 層15: 0.193/0.389/0.498、層19: 0.397/0.498/0.555。他の k（16, 32, 128 など）で編集した生成評価は、ログ全体を検索しても見つからなかった。
- 64 列の事後的な正当化（`turns/t099/answer.md`）: 「The selected 64 neurons account for about 91% and 98% of the measured repetitive-versus-productive writing contrast in layers 15 and 19, respectively, despite retaining only 50% and 40% of the directional weight norm.」
- 別セット（fwd_24, 6 組）で独立に選んだ上位 64 との重なりは、層15 で 55/64、層19 で 36/64（順位相関 0.666 / 0.583）。層19 の 64 列は新しい文脈では安定度が低い。
- 全列 α=4 との比較: 16,384 トークン上限の同一 64 問×2 で、top64 α=4 は 13→47/128（+26.6±4.5）、全列 α=4 は 14→48/128（+26.6±4.6）。空白だけで打ち切られた出力は 2 件 vs 10 件。Astra の言葉: 「The failure analysis favors the narrower candidate so far: the full edit had 10 whitespace cutoffs, versus two for the 64-neuron edit, despite similar accuracy at the same cap.」（`turns/t106/answer.md`）

### 1.4 強さ α=6 を選んだ根拠

- 試した順: 全列で α=1 → 2 → 3（生成）、全列 α=4（教師強制と 16k 生成。32k の生成は起動失敗で未測定）、top64 α=4（教師強制と 16k 生成）、top64 α=6（教師強制と 32k 生成）。α=5, 7, 8 以上は試していない。
- α=6 に上げた理由（`turns/t105/answer.md`）: 「I'll test strength 6 on the same 64-neuron subsets, which increases the targeted intervention while leaving the other MLP columns untouched.」
- 教師強制の裏付け（`turns/t109/answer.md`）: 「On aligned tokens, strength 6 suppresses repetition by 0.103 nats/token, close to the full strength-4 edit, while productive reasoning changes by −0.009.」（私の読み取り: top64 を 6 倍にして、全列 α=4 と同程度の反復抑制を狙った。）
- 生成での採否は 1 回の探索結果（fwd_45: 7→35/64、+43.8±6.6、上限 32,768、64 問×1）で決まり、直後に固定した。「I won't increase the strength further before learning whether this gain generalizes.」（`turns/t110/answer.md`）
- 注意: α の各段階は問題集合・サンプル数・上限が揃っていない（下表）。例えば fwd_27（α=2）と fwd_29（α=3）は問題集合が重複 0 の別の 128 問で、Astra 自身も「These sets differ, not a clean dose comparison.」と記録（`work_notes/experiment_history.md`）。top64 α=4（16k）と top64 α=6（32k）は上限が違うため直接比較できない。

### 1.5 試された候補と測定値（生成パス。ペア比較は同じ `both` パス内）

「fwd_k」は `runs/p50c/requests/fwd_k.*` の番号（Astra のメモの「Pass N」とは数え方が違う）。上限はすべて 32,768（特記以外）。結果は 原本→編集 の正解数、差は編集−原本（パーセント点）±SE。

| fwd | 編集 | 問題集合 | 問題×回 | 結果 |
|---|---|---|---|---|
| 1 | なし（原本） | 診断 96（GSM8K32+MATH64） | 96×1 | 11/96（11.5%）、85 件が上限到達 |
| 5 | 層19 ニューロン11 の列をゼロ | 診断のうち 32 問 | 32×1 | 3→5、+6.2±4.3 |
| 7 | 層19 のみ 全列 α=1 | 同 32 問 | 32×1 | 4→6、+6.2±4.3 |
| 8, 9 | 層15+19 全列 α=1（注） | 同 32 問 | 32×1 | 4→7、+9.4±6.9（8 と 9 は結果が完全一致） |
| 10, 11 | 同上（注） | 同 32 問 | 32×1 | 4→9、+15.6±6.5（10 と 11 は結果が完全一致） |
| 13 | 層15+19 全列 α=1 | 新規 128（GSM64+MATH64） | 128×1 | 11→42/128、+24.2±4.0 |
| 14 | 層15+19 全列 α=1 | 診断から 64 問（GSM32+MATH32） | 64×4 | 27→59/256、+12.5±2.6 |
| 16 | α=2 | 同 64 問 | 64×4 | 29→92/256、+24.6±4.0 |
| 27 | α=2 | 新規 128 | 128×2 | 21→110/256、+34.8±3.5 |
| 29 | α=3 | 別の 128（27 と重複 0） | 128×2 | 21→123/256、+39.8±3.8 |
| 35 | 対照: ノルム一致のランダム直交方向（α=4 相当） | 同 64 問 | 64×2 | 14→16/128、+1.6±3.5 |
| 41 | top64 α=4 | 同 64 問 | 64×2、上限 16,384 | 13→47/128、+26.6±4.5 |
| 42 | 全列 α=4 | 同 64 問 | 64×2、上限 16,384 | 14→48/128、+26.6±4.6 |
| 45 | top64 α=6 | 同 64 問 | 64×1 | 7→35/64、+43.8±6.6 |

（注）fwd_8〜11 は、Astra のメモ上は長く「層19 のみ」と記されていたが、`turns/t043/answer.md` の検証で、当時カーネルに入っていた編集は層15 と 19 のランク1除去（α=1）だったと分かった。Astra は「all paired tests reuse the same 32 diagnostic problems; duplicate outputs are not independent evidence」と記録している（`work_notes/experiment_history.md`）。なお 8≡9、10≡11 の一致の理由はログから確認できなかった。

教師強制プローブ（新規に Astra が書いた 6 組の生産的/反復テキスト＋早期クローズ対照 2 本、計 14 本。1 トークンあたりの対数確率の変化、nats）:

| fwd | 編集 | 生産的 | 反復的 |
|---|---|---|---|
| 25 | 全列 α=1 | +0.0023 | −0.0241 |
| 26 | 全列 α=2 | +0.0034 | −0.0487 |
| 32 | 全列 α=4 | +0.0045 | −0.1037 |
| 34 | ランダム対照（ノルム一致） | +0.0039 | +0.0095 |
| 37 | 層15 のみ α=4 | −0.0006 | −0.0813 |
| 38 | 層19 のみ α=4 | +0.0035 | −0.0345 |
| 39 | top64 α=4 | −0.0047 | −0.0645 |
| 44 | top64 α=6 | −0.0090 | −0.1031 |

出典: `work_notes/experiment_history.md`（"Mean branch deltas"、"Pass26/fwd39"、"Pass29/fwd44" の行）。

そのほか: 注意ヘッド L15h7 と L19h5 の出力を半分にした案は、反復側 −0.025 程度で選択性が低く、生成での検証は起動失敗のまま、採用されなかった（`turns/t061`〜`t064`、`work_notes/final_report.md`: "attention-head alternatives removed"）。

### 1.6 参考: 同じ実験の他の2回

README の対応表より。10 パスの回（p10c）は 層8〜22 の 30 個の出力行列から loop 方向を除去（W' = W − 1.5·u(uᵀW)）、25 パスの回（p25c）は 層15 の注意ヘッド7 を無効化（256 列をゼロ）。どちらも MATH-500 は 12.3% / 14.1% にとどまる（`LOGS/exp8_math/README.md`）。p50c の Astra は L15h7（層15 のヘッド7）を検討して不採用にしているが、p25c も「層15 のヘッド7」を選んでいる（ヘッド番号の添字の規約が同じかは未確認）。

---

## 2. 評価の使い方

- Astra に与えられたのは練習問題（GSM8K train 7,469、MATH train 7,275、AIME 714）だけ。ベンチマークは見ていない（`turns/t001/prompt.txt`、README「It never saw the benchmarks」）。
- 前方パスの予算 50 回。実際のリクエストは 69 回で、19 回は起動失敗などで予算に数えられていない（生成 17、教師強制 2。`runs/p50c/readable/passes.md`）。カウントされた 50 回の内訳は **生成 34・教師強制 16**。3 回の再開（handoff）があり、カーネルが消えて状態を回復し直す場面が複数回ある。
- 時期区分:
  1. **探索（fwd_45 まで、生成 16 パス）**: 層・α・top64・対照の選択。fwd_14, 16, 35, 41, 42, 45 は**同じ 64 問**（重複 64/64）、fwd_5〜11 は 32 問（診断の一部）に何度も依存。fwd_13/27/29 は新規の 128 問だが、27 と 29 は別集合。
  2. **固定と確認（fwd_50 以降、生成 18 パス）**: top64・α=6 の重みを固定し、ハッシュも監査（`work_notes/frozen_edit_audit.json`）。「No weight changes since before pass33」（`experiment_history.md`）。
- 分離の工夫（Astra 側の記録）: 確認用 1,152 問（validation_3 128 + validation_4/final_1/2/3 計 1,024）は、探索で使った 352 問と**重複 0**（`work_notes/submission_evidence.json`）。評価計画は実行前に SHA256 付きで固定（`work_notes/frozen_eval_plan.json`、`experiment_history.md`: "No weight tuning on these results."）。分離の意図を述べた文: 「This prevents later exploratory choices from leaking into the final confirmation set.」（`turns/t100/answer.md`）。
- 限界:
  - 確認は練習問題（train 分割）上で、1 問 1 サンプル、各 64 問バッチが独立の `both` パス。バッチごとの基底正答率は 3.1%〜23.4% とばらつく（`readable/passes.md`）。
  - α=6・top64 の採用判断は探索 1 回（fwd_45、64 問×1、SE 6.6）に基づき、確認用 1,152 問は事後の検証であって選択には使われていない（ログ上）。
  - 実験者側のベンチマーク評価は各ランの終了後に 1 回ずつ（`LOGS/exp8_math/README.md`）。ただし `eval/RESULTS_excerpt.md` には、ブログに載らない別プロンプト版・0/5 パスの行、さらに「ユーザーの依頼で」評価した未提出候補（p05c, p10d）がある。**プロンプトの改版（v2→v5）が評価結果を見て行われたかは確認できなかった。**
  - `eval/RESULTS_excerpt.md` のプロンプト v4・25 パスの行は 14.1 / 25.1 で、p25c の値と同一。理由は資料内で確認できなかった。

---

## 3. 最終評価の条件

| 数値 | 何の評価か | 問題数×サンプル | 条件 | 出典 |
|---|---|---|---|---|
| MATH-500 7.9% → 48.1% | 実験者の最終評価（ベンチマーク） | 500 問×4（2,000 生成） | 思考オン、上限 32,768 新規トークン、T=0.6, top_p=0.95, top_k=20, presence 1.5、seed 0。最後の `\boxed{}` を math-verify で採点（厳密採点）。編集は bfloat16 にして差し替え、vLLM | `eval/{unedited,p50c}/merged.json`、`parts/*/scores.json`（`_protocol`）、`code/evaluate.py` |
| 同: 上限到達の割合 | 同上 | 同上 | 基底 93.0% → 編集 33.05%（平均 30,868 → 22,037 トークン） | `merged.json`（`cut_share`, `mean_tokens`） |
| GSM8K 13.8% → 67.1% | 同上 | 1,319 問×2 | 同上。上限到達 87.5% → 7.2% | 同上 |
| **練習問題 10.6% → 60.4%** | Astra の最終確認（ベンチマークではない） | 1,152 問（GSM8K train 576 + MATH train 576）×1 | 同じサンプラー・上限 32,768、`both` で原本と編集を同じ問題に対して走らせる。内訳は GSM8K 11.6→70.8、MATH 9.5→50.0。差 +49.8点（SE 1.5）、救済 579・悪化 5。原本の 1,038 件（90%）が上限到達で、救済のうち 576 件がそれ | `runs/p50c/final_answer.md`、`work_notes/final_report.md`、`state.json` |
| その他（思考オン、上限 32,768） | 同上の最終評価 | AMC23 16 サンプル、Minerva・OlympiadBench・GPQA 4、IFEval 2、MMLU-Pro（固定 1,000 問）1 | AMC23 0.5→23.4、Minerva 1.5→25.4、OlympiadBench 1.3→21.8、MMLU-Pro 40.3→45.5、GPQA 22.7→36.6、IFEval 21.0→62.3 | `README.md` 対応表、`eval/RESULTS_excerpt.md` |

補足:

- 練習問題の編集後の失敗内訳（確認 1,152 問）: 上限到達 212、答えが間違いで終了 187、ボックスなしで終了 58。空白だけの打ち切りが 51（原本は 0）。正解 696 件はすべて思考を閉じており、正解の中央値は 9,172 トークン（`final_report.md` の BEHAVIOR）。
- 探索で唯一、上限を変えた測定は 16,384（fwd_41/42）で、その条件の基底は 10〜11% 台、編集は 37% 台。最終評価で上限を変えた結果は確認できなかった。
- GPQA の基底値: ブログの図は暫定の 24.5、完了した評価は 22.7（README の注記）。
- `code/evaluate.py` の冒頭コメントは非思考オンの別プロトコルの説明が残っていて、「thinking looped at 0.8B; the evaluation uses the model card's NON-thinking benchmark setting instead」という行もある。しかし p50c の `scores.json` は `protocol: think35`、`thinking: true` で、数値は思考オン条件のもの。

---

## 4. ブログ本文の限界・注意書き

ブログ（https://rikiyat.github.io/blog/weight-space-programming/ の「Experiment 8」節ほか）で、この実験の限界・注意に当たる記述は少ない。

- 設定の明示: 「Sampling: temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5. Qwen's model card notes that this model is prone to thinking loops.」「Astra never sees them」（ベンチマーク）。
- 上限到達率の開示: 表の最後の列「Hit token limit」で、基底 93% → 編集 33.1%（10 パス 87.2%、25 パス 86.4%）。
- 編集の意味づけの限定（Gemma の追加実験の節）: 「Fixing loops, however, is not the same as making a model better at math: it lets the model finish answers it could already work out.」「Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked.」
- 全体の位置づけ: 「very early results」「Our current results do show that capabilities are exciting but still limited, specifically due to inefficiencies in context length and harnesses.」
- 再現に関する記載: 「The edited Qwen3.5-0.8B from Experiment 8 is on Hugging Face. We thank Ahsen Khaliq at Hugging Face for rebuilding it from these logs, re-running its evaluation」（再評価の結果自体はブログ本文に載っていない）。

ブログに書かれていない点（ログ側にだけある）:

- 層・k・α の選び方、α=6 という値、編集式が 6 倍の引き算であること（ブログは "subtracted that direction from 64 output columns in each (131,072 weights)" とだけ書く）。
- 3 回のうち大きな効果が出たのは 50 パスの回だけで、10・25 パスの回は別の編集（上の 1.6）。
- 手書きの 4 組から方向を作ったこと、確認が練習問題上であること、Astra 自身の結論「These are practice results, not unseen benchmark scores; mathematical errors, missing boxes, and some degeneration remain.」（`runs/p50c/final_answer.md`）。
- ブログは「ログに reasoning summaries を含む」と書くが、実験8 の README は「no reasoning summaries were emitted in these runs」とし、代わりに途中（p50c は 29 ターン目）から「EXPLAIN EVERY STEP」で説明を書かせたと述べる。それ以前の判断は説明文がなくコードのコメントだけ。
- 1 回のラン内の運用上の乱れ: 起動失敗 19 回、カーネル再起動、同一結果の重複（fwd_8/9、10/11）、ターン 6 のコンテンツフィルタ拒否、ターン 154 付近の 4.7 時間の停止（README）。

---

## 5. 確認できなかったこと

- k=64 を選んだ理由（他の k を試した形跡はない）。α を 4→6 とし 5 を飛ばした理由（教師強制で全列 α=4 に近づくから、までは読めるが、明示はない）。
- 層15・19 以外（層14, 21, 22 など）を生成で試したか: 試していない（試みは未実行に終わった）。層選びの根拠は 8 本の手書きテキスト由来の分解表。
- Astra の内部推論（思考要約は出力されていない）。ターン 29 より前の判断理由は、コードのコメントと代入された変数名からの推測になる。
- fwd_8/9 と 10/11 が完全一致する理由、v4・25 パスの行が p25c と同一値である理由。
- プロンプト改版（v2→v5）と評価結果の見た順序の関係。
- Hugging Face での再評価の結果（ブログ本文に数値なし）。
- 練習問題の採点器が評価用の `Q.is_correct` と同一か（タスク文は「last \boxed{} answer is graded against the reference answer」）。

## 付録: 主な出典ファイル（`LOGS/exp8_math/` 以下）

- 概要・数値の対応: `README.md`、`eval/RESULTS_excerpt.md`、`eval/{unedited,p50c}/merged.json`
- 最終回答・証拠: `runs/p50c/final_answer.md`、`work_notes/final_report.md`、`work_notes/submission_evidence.json`、`work_notes/experiment_history.md`
- 選ばれた 64 列の番号とハッシュ: `work_notes/frozen_weights_manifest.json`、編集の差分: `runs/p50c/edit/`（適用は `code/apply_edit.py`。今回は実行していない）
- 判断の経緯: `runs/p50c/turns/t009`, `t010`, `t011`（分解表）、`t026`, `t029`（層15・19）、`t068`（top64 の作成）、`t098`〜`t101`、`t105`〜`t110`（α=6）、`readable/passes.md`（全パス一覧）
