# 関連研究メモ（2026-11-04）
[本]=ページ本文を取得（要約ツール経由）／[抄]=抄録のみ／[検]=検索要約のみ。A=RL B=黒箱探索 C=閉形式編集

## 方向・重み編集
- Lazaridis+2026 One Neuron Fix Repetition Loops (arxiv.org/abs/2606.13705)[本] Gemma4 E2B等のMLP down_proj列を零化・符号反転・倍率で編集。保持集合で反復46→12/384。ただし1ニューロン編集は汎化が弱く、E2Bは完走率低下、設定により乱択編集と同等。B,C:対照設計の手本。
- Arditi+2024 Refusal Single Direction (arxiv.org/abs/2406.11717)[本] W−r̂r̂ᵀW を出力行列全てに適用。方向は検証集合でKL<0.1等の条件付きで選ぶ。C:原型。
- Sun+2025 ThinkEdit (arxiv.org/abs/2503.22048)[本] 過短思考の方向をo_projの上位4%ヘッドから除去。係数なし、割合のみ調整、過剰編集は劣る。C。
- Huang+2025 Manifold Steering (arxiv.org/abs/2505.22411)[本] 過剰思考は1方向。強度を上げると頭打ち後に悪化。B,C。
- Duan+2026 (arxiv.org/abs/2601.05693)[本] ループは行き詰まりで始まり注意で自己強化。温度上昇は対症療法（Pipis+2025 arxiv.org/abs/2512.12895[抄]）。C。

## RL・方策勾配
- Schulman+2025 LoRA Without Regret (thinkingmachines.ai/blog/lora/)[本] 方策勾配ならrank1でも全更新並み。最適LRは約10倍、MLPで足りる。数学・8Bのみ。A。
- Yu+2025 DAPO (arxiv.org/abs/2503.14476)[本] 切り詰め標本は報酬ノイズ。損失マスクか線形罰則。A。
- verl Rollout Correction (verl.readthedocs.io/en/latest/algo/rollout_corr.html)[本] 生成側と学習側の対数確率差は不安定化要因。比の上限(TIS)で補正。A。

## 勾配なし探索・ノイズ下の最適化
- Qiu+2025 ES at Scale (arxiv.org/abs/2509.24372)[本] 全パラメータ摂動、N=30、σ=0.001、報酬z正規化、数学（3,000トークン上限）でRLに匹敵。B。
- Sarkar+2025 EGGROLL (arxiv.org/abs/2511.16652)[本] 低ランク摂動で大集団。LLMはRWKV-7のみ。B。
- Zhou+2023 (arxiv.org/abs/2310.05166)[抄] ノイズ下EIは現在最良点の不確実性を無視しがち。B。
- Iacobelli+2026 Evo-L2S (arxiv.org/abs/2604.06465)[抄] 正解率×長さの多目的進化探索（探索変数は未確認）。B。

## 強度の落とし穴
- Sun+2026 (arxiv.org/abs/2604.05655)[本] 8B、|α|≲0.8超で反復ループ化。Azizi+2025 (arxiv.org/abs/2507.04742)[抄] KL上界で強度を制約。B,C。

## Qwen公式
- Qwen3.5-0.8B card (huggingface.co/Qwen/Qwen3.5-0.8B)[本] 0.8Bは推奨値でも思考ループに入りやすい。思考時 presence_penalty1.5、0〜2で調整可。副作用は言語混在と性能の僅かな低下。

## 取り入れる価値がある点（根拠の強さ）
1. 強度は非単調。αに上限、またはループ率・KLの制約を置く（強：Manifold, Sun+2026, Lazaridis）。
2. 選択集合外の再評価と、ランダム方向の対照（中：Lazaridis、計画済み）。
3. 方法A：rank1・MLPのみで成立しうる（中：数学・8Bのみ）。切り詰めは損失マスク、サンプラー差はTISで補正（中）。
4. Cの方向選択に副作用の制約（KL等）を入れる（中：Arditi）。
5. Bの目的に長さ・ループ率を併用（弱：抄録のみ）。

## 確認できなかったこと
0.8B級Qwen3.5での査読済みの編集報告。二項ノイズ尤度のBO文献。rank1で16,384上限のRLを行った研究。ブログ元の編集は未査読の記述のみ。RLOO・Dr.GRPOは本文未確認。

## 照合の結果（メイン側で、arXiv の原ページを確認。2026-11-04）
- 実在と題名・著者を確認: 2606.13705（Lazaridis+、"When Can One Neuron Fix Repetition Loops in LLMs?"）、2601.05693（Duan+、"Circular Reasoning"）、2509.24372（Qiu+、"Evolution Strategies at Scale"）、2503.22048（Sun+、"ThinkEdit"）、2604.05655（Sun+、"LLM Reasoning as Trajectories"）。
- 2606.13705 の抄録で確認できた記述: 保持集合で検出された反復が 46/384 → 12/384、E2B は符号反転した1ニューロンで足りる、Qwen3.5 でも疎な編集で反復が減る（探索的）、26B・31B では残る失敗が「自己訂正の非収束（doom looping）」へ移る、汎用ベンチマークに有意な悪化なし。
- **確認できなかった記述:** 「E2B の完走率低下」「設定により乱択編集と同等」（2606.13705）、「|α|≲0.8 を超えると反復ループ化」（2604.05655。抄録は軌跡の幾何と推論の補正・長さ制御を述べるのみ）。本文での確認が済むまで、これらは根拠にしない。
- 各項目の「[本]」は、ページ本文を要約ツール経由で取得した意味で、原文との逐語の照合ではない。
