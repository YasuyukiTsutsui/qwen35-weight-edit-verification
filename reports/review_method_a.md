# 方法A（`scripts/train_rank1_pg.py`）独立レビュー

作成: 2026-11-04（独立レビュー担当）。対象: `scripts/train_rank1_pg.py`（251 行）、`scripts/eval_math.py` の `--save-tokens` 部分（L52-55, L63-72, L83-87）、設計意図 `reports/plan_v2_after_review.md` 表 #1〜#4。
**読み取り専用で実施。** プロジェクト内で作成・変更したファイルは本ファイルのみ。GPU・Docker・モデルの読み込み・探索プロセスへの接触はなし。ネットから取得した vLLM / PyTorch / transformers のソースは、スクラッチパッド（プロジェクト外）に保存して読んだだけで、実行していない。実行した計算は、ホストの `python -I` による小さな numpy 計算（重み 1 枚 7 MB 程度×2、`directions.pt` 98 KB）のみ。

「机上」は、コードとソースの読解・手計算による判断で、実機では測っていないことを指す。

## 0. 要約

| ID | 重大度 | 題 | 場所 |
|---|---|---|---|
| A-1 | 実行を壊す（黙って NaN の重みを書き出し得る） | 目標トークンが top-p で外れた行で `logsumexp` の逆伝播が NaN になり、全パラメータへ伝播する | `train_rank1_pg.py:55-56`, `:234-243` |
| A-2 | 実行を止める（OOM、机上） | 活性化チェックポイントがなく、16 GB GPU では 16,384 トークン系列（既定の 8,192 でも）の逆伝播が入らない見込み | `:174-178`, `:151` |
| A-3 | 結果を誤らせる | 初期編集（α=12）の相対変化 0.235/0.280 が上限 0.12 を超え、1 ステップ目の射影で編集が約半分になる | `:9-10`, `:66-77`, `:242` |
| A-4 | 結果を誤らせる | `--max-len` 既定 8,192 が標本の上限 16,384 より短い。報酬は全文で決まるのに、勾配は前半だけ。ループは末尾に出る | `:148-154`, `:167` |
| A-5 | 結果の解釈を誤らせる | 既定の `--steps 4`・単一パスでは、Adam の更新が各要素で高々 4×lr。学習ではなく「射影による縮小」が結果を決める | `:165-168`, `:216-217`, `:222-244` |
| A-6 | 軽微〜中 | 同じ標本を複数ステップで使い、重要度比がない（オフポリシー）。HF と vLLM の数値差の大きさも未測定 | `:222-244` |
| A-7 | 軽微 | 混合報酬の問題が 0 件でも黙って「学習済み」を書き出す。書き出しで 2 層の編集を検証しない。途中保存がない | `:214-215`, `:224-225`, `:111-133`, `:245-247` |
| A-8 | 軽微 | `check` モードの対象が最短の n 本だけで、データの出所（どの重みから標本を取ったか）の検証がない | `:185-192` |
| A-9 | 軽微 | 文書・既定値の不一致（`--rounds`、1 バッチ 6 問 vs 計画の 12 問以上、など） | `:13`, `:166` |
| A-10 | 軽微 | 同順位・境界の扱いが vLLM の Triton 実装と完全には同じでない（影響は小さい） | `:50-53` |
| A-11 | 軽微 | 使われない softmax がグラフに残る／`torch.utils.checkpoint` の明示 import がない | `:51`, `:61` |

重大な指摘（実行を壊す／止める／結果を誤らせる）は A-1〜A-5 の 5 件。

## 1. 確認できて、計画どおりに一致していた項目（根拠つき）

### 1.1 サンプラー整合（依頼項目 1）

Docker のベースは `vllm/vllm-openai:v0.32.0`（`docker/Dockerfile:1`）。以降の URL はすべてタグ `v0.32.0`（`https://github.com/vllm-project/vllm/blob/v0.32.0/…`）。行番号は取得したファイルのもの。

v0.32.0 では、Triton があり、未対応機能（外部ランチャ上の PP、一部の投機的デコードなど）を使わない場合、**既定が Model Runner V2** になる（`vllm/config/vllm.py:720-739`）。V2 は `vllm/v1/worker/gpu/sample/sampler.py` を使い、従来の `vllm/v1/sample/sampler.py`（V1）とはファイルが別である。**コンテナ内でどちらが動いたかは確認できていない**（`VLLM_LOGGING_LEVEL=WARNING` のため `results/logs/` に記録がない）が、両方を読み、処理順と意味が同じであることを確認した。

| 論点 | `sampler_logprobs` の実装 | vLLM の実際 | 一致 | 根拠 |
|---|---|---|---|---|
| 処理順 | presence → 温度 → top-k → top-p | V2: logit_bias → penalties → bad_words → (thinking budget) → 温度 → min_p → top-k/top-p。V1: penalties → 温度 → argmax 不変プロセッサ(min_p) → top-k/top-p | 一致 | V2 `sampler.py:71-76`（プロセッサ列）, `:254-276`（`apply_sampling_params`）, `:345`。V1 `sampler.py:26-52`（docstring）, `:278-294`, `:406-407` |
| presence penalty の対象 | 応答（出力）トークンのみ。プロンプトは含まない | **出力トークンのみ**。プロンプトの集合は repetition penalty にだけ使う | 一致 | V2 `penalties.py:172-179`（`pres_penalty * output_bin_mask`）、プロンプトは `prompt_bin_mask` で `use_rep_penalty` の分岐内のみ（`:160-174`）。出力カウントは `block >= prompt_len` のトークンのみ（`:253-265`）。V1 `vllm/model_executor/layers/utils.py:63-80`（`logits -= presence_penalties * output_mask`, プロンプトは `prompt_mask` を repetition にだけ使用） |
| 「その位置より前に出たか」 | `first < tpos` | 各ステップで、それまでの出力に出たトークンにだけ減算（`output_bin_counts` はサンプル後に更新） | 一致（更新カーネルは未読。呼び出し位置のみ確認: `vllm/v1/worker/gpu/model_runner.py:1688-1696`） | 同左 |
| 温度の位置 | penalty の後に `/TEMP` | penalty の後、top-k/p の前。fp32 | 一致 | V2 `sampler.py:262-265`, `:242`（fp32 コピー）|
| top-k → top-p の順 | top-k（20）してから、その 20 個の softmax に対して top-p | 「top-k を先に（ロジット値で）、残った k 個に top-p（確率で）」 | 一致 | `vllm/v1/sample/ops/topk_topp_triton.py:1619-1622`（docstring）。参照実装 `topk_topp_sampler.py:494-511` も top-k のマスク後に softmax して top-p |
| top-p の境界 | `(cumsum - p) < 0.95` を残す（直前までの累積が 0.95 未満） | 質量が p 以上になる最小集合。ネイティブ実装は昇順の累積が `<= 1-p` を除外（同値）。Triton は `p_pivots_sum >= p` かつ最小要素を除くと `< p` | 一致（浮動小数の誤差を除く） | `topk_topp_sampler.py:504-511`, `topk_topp_triton.py:573-575` |
| `processed_logprobs` の中身 | top-k/top-p 後のロジットでの log_softmax | **penalty・温度・top-k・top-p を全部適用した後のロジット**の log_softmax。マスクされた要素は −inf。サンプルされたトークンの値だけを返す（`logprobs=0`） | 一致 | V2 `sampler.py:171-186`（処理済みロジットを渡す）, `:290-300`, `:345`、`logprob.py:28-58, 84-108`（全語彙で max と logsumexp を取る）。V1 `topk_topp_sampler.py:274-284`（`logits.log_softmax` を top-k/p 後に計算）、`sampler.py:103-105`。定義の文書 `vllm/config/model.py:287-295` |
| FlashInfer 経路に落ちないか | — | `processed_*` モードでは FlashInfer を使わない（V1 は `forward_native` に固定、V2 も `fused_sampler_eligible` が偽） | 一致 | V1 `topk_topp_sampler.py:220-226`, V2 `sampler.py:307-312` |
| 子サンプルの種 | — | `n>1` で seed 指定時、子ごとに `seed+index`（k 番目が同一軌跡にならない） | 問題なし | `vllm/v1/engine/parallel_sampling.py:58-81` |
| 既定の暗黙ペナルティ | 0 | `SamplingParams` を明示して渡すので `generation_config` の既定値は混ざらない。モデルフォルダに `generation_config.json` なし（`models/Qwen3.5-0.8B` の一覧で確認）。repetition=1.0、frequency=0 | 問題なし | — |

したがって、**処理順・presence の対象・top-k→top-p・processed_logprobs の意味はいずれも vLLM と一致**している。残る差は A-1（NaN）と A-10（同順位）。

### 1.2 RLOO・符号・境界ケース（依頼項目 2）

- 優位度 `adv = r_i − mean_{j≠i} r_j`（`:232`）は定義どおり。`len(g)≥2` の問題だけが `mixed` に入る（`:214`）ので、`len(g)-1` が 0 になることはない。
- 符号: `loss = −adv·Σlogp/(nseq·C)`（`:235`）を最小化 → 正の優位度の標本の対数確率が上がる。方策勾配として正しい。編集の符号（ΔW = −u vᵀ）とは独立。
- 全標本が同じ報酬の問題: RLOO では優位度が厳密に 0 なので、除外しても勾配の和は変わらない（バイアスなし）。分母 `nseq` が混合問題の標本数だけになるのでスケールは変わるが、Adam はスケール不変（A-5 の eps の注を除く）。
- 長さ正規化なし・定数 C（`:235`）。系列の対数確率の和に対する REINFORCE で、偏りはない。ただし長い（16,384 で打ち切られた＝ほぼ失敗）系列が勾配の項数の大半を占める。分散の話で、誤りではない。
- 標本数 1 の場合は A-7。

### 1.3 編集の適用（hook）と書き出しの一致（依頼項目 3）

- 式: `W' = W − u vᵀ`（列は `S`）。`W` は `(1024, 3584)` の `nn.Linear` 重みで、`out = x Wᵀ`。よって `out' = out − (x_S·v) u`。hook（`:89`）はこの式そのもの。書き出し（`:127`）は `W[:, S] -= outer(u, v)` で、行=出力 1024、列=入力 3584。転置・列集合・符号とも一致。
- 初期値 `v_j = α·(u·W_j)`（`:76`）は `make_edit.py:76` の `W[:,S] = WS − α·outer(u, u@WS)` と同じ（`directions.pt` は float32 の (24,1024)、層 15・19 のノルムは 1.0 を numpy で確認。`top_neurons` は 256 個で重複なし）。
- bf16 丸め: 実際の層 15・19 の重みで、α=6 の ΔW を作り、`W − ΔW` を bf16 に丸めて書き出した場合の実効的な差分を numpy で計算した（最近接偶数丸め）。意図した ΔW との差は ‖Δ実効−Δ意図‖/‖Δ意図‖ = 0.23%（層 15）/ 0.21%（層 19）、出力差分のコサイン類似度 0.999997。S 以外の列は変化なし。**hook と書き出しの数学的な不一致は無視できる**。
- キー名・dtype: `model.language_model.layers.15.mlp.down_proj.weight` と `…19…`、dtype は BF16（safetensors ヘッダで確認）。正規表現 `layers\.{L}\.mlp\.down_proj\.weight$` はこれに一致し、`mtp.*` は除外される（`mtp.layers.0.mlp…` は L=15,19 に一致しないので影響なし）。出力ファイル名は元のシャード名のまま、`index.json` はコピーされる。
- ただし、書き出しの検証が弱い点は A-7。

### 1.4 AdamW・クリップ・射影（依頼項目 4）

- パラメータ群: v 群 lr 2e-3、u 群 lr 2e-4、`betas=(0.9, 0.99)`、`weight_decay=0.0`（`:216-217`）。weight decay を使っていない。eps は既定の 1e-8。
- u, v は fp32 の `Parameter`（`:67-77`）。モデルの凍結後に作られるので `requires_grad=True`。
- クリップ: 4 パラメータの全体ノルムを 1.0 に（`:240`）。ただし非有限のガードがない（A-1 の修正案に含めた）。
- 射影: ‖v‖/‖W‖_F ≤ 0.12（`:103-107`）。`Wnorm` は W 全体のフロベニウスノルム（`:77`）で、`make_edit.py` の `rel_change` と同じ定義。‖ΔW‖_F = ‖u‖‖v‖ = ‖v‖（u は単位ノルム）なので正しい。u は毎回単位化（`:102`）。重みのノルムは層 15 で 16.443、層 19 で 17.255（numpy で確認）。よって上限は ‖v‖ ≤ 1.97 / 2.07。
- 実装は計画 #2（射影・weight decay なし）どおり。問題は初期値との整合（A-3）。

### 1.5 トークン境界・再トークン化・テンプレート（依頼項目 6 の一部）

- 再トークン化なし: `prompt_ids` と `token_ids` は vLLM の ID をそのまま保存し（`eval_math.py:86-87`）、学習側もそのまま連結（`:148`）。計画 #4 を満たす。
- 位置の対応: `h[P-1 : P-1+n_resp]` が `y = x[P:]` を予測（`:152-154`）。添字の算術は正しい。
- テンプレート: 生成プロンプトは `<|im_start|>assistant\n<think>\n` で終わり、`tokenizer_config.json` は `add_bos_token: false`、`bos_token: null`。BOS の二重付与はなし。応答の最後が `<|im_end|>`（EOS）の標本では、EOS の対数確率も含む（`token_ids` に EOS が入る前提。`eval_math.py:85` で長さ不一致なら生成後に IndexError になるので、黙って誤る形ではない）。
- `position_ids`: テキストのみなら HF 側の既定（3 軸を同一に展開）で足りる（`modeling_qwen3_5.py` L1234-1263、main ブランチ）。

## 2. 指摘

### A-1（実行を壊す）目標トークンが top-p で外れた行の `logsumexp` 逆伝播が NaN になる

- **場所**: `scripts/train_rank1_pg.py:55-56`（`hit` と `logsumexp`）、`:234-243`（`lp[ok]` と `clip_grad_norm_`）。
- **問題**: HF 側の再計算で、サンプルされたトークン y が top-20 には入るが top-p で除かれた場合、`lp.masked_fill(~hit, -inf)` の行は 20 要素すべて −inf になる。前方は `−inf` を返す（`ok` で除外される）が、**後方は `grad * (self − result).exp()`、つまり `0 * exp(−inf − (−inf)) = 0 * NaN = NaN`**。`masked_fill` の逆伝播は `~hit` の位置を 0 にするが、`hit` の位置の NaN は残り、`log_softmax` の逆伝播（`grad − exp(out)·sum(grad)`）で行全体が NaN になり、`topk` → `lm_head` → 全層へ流れる。結果として u, v の勾配が NaN → `clip_grad_norm_` のノルムが NaN（`error_if_nonfinite` が既定で False）→ AdamW の更新で u, v が NaN → `export` が NaN の重みを書き出す。y がそもそも top-20 に入らない場合は `hit` が全て偽で、`masked_fill` の逆伝播が全要素を 0 にするので NaN にならない。危険なのは「top-20 には入るが top-p で外れる」境界上の場合だけである。
- **なぜ起きるか**: vLLM は y をサンプルした時点の自分の logits で top-p の集合を決めている。HF の logits は bf16 の端数が一致しないため、top-p の境界（累積が 0.95 を跨ぐ最後の採用トークン）にいた y が HF 側で外れる事象が起こり得る。1 本が 1 万トークン規模なので、起きても不思議はない。**発生頻度は未確認**（GPU を使えないため）。`check` モードの「finite」の割合が 1.0 未満なら、その分が候補（−inf の中に「top-k 外」の無害な場合も含まれる）。`bench` モードは同じ損失式で逆伝播するので、NaN が出れば `|grad v15|` が `nan` と表示される。
- **根拠**:
  - PyTorch の `logsumexp_backward`: `https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/FunctionsManual.cpp` L1122-1133（`return grad * (self - result).exp().conj();`）。
  - 前方が −inf になる実装: `https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ReduceOps.cpp` L1570-1585（`maxes` の ±inf を 0 に置換してから `exp`・`sum`・`log`）。
  - コンテナ内の PyTorch の版は未確認（main ブランチのソースを読んだ）。
  - 後段の連鎖（`masked_fill`・`log_softmax` の逆伝播）は机上の推論で、実行して確かめていない。
- **修正案**（`logsumexp` をやめ、有限な位置だけを `where` で拾う。−inf 行は定数で返すので、後方に NaN が出ない）:

```diff
-        hit = idx == yc[:, None]
-        return lp.masked_fill(~hit, float("-inf")).logsumexp(-1)
+        hit = (idx == yc[:, None]) & keep            # y が top-k かつ top-p の内側にあるときだけ真
+        val = torch.where(hit, lp, torch.zeros_like(lp)).sum(-1)
+        return torch.where(hit.any(-1), val, torch.full_like(val, float("-inf")))
```

  併せて、勾配が非有限なら更新をスキップする:

```diff
-        gn = torch.nn.utils.clip_grad_norm_(r1.params(), 1.0)
-        opt.step()
+        gn = torch.nn.utils.clip_grad_norm_(r1.params(), 1.0)
+        if not math.isfinite(float(gn)):
+            print(json.dumps({"step": step, "skipped": "non-finite grad"}), flush=True)
+            continue
+        opt.step()
```

  （`continue` の前に `rel` の記録が飛ぶ点に注意。`log` へも記録するとよい。）

### A-2（実行を止める、机上）16 GB の GPU で、活性化チェックポイントなしでは逆伝播が入らない

- **場所**: `train_rank1_pg.py:151`（`dec(input_ids=x, use_cache=False)`）、`:174-178`（`model.eval()` のまま、チェックポイントの設定なし）。
- **前提（確認済み）**:
  - GPU は RTX 5070 Ti 16 GB（メモリノート `wsp-environment.md`）。
  - Gated DeltaNet は高速経路が無く、参照実装にフォールバックする。コンテナのログ `results/logs/t1.log` の 5-6 行目: 「`chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed」「`causal_conv1d_fn` is falling back …」。
  - 全 24 層のうち、層 3, 7, 11, 15, 19, 23 が全注意層、残り 18 層が線形注意（`config.json` の `layer_types`）。
- **動くか（依頼項目 5 の前半）**: フォールバックの `torch_chunk_gated_delta_rule` は純 PyTorch 演算（`transpose`/`to(fp32)`/`l2norm`/`F.pad`/行列積/`solve_triangular`/チャンクごとの Python ループ）だけでできていて、自動微分できる。`https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py` L254-285（conv の参照実装）、L286-419（チャンク則。main ブランチを読んだ。コンテナの版は未確認だが、ログの警告文は同ファイルの `use_kernel_func_from_hub_with_fallback` による）。**逆伝播自体は動くはず**。実機では未実行。
- **メモリの机上計算**（T ≈ 16.5k = プロンプト約 100〜300 + 応答 16,384、バッチ 1、1 系列ずつ `backward`）:
  - 編集より前の層（0〜14、層 15 の注意まで）は、入力もパラメータも勾配を要求しないので、計算グラフが作られない。グラフが残るのは、層 15 の MLP 以降の層 15〜23（9 層）。線形注意はそのうち 6 層（16, 17, 18, 20, 21, 22）、全注意は 19 と 23。
  - 線形注意 1 層（H=16, dk=dv=128, fp32 の中間量が 1 つ約 135 MB）。q/k/v の fp32 化・l2norm・スケール・パディング・β 掛け・減衰掛けなどで約 17 個の同サイズ量、チャンク内 64×64 の量が約 0.4 GB、チャンク走査（258 回）で約 0.4 GB、bf16 の射影・conv・ゲート・出力で約 1.1 GB。**合計 3〜5 GB／層**（main の実装は `solve_triangular` を使う。記憶では古い版に 63 回の in-place 前進代入ループがあり、コンテナの版がそれなら約 1.4 GB／層増える。版は未確認）。
  - MLP は 1 層あたり約 0.5 GB（gate・up・act・積が 118 MB×4 と入力）×9 ≈ 5 GB。全注意 2 層とノルム・残差で数 GB。
  - **合計 約 27〜39 GB（±50%）**。使える容量はモデル 1.7 GB と CUDA の領域、`lm_head` のチャンク（約 1.5 GB）を除いて約 12 GB。16,384 トークンは入らない。既定の `--max-len 8192` でも約 14〜20 GB で、入らない見込み。4,096 トークンなら約 7〜10 GB で入る見込み。
  - 標本の実際の長さ: `results/rb16k_own_k64_a6/practice_select128/summary.json` で平均 12,825 トークン、打ち切り 52%。
  - これは `bench` モードで測る設計になっている（計画 v3 の工程 4）。**ただし、`bench` が OOM で落ちるのは想定内の結果で、再現した場合は設計の修正が必要**。
- **修正案**: デコーダ層ごとの再計算を有効にする。`Qwen3_5DecoderLayer` は `GradientCheckpointingLayer`（`modeling_qwen3_5.py` L847）で、`supports_gradient_checkpointing = True`（L906）。保存されるのは層境界の隠れ状態（T×1024×2 B ≈ 34 MB ×9 ≈ 0.3 GB）と、再計算中の 1 層分（約 4〜5 GB）で、全体で約 9〜10 GB と見込まれる（机上）。

```diff
     model, dec, _ = load_model(a.model)
     for q in model.parameters():
         q.requires_grad_(False)
+    if a.mode != "check":
+        # 層ごとに再計算（非リエントラント版なら hook・フリーズしたパラメータとも相性が良い）。
+        # attention_dropout=0 なので train() にしても挙動は変わらない（GradientCheckpointingLayer は training 時のみ有効）。
+        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
+        model.train()
```

  計算時間は前進が約 1 回分増える。コンテナの `transformers` の版で `gradient_checkpointing_enable` の挙動が変わっている可能性があるため、`bench` で最大メモリを測ってから本番にすること。全注意層の SDPA 逆伝播（head_dim=256）が sm_120 でどのカーネルに落ちるかも未確認（フォールバックが math 版だと 16,384² の行列で 4 GB 超が必要）。

### A-3（結果を誤らせる）初期編集が予算の上限を超えている

- **場所**: docstring `train_rank1_pg.py:9-10`（「init = the own edit (k=64, alpha=12)」「‖v‖/‖W‖_F is projected to ≤ --rho-max」）、`:66`（`alpha_init=12.0`）、`:169`（`--rho-max 0.12`）、`:242`（射影）。
- **問題**: α=12 の初期編集の相対変化は、層 15 で 0.2354、層 19 で 0.2798（`models/edits/own_k64_a12/edit_log.json`。私の numpy 計算でも 0.23539 / 0.27983 で一致）。上限 0.12 の約 2 倍。`opt.step()` の直後に `project()` が呼ばれると、v が層 15 で約 0.51 倍、層 19 で約 0.43 倍に縮む。つまり**標本を取った方策（α=12）と、最初の更新後の方策（実質 α≈6）が大きく違う**。標本はこの 1 ステップで古くなり、2 ステップ目以降は A-6 のずれが大きい。さらに、`reports/plan_v3_current.md` の測定では α=6 が 41.4%、α=12 が 35.2% で、**縮小しただけで精度が上がり得る**。この縮小の効果を「学習の成果」と読み誤る恐れがある（A-5 と合わせて）。
- **根拠**: `edit_log.json`（α=1, 2, 3, 4 で 0.0196/0.0233 … 0.0785/0.0933、α=12 で 0.2354/0.2798。ほぼ α に比例）。numpy 計算: α=5 で 0.0981/0.1166（上限内）、α=6 で 0.1177/0.1399（層 19 が超過）。`models/edits/bo_k64_a5` は α=5 で 0.0981/0.1166 と一致する。
- **修正案**: 初期値を予算内にし、標本もその重みから取る。

```diff
-    def __init__(self, layers, dirpath, k_mask=256, k_init=64, alpha_init=12.0):
+    def __init__(self, layers, dirpath, k_mask=256, k_init=64, alpha_init=5.0):   # ρ = 0.098/0.117 <= 0.12
...
     r1.attach(layers)
+    print("init rel_change:", r1.project(a.rho_max), flush=True)   # 予算内なら不変
```

  第一案: α=5（`models/edits/bo_k64_a5` と同一）から標本を取る。第二案: α=12 から学習したいなら `--rho-max 0.30` にして、「予算 12%」という前提（計画 #2）を外す。どちらでも、docstring の初期値の記述を実装と揃える。

### A-4（結果を誤らせる）応答を 8,192 トークンで切るが、報酬は 16,384 トークン全体で決まる

- **場所**: `train_rank1_pg.py:148`（`rec["token_ids"][:max_len]`）、`:167`（`--max-len` 既定 8192）、`:205`。
- **問題**: 標本は 16,384 上限で取る（計画 v2 #6）。報酬 `correct` は全文の最終 `\boxed{}` で決まる。学習は先頭 `max_len` トークンの対数確率だけに勾配を流す。これは「前半部分の方策だけが報酬に効く」という一部分の勾配で、後半の変化は学習されない。
  - 編集の目的はループの抑制で、ループは末尾に出る（`find_direction.py:8` の「cut at max tokens whose tail is a loop」）。**ループの部分こそが切り捨てられる**。
  - 実測: 16,384 上限での平均長 12,825 トークン、打ち切り 52%、うち 67 件中 35 件が末尾ループ（`results/rb16k_own_k64_a6/practice_select128/summary.json`）。
- **修正案**: A-2 のチェックポイントを入れた上で `--max-len` の既定を 16384 にする（または全標本を丸ごと使うために `--max-len` を超える系列を除外するのではなく、上限を標本の上限に揃える）。メモリが足りず切るなら、切った系列は報酬の対象外にする（除外）ほうが、前半だけを学習するより偏りが分かりやすい。いずれも、`train_log.json` に「切った標本の割合」を記録する。

### A-5（結果の解釈を誤らせる）既定の更新量が極端に小さい

- **場所**: `train_rank1_pg.py:165-168`（`--steps 4`, `--problems-per-step 6`, `--lr-v 2e-3`, `--lr-u 2e-4`）、`:216-217`、`:222`（`mixed` を 1 回だけ通る）。
- **問題**: Adam（β1=0.9, β2=0.99）の 1 ステップの更新は各要素で概ね lr 以下（正規化した更新量の上界が (1−β1)/√(1−β2) = 1 であることからの机上の見積もり）。4 ステップで v の各要素は高々 4×2e-3 = 0.008 しか動かない。射影後（ρ=0.12）の v の上位 64 要素の二乗平均平方根は 約 0.25（‖v‖≈1.97 を √64 で割る）なので、**高々 約 3%**。u は 4×2e-4 = 8e-4 で、要素の大きさ（約 1/32 ≈ 0.031）の約 2.6%。つまり、学習後の編集は「射影された初期値」とほぼ同じ。精度が変わっても、それは A-3 の縮小の効果で、方策勾配の効果とは言えない。
  - 1 パスの上限: `mixed`（混合報酬の問題）を 1 回しか使わない（`:223`）。例えば 256 問・1 問 8 標本で混合が 100 問なら、1 バッチ 12 問で約 8 ステップ。これも総移動量の上界は 8×2e-3 = 0.016。
  - eps: 損失を `nseq·C`（約 10⁶）で割るので、勾配の要素は 1e-7〜1e-5 と見込まれる（机上の概算、未測定）。eps=1e-8 に近い場合は更新が縮む。`bench` の `|grad v15|` で確認を。
- **修正案**: 「標本→数ステップ→書き出し→標本を取り直す」のラウンドを繰り返す設計にする（docstring は `--rounds` と書いているが、引数は存在しない: A-9）。1 ラウンドのステップ数は、標本のずれが小さい範囲（A-6）に抑える。変化を見たいなら lr_v を上げる（例 1e-2）か、ステップ数を増やす。報告では、学習前（射影のみ）の編集と、学習後を**必ず並べて比較**する（射影のみの対照を 1 本入れる）。

### A-6（軽微〜中）古い標本の再利用と、HF–vLLM の数値差

- **場所**: `train_rank1_pg.py:222-244`、`:238-239`（`drift`）。
- **問題**: 全ステップが方策 θ₀ の標本を使い、重要度比（π_θt/π_θ0）を掛けていない。θ が動くほど偏る。A-3 で見たように、1 ステップ目の射影でずれが最も大きくなる。また、`drift`（`lp − ref` の平均）には、方策の変化に加えて **HF と vLLM の数値差**（bf16 の端数、カーネルの違い、16k 系列での蓄積）が含まれる。後者の大きさは `check` モードで測る設計だが、まだ実行していない（未確認）。
- **修正案**: 標本をラウンドごとに取り直す（A-5）。重要度比を入れる場合はトークン単位でクリップする（系列全体の比は 10⁴ 個の差を足すので不安定）。`check` で `mean|diff|` が大きい（例 0.05 超）なら、重要度比は使わない。

### A-7（軽微）無音の失敗モード

- **場所**: `train_rank1_pg.py:214-215`, `:224-225`, `:111-133`, `:245-247`。
- **問題**:
  1. 混合報酬の問題が 0 件（または全問題が 1 標本）だと、`batch` が空で最初に `break` し、**何も学習せずに初期値（射影前）を「学習済み」として書き出す**。実例として、既存の 16,384 上限の測り直しデータは 1 問 1 標本（`samples: 1`、`rb16k_*` で確認）なので、そのまま渡すと混合が 0 件になる。
  2. `export` は 2 層の編集が両方入ったかを検証しない（`make_edit.py` の `assert len(log["tensors"]) == len(layers)` に相当するものがない）。キー名が合わなければ黙ってスキップされる。
  3. 学習の途中で落ちると何も残らない（u, v の途中保存がない）。
- **修正案**:

```diff
     mixed = [g for g in groups.values() if len(g) >= 2 and 0 < sum(x["reward"] for x in g) < len(g)]
     print(f"problems {len(groups)}, with mixed rewards {len(mixed)}", flush=True)
+    if a.mode == "train" and len(mixed) < a.problems_per_step:
+        sys.exit(f"not enough mixed-reward problems: {len(mixed)} < {a.problems_per_step} (need >=2 samples per problem)")
```

```diff
+        assert len(log) == len(LAYERS), f"edited {list(log)} (expected layers {LAYERS})"
         json.dump(log, open(os.path.join(out_dir, "edit_log.json"), "w"), indent=1)
```

  （上の `assert` は `save_file` のループを抜けた後、`json.dump` の直前に置く。）各ステップ後に `torch.save({L: (u, v)}, out/ckpt.pt)` を入れる。

### A-8（軽微）`check` の対象とデータの出所

- **場所**: `train_rank1_pg.py:185-192`。
- **問題**: 対象が最短の n 本（既定 4）。最短の系列は短い答え（数百トークン）で、presence penalty の対象語彙・長い文脈での GDN 状態・top-p 境界の事象が少ない。A-1 や HF–vLLM のずれを検出する力が弱い。また、`check` は「標本を取った重み＝hook の初期値」を前提にするが、どの重みから標本を取ったかを検証しない。ずれていれば `mean diff` が大きく出るだけで、原因が分からない。
- **修正案**: 長さの異なる系列（短い 1 本＋中くらい 1 本＋最長 2 本）を使い、`finite` の割合が 0.99 未満、または `mean|diff|` が大きい（例 0.1 超）なら終了コードを非ゼロにする。標本の取得元（`--model` のパス）を `samples` の隣に記録しておく。

### A-9（軽微）文書・既定値の不一致

- **場所**: `train_rank1_pg.py:13`（「`train` (--rounds of steps over the data, …)」）に対し引数は `--steps`（`:165`）、`--rounds` は存在しない。`--problems-per-step` の既定は 6（`:166`）だが、計画 v2 の #1 は「1 バッチ 12 問以上」「1 問 8 標本以上」。標本数のチェックもない。
- **修正案**: 既定を 12 に。`min(len(g))<8` なら警告。docstring を実装に合わせる。

### A-10（軽微）同順位・境界の扱い

- **場所**: `train_rank1_pg.py:50-53`。
- **問題**: top-k の 20 番目に同値がある場合、vLLM の Triton 実装は「ちょうど k 個」に絞り、同値の中では語彙の添字順に残す（`topk_topp_triton.py:382-389`, `:421-437`）。`torch.topk` は同値の順が不定。top-p でも同値は添字順に部分的に残す（`:596-602`）。ロジットが bf16 由来の粗い格子（|z|∈[8,32) で間隔 0.0625〜0.125）を持つので、同値は起こり得る。ただし境界の 20 番目は確率がごく小さく、top-p で先に切られることが多いので、対数確率への影響は小さい。y が両側で食い違った場合の NaN は A-1 で防げる。
- **未確認**: sm_120 での Triton カーネルの厳密さ（二分探索が 18 回で止まる近似）。ほぼ厳密と見込まれるが、テストしていない。

### A-11（軽微）グラフの無駄・import

- `p = torch.softmax(vals, -1)`（`:51`）は `keep` の計算にしか使われない。勾配が不要なのに計算グラフに残る。`with torch.no_grad():` 内で `keep` を作ればよい。
- `torch.utils.checkpoint` は `import torch` だけでは保証されない（`:61`）。`load_model` 内の `import transformers` が先に読むので実際には動くが、`import torch.utils.checkpoint` を明示するのが安全。

## 3. 未確認の項目

- コンテナ内で実際に動いた vLLM の経路（V1 / V2）と、`vllm`・`torch`・`transformers` の実際の版。`Dockerfile` は `vllm/vllm-openai:v0.32.0` から作るが、コンテナ内で版を確認していない。ログのレベルが WARNING のため、経路の記録がない。
- A-1 の発生頻度。A-2 の実測メモリと所要時間。HF と vLLM の対数確率の差の大きさ（`check` を実行していない）。全注意層（head_dim=256）の SDPA 逆伝播が sm_120 で使うカーネル。
- V2 の `output_bin_counts` の更新カーネル本体（呼び出し位置 `model_runner.py:1688-1696` のみ確認）。vLLM の `lm_head` が出す logits の dtype（HF と同じ bf16 を前提にしている）。
- Triton の top-k/top-p カーネルが sm_120 で近似に落ちる頻度。
- 学習が実際に精度を上げるかどうか（実行していないので、いかなる成果も主張しない）。

## 4. 実行前チェックの推奨順

1. A-1（`where` 版）と A-7 の入力検査を入れる。
2. A-2 のチェックポイントを入れ、`--mode bench --max-len 4096` → 8192 → 16384 の順に、最大メモリと `|grad v15|`（有限か）を確認する。
3. A-3（α=5 から標本を取り直す）。A-4（`--max-len` を標本の上限に揃える）。
4. `--mode check` を長さの異なる系列で実行し、`finite`・`mean|diff|` を見る。
5. 射影のみ（学習なし）の編集を対照として 1 本評価に回す（A-5）。

## 5. 参照した主なソース（取得日 2026-11-04、すべて読むだけで実行していない）

- vLLM v0.32.0: `vllm/config/vllm.py`、`vllm/config/model.py`、`vllm/v1/sample/sampler.py`、`vllm/v1/sample/ops/topk_topp_sampler.py`、`vllm/v1/sample/ops/topk_topp_triton.py`、`vllm/v1/sample/ops/penalties.py`、`vllm/model_executor/layers/utils.py`、`vllm/v1/worker/gpu/sample/{sampler,penalties,states,logprob}.py`、`vllm/v1/worker/gpu/model_runner.py`、`vllm/v1/engine/parallel_sampling.py`（`https://github.com/vllm-project/vllm/blob/v0.32.0/<path>`）。
- PyTorch main: `torch/csrc/autograd/FunctionsManual.cpp`（L1122-1133）、`aten/src/ATen/native/ReduceOps.cpp`（L1570-1585）。
- transformers main: `src/transformers/models/qwen3_5/modeling_qwen3_5.py`（L254-285, L286-419, L489-640, L847, L906, L1196-1263）。
- プロジェクト内: `models/edits/own_k64_a12/edit_log.json`、`models/edits/bo_k64_a5/edit_log.json`、`results/logs/t1.log`、`results/rb16k_own_k64_a6/practice_select128/summary.json`、`scripts/make_edit.py`、`scripts/find_direction.py`、`models/Qwen3.5-0.8B/{config.json,tokenizer_config.json,chat_template.jinja}`。
