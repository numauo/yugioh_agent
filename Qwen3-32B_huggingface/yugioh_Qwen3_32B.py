# yugioh_Qwen3_32B.py
# -*- coding: utf-8 -*-
"""
Yu-Gi-Oh! Duel Agent (CLI) powered by Hugging Face Qwen/Qwen3-32B

- 長文プロンプトをそのまま投げて対話進行
- 先攻/後攻は乱数
- あなた（ユーザー）の初期手札5枚はコンソールだけに表示（モデルには送らない）
- デッキは日本語キーの JSON/JSONC（コメント可）から読み込み（最低 "カード名" が必要）
- Qwen3-32B の生成は attention_mask 明示・dtype自動切替で安定化
- 生成長や初回スキップ、温度/Top-p を CLI で調整可能

使い方:
    python yugioh_Qwen3_32B.py --prompt-file prompt_yugioh.txt \
        --user-deck-json user_deck.json \
        --ai-deck-json ai_deck.json \
        --max-new-tokens 300 \
        --temperature 0.7 \
        --top-p 0.9 \
        [--skip-open]
"""

import argparse
import os
import random
import re
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.utils import is_torch_cuda_available

# Qwen3-32B は思考モード/非思考モードを切り替え可能な統合モデル（Instruct用途も含む）
MODEL_ID_DEFAULT = "Qwen/Qwen3-32B"  # 環境変数 MODEL_ID で上書き可

# ====== 対話用メタ指示 ======
SYSTEM_PRIMER = """以下は“対戦進行のためのメタ指示”です。必ず守ってください。
- あなたは遊戯王OCGの進行役AI。ルール厳守、裁定は合理的に。
- あなたの手札/デッキの中身は絶対に公開しない（推測表現もしない）。
- 先攻/後攻は乱数決定結果に従う。毎フェイズの区切りで「チェーンしますか？」等、相手のクイックエフェクト/罠の有無を確認する。
- ターンごとにLP・フィールドの公開情報を簡潔に要約し、ユーザーが入力しやすい形で質問する。
- 出力は日本語・簡潔・手番/フェイズ名を明記。長くなりすぎないこと。
- 不確定情報（相手手札など）は断定禁止。必要なら「確認しますか？」と尋ねる。
"""

MIN_FALLBACK_USER_PROMPT = """遊戯王で私と戦ってください。LPは4000、初手は5枚、先攻/後攻は乱数で決めてください。
あなたのプレイ中に私が手札や魔法・罠を発動できるなら、毎ステップで確認してください。
あなたの手札は絶対に公開しないでください。デッキリストは会話の前提として参照してください。
"""

# ====== ユーティリティ ======
def strip_json_comments(text: str) -> str:
    """// と /* */ のコメントを取り除いた JSON 文字列を返す（簡易JSONC対応）"""
    # ブロックコメント
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # 行コメント
    text = re.sub(r"(^|\s)//.*?$", r"\1", text, flags=re.MULTILINE)
    return text

def load_json_maybe_with_comments(path: str) -> Any:
    import json
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = strip_json_comments(raw).strip()
    return json.loads(cleaned)

def deck_to_name_list(deck_objs: List[Dict[str, Any]]) -> List[str]:
    """カード配列（辞書） -> 表示用のカード名配列。最低 'カード名'（なければ 'name'）を使用。"""
    names = []
    for obj in deck_objs:
        if isinstance(obj, dict):
            name = obj.get("カード名") or obj.get("name")
            if name:
                names.append(str(name))
    return names

def load_long_prompt(path: str) -> str:
    if not path:
        return MIN_FALLBACK_USER_PROMPT
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[warn] 指定のプロンプトファイルが見つかりませんでした: {path}")
        return MIN_FALLBACK_USER_PROMPT

def draw_starting_hand(deck_names: List[str], n: int = 5) -> List[str]:
    deck_copy = deck_names[:]
    random.shuffle(deck_copy)
    return deck_copy[:n]

def coinflip_first_player() -> str:
    return random.choice(["あなた（AI）", "ユーザー（あなた）"])

def build_messages(system_primer: str, long_prompt: str, first_player: str) -> List[Dict[str, str]]:
    # 先攻/後攻の結果だけは会話に含める（ユーザーの初手は含めない）
    return [
        {"role": "system", "content": system_primer.strip()},
        {"role": "user", "content": long_prompt.strip()},
        {"role": "user", "content": f"乱数の結果：先攻は {first_player} です。デュエルを開始してください。"},
    ]

# ====== 生成 安定化 ======
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def _pick_dtype():
    if is_torch_cuda_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32

def generate_reply(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 800,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Qwen系テンプレートを使って応答を生成。attention_mask 明示、分布正規化あり。"""
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model.device)
    except Exception:
        merged = ""
        for m in messages:
            prefix = "System: " if m["role"] == "system" else "User: " if m["role"] == "user" else "Assistant: "
            merged += f"{prefix}{m['content']}\n"
        input_ids = tokenizer.encode(merged + "Assistant: ", return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model.device)

    eos = tokenizer.eos_token_id

    class _NanSafeLogitsProcessor(LogitsProcessor):
        """Clamp/logits sanitize to avoid NaN/Inf causing multinomial device asserts."""

        def __call__(self, input_ids, scores):
            orig_dtype = scores.dtype
            if scores.dtype != torch.float32:
                scores = scores.float()
            scores = torch.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=-50.0)
            scores = scores.clamp_(min=-50.0, max=50.0)
            return scores.to(orig_dtype)

    extra_processors = LogitsProcessorList([_NanSafeLogitsProcessor()])
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            renormalize_logits=True,  # NaN/inf や負の確率の事故を抑止
            pad_token_id=eos,
            eos_token_id=eos,
            logits_processor=extra_processors,
        )
    out = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return out.strip()

# ====== メイン ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", MODEL_ID_DEFAULT))
    parser.add_argument("--prompt-file", default=None, help="長文プロンプトのファイル（例: prompt_yugioh.txt）")
    parser.add_argument("--user-deck-json", default=None, help="ユーザー側デッキ（JSON/JSONC, 配列）のパス")
    parser.add_argument("--ai-deck-json", default=None, help="AI側デッキ（JSON/JSONC, 配列）のパス")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "800")),
                        help="1回の生成トークン上限。重い/長い場合は下げる（例: 300）")
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.7")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "0.9")))
    parser.add_argument("--skip-open", action="store_true",
                        help="AIの初回応答をスキップして、すぐユーザー入力へ移る")
    parser.add_argument("--open-max-new-tokens", type=int, default=None,
                        help="初回応答専用のトークン上限（未指定なら --max-new-tokens と同じ）")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    print(f"[info] モデル: {args.model_id}")

    # ---- モデル読み込み ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = _pick_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if getattr(model, "generation_config", None) is not None:
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        if model.generation_config.eos_token_id is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.eval()

    # ---- デッキ読込（JSON/JSONC） ----
    user_deck_objs: List[Dict[str, Any]] = []
    ai_deck_objs: List[Dict[str, Any]] = []
    if args.user_deck_json:
        try:
            loaded = load_json_maybe_with_comments(args.user_deck_json)
            if isinstance(loaded, list):
                user_deck_objs = loaded
            else:
                raise ValueError("ユーザーデッキJSONは配列（list）である必要があります。")
        except Exception as e:
            print(f"[error] ユーザーデッキの読み込みに失敗: {e}")
            return
    if args.ai_deck_json:
        try:
            loaded = load_json_maybe_with_comments(args.ai_deck_json)
            if isinstance(loaded, list):
                ai_deck_objs = loaded
            else:
                raise ValueError("AIデッキJSONは配列（list）である必要があります。")
        except Exception as e:
            print(f"[error] AIデッキの読み込みに失敗: {e}")
            return

    # 名前配列へ（初期手札表示などに使用。モデルへは送らない）
    user_deck_names = deck_to_name_list(user_deck_objs)
    ai_deck_names = deck_to_name_list(ai_deck_objs)

    if not user_deck_names:
        print("[warn] ユーザー側デッキが空です。--user-deck-json を指定し、各要素に『カード名』を含めてください。")
    if not ai_deck_names:
        print("[warn] AI側デッキが空です。--ai-deck-json を指定し、各要素に『カード名』を含めてください。")

    # ---- 先攻/後攻 ----
    first = coinflip_first_player()

    # ---- ユーザーの初期手札（モデルには送らない！）----
    starting_hand = draw_starting_hand(user_deck_names if user_deck_names else [], 5)

    print("\n================ 初期セットアップ ================\n")
    print(f"先攻: {first}")
    print(f"あなたのデッキ枚数: {len(user_deck_names)}  / AIのデッキ枚数: {len(ai_deck_names)}")
    print("あなたの初期手札（モデルには送っていません）:")
    for c in starting_hand:
        print(f"  - {c}")
    print("\n※以降、モデル（対戦AI）の問いかけに対して日本語で入力してください。終了は 'exit' または 'quit'。\n")

    # ---- プロンプト初期化 ----
    long_prompt = load_long_prompt(args.prompt_file)
    messages = build_messages(SYSTEM_PRIMER, long_prompt, first)

    # ---- 開始発話 ----
    if not args.skip_open:
        open_max_new_tokens = args.open_max_new_tokens or args.max_new_tokens
        print(f"[info] 初回応答を生成中... (max_new_tokens={open_max_new_tokens}, temp={args.temperature}, top_p={args.top_p})")
        assistant = generate_reply(
            model, tokenizer, messages,
            max_new_tokens=open_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("\nAI> " + assistant + "\n")
        messages.append({"role": "assistant", "content": assistant})
    else:
        print("[info] 初回応答をスキップ。あなたの入力を待ちます。")

    # ---- 対話ループ ----
    while True:
        try:
            user_in = input("あなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[info] 終了します。")
            break
        if user_in.lower() in ("exit", "quit"):
            print("[info] 終了します。")
            break

        messages.append({"role": "user", "content": user_in})
        assistant = generate_reply(
            model, tokenizer, messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("\nAI> " + assistant + "\n")
        messages.append({"role": "assistant", "content": assistant})

if __name__ == "__main__":
    main()
