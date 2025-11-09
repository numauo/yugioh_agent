#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yu-Gi-Oh! Duel Agent (CLI) powered by Groq GPT-OSS 120B

このスクリプトは Groq の「openai/gpt-oss-120b」モデルを利用して、
遊戯王のデュエル進行をチャット形式で行う CLI ツールです。
Groq API キーを環境変数 GROQ_API_KEY か --api-key で指定して実行してください。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError as exc:  # pragma: no cover - dependency check
    print(
        "[error] Python package 'requests' が必要です。"
        " 'pip install requests' でインストールしてください。",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

MODEL_ID_DEFAULT = "openai/gpt-oss-120b"
API_BASE_DEFAULT = "https://api.groq.com/openai/v1"

# ====== 対話用メタ指示 ======
SYSTEM_PRIMER = """以下は“対戦進行のためのメタ指示”です。必ず守ってください。
- あなたは遊戯王OCGの進行役AI。裁定は合理的に。
- prompt_yugioh_copy.txt の内容を厳密に守る。
- ai-deck.json , user-deck.json のデッキリストに記載されていることを遵守する。
- あなたの手札/デッキの非公開情報は絶対に公開しない（推測表現もしない）。
- 私がドローした際は、なんのカードを引いたか私に教えること。
- 表側表示でモンスターを召喚したら、なんのモンスターを召喚したか教えること。
- 先攻/後攻は乱数決定結果に従う。各フェイズで相手のクイックエフェクトや罠の有無を丁寧に確認する。
- ターンごとにLP・フィールドの公開情報をまとめて表示すること。
- あなたのプレイ中に私が手札や魔法・トラップを発動できるなら、私が発動するか確認してください。
- 出力は日本語・簡潔・手番/フェイズ名を明記。長すぎないこと。
- 不確定情報（相手手札など）は断定禁止。必要なら「確認しますか？」と尋ねる。
"""

DEFAULT_PROMPT_CANDIDATES = ["prompt_yugioh.copy.txt", "prompt_yugioh.txt"]

MIN_FALLBACK_USER_PROMPT = """遊戯王で私と戦ってください。LPは4000、初手は5枚、先攻/後攻は乱数で決めてください。
あなたのプレイ中に私が手札や魔法・罠を発動できるなら、毎ステップで確認してください。
私がルール違反をしたら指摘し、巻き戻して再開してください。あなたの手札は公開しないでください。
"""


# ====== ユーティリティ ======
def strip_json_comments(text: str) -> str:
    """JSONCからコメントを除去する簡易処理。"""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"(^|\s)//.*?$", r"\1", text, flags=re.MULTILINE)
    return text


def load_json_maybe_with_comments(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = strip_json_comments(raw).strip()
    return json.loads(cleaned)


def deck_to_name_list(deck_objs: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for obj in deck_objs:
        if isinstance(obj, dict):
            name = obj.get("カード名") or obj.get("name")
            if name:
                names.append(str(name))
    return names


def load_long_prompt(path: Optional[str]) -> str:
    candidates: List[str] = []
    if path:
        candidates.append(path)
    else:
        env_prompt = os.environ.get("PROMPT_FILE")
        if env_prompt:
            candidates.append(env_prompt)
        candidates.extend(DEFAULT_PROMPT_CANDIDATES)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            continue

    if path:
        print(f"[warn] 指定のプロンプトファイルが見つかりませんでした: {path}")
    else:
        print("[warn] 指定されたプロンプトファイルや既定候補が見つかりませんでした。")
    return MIN_FALLBACK_USER_PROMPT


def draw_starting_hand(deck_names: List[str], n: int = 5) -> List[str]:
    deck_copy = deck_names[:]
    random.shuffle(deck_copy)
    return deck_copy[:n]


def coinflip_first_player() -> str:
    return random.choice(["AI（あなた）", "ユーザー（相手）"])


def format_deck_section(title: str, cards: List[str]) -> str:
    if not cards:
        return f"{title}:\n- （デッキリストが提供されていません）"
    lines = "\n".join(f"- {card}" for card in cards)
    return f"{title}:\n{lines}"


def build_messages(
    system_primer: str,
    duel_prompt: str,
    first_player: str,
    user_deck_names: List[str],
    ai_deck_names: List[str],
) -> List[Dict[str, str]]:
    deck_blob = "\n\n".join(
        [
            "【デッキリスト共有（会話前提）】",
            format_deck_section("ユーザー側デッキ", user_deck_names),
            format_deck_section("AI側デッキ", ai_deck_names),
        ]
    )
    return [
        {"role": "system", "content": system_primer.strip()},
        {"role": "user", "content": f"{duel_prompt.strip()}\n\n{deck_blob.strip()}"},
        {"role": "user", "content": f"乱数の結果：先攻は {first_player} です。デュエルを開始してください。"},
    ]


# ====== Groq API 呼び出し ======
def call_groq_chat_completion(
    *,
    api_key: str,
    api_base: str,
    model: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_new_tokens),
        "stream": False,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=request_timeout,
            )
        except requests.RequestException as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Groq API request failed: {exc}") from exc
            time.sleep(retry_backoff ** attempt)
            continue

        if response.status_code >= 400:
            try:
                api_error = response.json()
            except ValueError:
                api_error = {"error": {"message": response.text}}

            if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait_for = retry_backoff ** attempt
                print(f"[warn] Groq API error {response.status_code}: {api_error}. Retrying in {wait_for:.1f}s...")
                time.sleep(wait_for)
                continue
            raise RuntimeError(f"Groq API error {response.status_code}: {api_error}")

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Groq API returned no choices in response.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise RuntimeError("Groq API response missing 'content' field.")
        return content.strip()

    raise RuntimeError("Groq API request exhausted all retries.")  # defensive


# ====== メイン ======
def main() -> None:
    parser = argparse.ArgumentParser(description="Yu-Gi-Oh! duel agent using Groq GPT-OSS-120B.")
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", MODEL_ID_DEFAULT))
    parser.add_argument("--prompt-file", default=None, help="長文プロンプトのファイル（例: prompt_yugioh.txt）")
    parser.add_argument("--user-deck-json", default=None, help="ユーザー側デッキ（JSON/JSONC, 配列）のパス")
    parser.add_argument("--ai-deck-json", default=None, help="AI側デッキ（JSON/JSONC, 配列）のパス")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "2000")),
                        help="1回の生成トークン上限（例: 500）。")
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.7")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "0.9")))
    parser.add_argument("--request-timeout", type=float, default=float(os.environ.get("REQUEST_TIMEOUT", "180")),
                        help="HTTPリクエストのタイムアウト秒数。")
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("MAX_RETRIES", "3")))
    parser.add_argument("--skip-open", action="store_true",
                        help="AIの初回応答をスキップして、すぐユーザー入力へ移る。")
    parser.add_argument("--open-max-new-tokens", type=int, default=None,
                        help="初回応答専用のトークン上限（未指定なら --max-new-tokens と同じ）。")
    parser.add_argument("--api-base", default=os.environ.get("GROQ_API_BASE", API_BASE_DEFAULT),
                        help="Groq API のベースURL。")
    parser.add_argument("--api-key", default=os.environ.get("GROQ_API_KEY"),
                        help="Groq API キー。未指定なら環境変数 GROQ_API_KEY を参照。")
    parser.add_argument("--no-deck-sharing", action="store_true",
                        help="モデルへデッキリスト全文を送らない（トークン節約用）。")
    args = parser.parse_args()

    if not args.api_key:
        print("[error] Groq API key is required. Set --api-key or environment variable GROQ_API_KEY.", file=sys.stderr)
        raise SystemExit(1)

    if args.seed is not None:
        random.seed(args.seed)

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
        except Exception as exc:
            print(f"[error] ユーザーデッキの読み込みに失敗: {exc}", file=sys.stderr)
            raise SystemExit(1)
    if args.ai_deck_json:
        try:
            loaded = load_json_maybe_with_comments(args.ai_deck_json)
            if isinstance(loaded, list):
                ai_deck_objs = loaded
            else:
                raise ValueError("AIデッキJSONは配列（list）である必要があります。")
        except Exception as exc:
            print(f"[error] AIデッキの読み込みに失敗: {exc}", file=sys.stderr)
            raise SystemExit(1)

    user_deck_names = deck_to_name_list(user_deck_objs)
    ai_deck_names = deck_to_name_list(ai_deck_objs)

    if not user_deck_names:
        print("[warn] ユーザー側デッキが空です。--user-deck-json を指定し、各要素に『カード名』を含めてください。")
    if not ai_deck_names:
        print("[warn] AI側デッキが空です。--ai-deck-json を指定し、各要素に『カード名』を含めてください。")

    # ---- 先攻/後攻 ----
    first = coinflip_first_player()

    # ---- ユーザーの初期手札（モデルには送らない）----
    starting_hand = draw_starting_hand(user_deck_names if user_deck_names else [], 5)

    print("\n================ 初期セットアップ ================\n")
    print(f"先攻: {first}")
    print(f"あなたのデッキ枚数: {len(user_deck_names)}  / AIのデッキ枚数: {len(ai_deck_names)}")
    print("あなたの初期手札（モデルには送っていません）:")
    if starting_hand:
        for card in starting_hand:
            print(f"  - {card}")
    else:
        print("  - （デッキ未指定のため手札を引けません）")
    print("\n※以降、モデル（対戦AI）の問いかけに対して日本語で入力してください。終了は 'exit' または 'quit'。\n")

    # ---- プロンプト初期化 ----
    duel_prompt = load_long_prompt(args.prompt_file)
    if args.no_deck_sharing:
        messages = [
            {"role": "system", "content": SYSTEM_PRIMER.strip()},
            {"role": "user", "content": duel_prompt.strip()},
            {"role": "user", "content": f"乱数の結果：先攻は {first} です。デュエルを開始してください。"},
        ]
    else:
        messages = build_messages(SYSTEM_PRIMER, duel_prompt, first, user_deck_names, ai_deck_names)

    open_max_new_tokens = args.open_max_new_tokens or args.max_new_tokens

    # ---- 開始発話 ----
    if not args.skip_open:
        print(f"[info] Groqモデル: {args.model_id}")
        print(f"[info] 初回応答を生成中... (max_new_tokens={open_max_new_tokens}, temp={args.temperature}, top_p={args.top_p})")
        try:
            assistant_reply = call_groq_chat_completion(
                api_key=args.api_key,
                api_base=args.api_base,
                model=args.model_id,
                messages=messages,
                max_new_tokens=open_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
        except Exception as exc:
            print(f"[error] Groq API呼び出し中にエラーが発生しました: {exc}", file=sys.stderr)
            raise SystemExit(1)
        print("\nAI> " + assistant_reply + "\n")
        messages.append({"role": "assistant", "content": assistant_reply})
    else:
        print(f"[info] Groqモデル: {args.model_id}")
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
        try:
            assistant_reply = call_groq_chat_completion(
                api_key=args.api_key,
                api_base=args.api_base,
                model=args.model_id,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
        except Exception as exc:
            print(f"[error] Groq API呼び出し中にエラーが発生しました: {exc}", file=sys.stderr)
            break

        print("\nAI> " + assistant_reply + "\n")
        messages.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()
