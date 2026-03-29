#!/usr/bin/env python3
"""ANE inference wrapper — handles Qwen3 tokenization for the C binary.

Usage:
    python3 infer.py "Once upon a time"
    python3 infer.py --ckpt PATH --temp 0.8 --topk 40 --maxlen 200 "Your prompt here"
    echo "Tell me a story" | python3 infer.py --stdin
"""
import subprocess
import sys
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="ANE inference with Qwen3 tokenization")
    parser.add_argument("prompt", nargs="?", default=None, help="Text prompt")
    parser.add_argument("--stdin", action="store_true", help="Read prompt from stdin")
    parser.add_argument("--ckpt", default="ane_qwen3_06b_dyn_ckpt.bin", help="Checkpoint path")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature (0=argmax)")
    parser.add_argument("--topk", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--maxlen", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--binary", default="./infer", help="Path to infer binary")
    parser.add_argument("--token-ids", action="store_true",
                        help="Skip tokenizer — prompt is comma-separated token IDs")
    args = parser.parse_args()

    # Get prompt
    if args.stdin:
        prompt = sys.stdin.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = input("Prompt: ")

    if not prompt:
        print("Error: empty prompt", file=sys.stderr)
        sys.exit(1)

    # Tokenize
    if args.token_ids:
        token_ids = [int(x.strip()) for x in prompt.split(",") if x.strip()]
        prompt_text = f"[{len(token_ids)} token IDs]"
    else:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print("Error: pip install transformers  (needed for Qwen3 tokenizer)", file=sys.stderr)
            print("Or use --token-ids to pass raw token IDs", file=sys.stderr)
            sys.exit(1)

        print("Loading Qwen3 tokenizer...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        token_ids = tokenizer.encode(prompt)
        prompt_text = prompt

    print(f"Prompt: {prompt_text}", file=sys.stderr)
    print(f"Prompt tokens: {len(token_ids)}", file=sys.stderr)

    # Build command
    cmd = [
        args.binary,
        "--ckpt", args.ckpt,
        "--temp", str(args.temp),
        "--topk", str(args.topk),
        "--maxlen", str(args.maxlen),
    ]

    # Launch C binary
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    )

    # Send prompt token IDs
    for tid in token_ids:
        proc.stdin.write(f"{tid}\n")
    proc.stdin.write("\n")  # end of prompt
    proc.stdin.flush()

    # Read generated tokens and decode
    generated_ids = []
    if not args.token_ids:
        # Print prompt first
        sys.stdout.write(prompt)
        sys.stdout.flush()

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        tid = int(line)
        generated_ids.append(tid)

        if args.token_ids:
            sys.stdout.write(f"{tid} ")
            sys.stdout.flush()
        else:
            # Incremental decode
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            # Overwrite line with full decoded text
            sys.stdout.write(f"\r{prompt}{text}")
            sys.stdout.flush()

    proc.wait()
    print()  # final newline

    if not args.token_ids:
        total_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"\n--- Generated {len(generated_ids)} tokens ---", file=sys.stderr)

if __name__ == "__main__":
    main()
