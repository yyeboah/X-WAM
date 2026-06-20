#!/usr/bin/env python3
"""Resumable Hugging Face downloader with conservative rate-limit handling."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face repo to a local directory with 429 backoff."
    )
    parser.add_argument("repo_id", help="Repository id, e.g. sharinka0715/X-WAM-RoboTwin")
    parser.add_argument("--repo-type", default=None, help="Repo type, e.g. dataset")
    parser.add_argument("--local-dir", required=True, help="Destination directory")
    parser.add_argument("--revision", default=None, help="Optional revision/commit")
    parser.add_argument(
        "--include",
        action="append",
        dest="allow_patterns",
        help="Optional allow pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        dest="ignore_patterns",
        help="Optional ignore pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--chunk-from",
        type=int,
        default=None,
        help="Download only chunk directories from this index, e.g. 6 for chunk-0006.",
    )
    parser.add_argument(
        "--chunk-to",
        type=int,
        default=None,
        help="Last chunk index to include. Defaults to --chunk-from when omitted.",
    )
    parser.add_argument(
        "--chunk-prefix",
        default="data/chunk-",
        help="Repo-relative chunk prefix used with --chunk-from.",
    )
    parser.add_argument(
        "--chunk-width",
        type=int,
        default=4,
        help="Zero padding width for chunk indexes.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel downloads. Keep this at 1 for repos with many small files.",
    )
    parser.add_argument(
        "--base-sleep",
        type=float,
        default=330.0,
        help="Seconds to sleep after a 429 when Retry-After is absent.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=100,
        help="Maximum retry attempts after rate-limit or transient HTTP errors.",
    )
    return parser.parse_args()


def build_allow_patterns(args: argparse.Namespace) -> list[str] | None:
    patterns = list(args.allow_patterns or [])
    if args.chunk_from is not None:
        chunk_to = args.chunk_to if args.chunk_to is not None else args.chunk_from
        if chunk_to < args.chunk_from:
            raise ValueError("--chunk-to must be greater than or equal to --chunk-from")
        patterns.extend(
            f"{args.chunk_prefix}{index:0{args.chunk_width}d}/*"
            for index in range(args.chunk_from, chunk_to + 1)
        )
    return patterns or None


def retry_after_seconds(error: HfHubHTTPError, fallback: float, attempt: int) -> float:
    response = getattr(error, "response", None)
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass

    jitter = random.uniform(0, 30)
    return fallback + min(attempt, 6) * 30 + jitter


def main() -> int:
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = build_allow_patterns(args)

    for attempt in range(args.max_retries + 1):
        try:
            path = snapshot_download(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=args.ignore_patterns,
                max_workers=args.max_workers,
                resume_download=True,
            )
            print(f"Download complete: {path}")
            return 0
        except HfHubHTTPError as error:
            status_code = error.response.status_code if error.response is not None else None
            if status_code not in {429, 500, 502, 503, 504} or attempt >= args.max_retries:
                raise

            sleep_for = retry_after_seconds(error, args.base_sleep, attempt)
            print(
                f"HTTP {status_code}; sleeping {sleep_for:.0f}s before retry "
                f"{attempt + 1}/{args.max_retries}...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_for)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
