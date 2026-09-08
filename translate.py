#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
from pathlib import Path

try:
    from anthropic import Anthropic
    from anthropic import APIStatusError, RateLimitError
except ImportError:  # pragma: no cover
    Anthropic = None
    APIStatusError = RateLimitError = Exception


MODEL = "claude-opus-5"
SYSTEM_PROMPT = (
    "당신은 문학 번역가입니다. 아래 번역 방침을 엄격히 따라 영문 소설을\n"
    "한국어로 옮깁니다. 번역문만 출력하고, 설명이나 해설은 붙이지 않습니다."
)
REQUEST_DELAY_SECONDS = 3
MAX_RETRIES = 3


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Makers 번역 파이프라인")
    parser.add_argument("--only", type=int, help="특정 조각 번호만 처리합니다. 예: --only 12")
    parser.add_argument("--range", help="처리 범위를 지정합니다. 예: --range 10-20")
    parser.add_argument("--dry-run", action="store_true", help="실제 API 호출 없이 처리 대상만 출력합니다.")
    return parser.parse_args()


def parse_range(value: str) -> tuple[int, int]:
    if not value or "-" not in value:
        raise ValueError("--range 형식은 10-20 처럼 시작-끝이어야 합니다.")
    start_text, end_text = value.split("-", 1)
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:  # pragma: no cover
        raise ValueError("--range 값은 정수여야 합니다.") from exc
    if start > end:
        raise ValueError("--range의 시작 번호가 끝 번호보다 큽니다.")
    return start, end


def chunk_numbers(only: int | None, range_text: str | None) -> list[int]:
    if only is not None:
        return [only]
    if range_text:
        start, end = parse_range(range_text)
        return list(range(start, end + 1))
    return list(range(1, 49))


def load_policy() -> str:
    policy_path = Path("방침.md")
    if not policy_path.exists():
        raise FileNotFoundError("방침.md가 없습니다. 프로젝트 루트에서 실행해 주세요.")
    return policy_path.read_text(encoding="utf-8")


def read_previous_context(out_dir: Path, chunk_no: int) -> str:
    prev_no = chunk_no - 1
    prev_path = out_dir / f"{prev_no:03d}.txt"
    if not prev_path.exists():
        return ""
    text = prev_path.read_text(encoding="utf-8")
    if not text.strip():
        return ""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    if not paragraphs:
        return ""
    context = paragraphs[-3:]
    return "\n\n".join(context)


def build_prompt(policy_text: str, previous_context: str, source_text: str) -> str:
    parts = ["# 번역 방침", "", policy_text.strip(), "", "---", ""]

    if previous_context.strip():
        parts.extend([
            "# 직전 대목의 끝부분 (문맥 참고용, 번역하지 말 것)",
            "",
            previous_context.strip(),
            "",
            "---",
            "",
        ])

    parts.extend([
        "# 번역할 원문",
        "",
        "조각 전체를 한 번에 번역하십시오. 장면 단위로 끊지 마십시오.",
        "",
        source_text,
    ])
    return "\n".join(parts)


class MaxTokensExceededError(RuntimeError):
    def __init__(self, message: str = "API 응답이 max_tokens로 종료되었습니다."):
        super().__init__(message)


def write_output_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fetch_translation(client: Anthropic, prompt: str) -> str:
    text_parts: list[str] = []
    final_stop_reason = None

    with client.messages.stream(
        model=MODEL,
        max_tokens=32000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text_block in stream.text_stream:
            text_parts.append(text_block)

        final_message = stream.get_final_message()
        final_stop_reason = getattr(final_message, "stop_reason", None)

        content = getattr(final_message, "content", [])
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
            elif hasattr(block, "text"):
                text_parts.append(str(block.text))

    translation = "".join(text_parts).strip()
    if final_stop_reason == "max_tokens":
        raise MaxTokensExceededError("경고: API 응답이 max_tokens로 종료되었습니다. 조각을 더 작게 나누거나 max_tokens를 늘려야 합니다.")

    if not translation or len(translation) < 100:
        raise ValueError("응답이 비어 있거나 100자 미만입니다.")
    return translation


def sleep_for_retry(status_code: int | None, exc: Exception | None = None) -> float:
    if status_code == 429:
        retry_after = None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        if hasattr(headers, "get"):
            retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                return float(retry_after)
            except (TypeError, ValueError):
                pass
    return 0.0


def call_with_retry(client: Anthropic, prompt: str) -> str:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return fetch_translation(client, prompt)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            is_rate_limit = status_code == 429 or isinstance(exc, RateLimitError)
            if attempt > MAX_RETRIES:
                raise
            wait_seconds = 2 ** attempt
            if is_rate_limit:
                retry_after = sleep_for_retry(status_code, exc)
                if retry_after > 0:
                    wait_seconds = retry_after
            time.sleep(wait_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("API 호출이 실패했습니다.")


def process_chunk(client: Anthropic, chunk_no: int, chunk_path: Path, policy_text: str, out_dir: Path, failed_log: Path, dry_run: bool) -> tuple[str, int]:
    out_path = out_dir / f"{chunk_no:03d}.txt"
    if out_path.exists() and out_path.read_text(encoding="utf-8").strip():
        return "skip", 0

    previous_context = read_previous_context(out_dir, chunk_no)
    source_text = chunk_path.read_text(encoding="utf-8")
    prompt = build_prompt(policy_text, previous_context, source_text)

    if dry_run:
        print(f"[DRY-RUN] {chunk_no:03d}.txt 처리 예정")
        return "dry-run", 1

    started = time.time()
    translation = call_with_retry(client, prompt)
    write_output_text(out_path, translation)
    duration = time.time() - started
    print(f"[{chunk_no:03d}/{48}] {chunk_no:03d}.txt 처리 중... 완료 ({len(translation):,}자, {duration:.1f}초)")
    return "done", len(translation)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    load_dotenv(project_root / ".env")

    chunks_dir = project_root / "chunks"
    out_dir = project_root / "out"
    failed_log = project_root / "failed.log"
    out_dir.mkdir(exist_ok=True)

    try:
        policy_text = load_policy()
    except FileNotFoundError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    target_numbers = chunk_numbers(args.only, args.range)
    if args.only is not None:
        target_numbers = [args.only]
    if args.range:
        target_numbers = chunk_numbers(None, args.range)

    completed_count = 0
    for number in target_numbers:
        out_path = out_dir / f"{number:03d}.txt"
        if out_path.exists() and out_path.read_text(encoding="utf-8").strip():
            completed_count += 1

    total_chunks = len(target_numbers)
    print(f"전체 조각 수: {total_chunks}")
    print(f"이미 완료된 수: {completed_count}")

    if args.dry_run:
        for number in target_numbers:
            chunk_path = chunks_dir / f"{number:03d}.txt"
            if chunk_path.exists():
                print(f"[DRY-RUN] {number:03d}.txt 처리 예정")
        return 0

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("오류: ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.", file=sys.stderr)
        return 1

    if Anthropic is None:
        print("오류: anthropic 패키지가 설치되지 않았습니다.", file=sys.stderr)
        return 1

    client = Anthropic(api_key=api_key)

    processed = 0
    skipped = 0
    failed = 0
    start = time.time()

    allow_reprocess = args.only is not None

    try:
        for chunk_no in target_numbers:
            chunk_path = chunks_dir / f"{chunk_no:03d}.txt"
            if not chunk_path.exists():
                print(f"[{chunk_no:03d}] 파일이 없어 건너뜁니다.")
                continue

            time.sleep(REQUEST_DELAY_SECONDS)

            try:
                out_path = out_dir / f"{chunk_no:03d}.txt"
                if out_path.exists() and out_path.read_text(encoding="utf-8").strip() and not allow_reprocess:
                    skipped += 1
                    continue

                previous_context = read_previous_context(out_dir, chunk_no)
                source_text = chunk_path.read_text(encoding="utf-8")
                prompt = build_prompt(policy_text, previous_context, source_text)
                translation = call_with_retry(client, prompt)
                write_output_text(out_path, translation)
                processed += 1
                duration = time.time() - start
                print(f"[{chunk_no:03d}/{48}] {chunk_no:03d}.txt 처리 중... 완료 ({len(translation):,}자, {duration:.1f}초)")
            except KeyboardInterrupt:
                print("\n중단 요청을 받았습니다. 이미 저장된 파일은 유지됩니다.")
                raise
            except Exception as exc:  # noqa: BLE001
                failed += 1
                with failed_log.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"{chunk_no:03d}\n")

                if isinstance(exc, MaxTokensExceededError):
                    print(f"[{chunk_no:03d}] 경고: max_tokens로 종료되어 실패했습니다. 해당 조각을 건너뜁니다.", file=sys.stderr)
                else:
                    print(f"[{chunk_no:03d}] 실패: {exc}", file=sys.stderr)
                continue
    except KeyboardInterrupt:
        print("중단됨. 프로그램을 종료합니다.")
        return 130

    total_seconds = time.time() - start
    print("\n번역 완료 요약")
    print(f"처리한 조각 수: {processed}")
    print(f"건너뛴 수: {skipped}")
    print(f"실패한 수: {failed}")
    print(f"총 소요 시간: {total_seconds:.1f}초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
