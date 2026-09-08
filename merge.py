#!/usr/bin/env python3
from pathlib import Path

HEADER = """Makers — Cory Doctorow
원문: https://craphound.com/makers/
라이선스: CC BY-NC-SA 3.0 US
이 번역문 역시 동일한 조건으로 배포됩니다.
"""


def main() -> None:
    project_root = Path(__file__).resolve().parent
    out_dir = project_root / "out"
    output_path = project_root / "makers-ko.md"

    if not out_dir.exists():
        output_path.write_text(HEADER + "\n", encoding="utf-8")
        print("출력 파일을 생성했습니다: makers-ko.md")
        return

    chunk_files = sorted(out_dir.glob("*.txt"), key=lambda p: int(p.stem))
    contents = []
    for chunk_file in chunk_files:
        text = chunk_file.read_text(encoding="utf-8").strip()
        if text:
            contents.append(text)

    merged = HEADER + "\n\n" + "\n\n".join(contents) + "\n"
    output_path.write_text(merged, encoding="utf-8")
    print(f"완성: makers-ko.md ({len(contents)}개 조각)")


if __name__ == "__main__":
    main()
