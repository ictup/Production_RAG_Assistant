from collections.abc import Iterable

NOISE_LINES = {
    "[toc]",
    "table of contents",
    "back to top",
    "edit this page",
}


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")


def is_code_fence(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def clean_text(text: str, noise_lines: Iterable[str] = NOISE_LINES) -> str:
    normalized = normalize_newlines(text)
    noise = {line.casefold() for line in noise_lines}

    cleaned_lines: list[str] = []
    blank_count = 0
    in_code_block = False

    for raw_line in normalized.split("\n"):
        line = raw_line.rstrip()

        if is_code_fence(line):
            in_code_block = not in_code_block
            cleaned_lines.append(line)
            blank_count = 0
            continue

        if in_code_block:
            cleaned_lines.append(line)
            continue

        stripped = line.strip()
        if stripped.casefold() in noise:
            continue

        if stripped == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append("")
            continue

        blank_count = 0
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
