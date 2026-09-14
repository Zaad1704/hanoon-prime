#!/usr/bin/env python3
"""Collapse every multi-line docstring in phase13.py to a one-line first
sentence while leaving all code bytes untouched. Deterministic transform."""
from pathlib import Path

P = Path("src/hanoon_prime/phase13.py")


def collapse(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        stripped = ln.strip()
        # single-line inline docstring ("""...""") -> keep as-is
        if stripped.startswith('"""') and stripped.count('"""') >= 2:
            out.append(ln)
            i += 1
            continue
        # opener of a multi-line docstring (ends without closing """)
        if stripped.startswith('"""') and not stripped.endswith('"""'):
            indent = ln[: len(ln) - len(ln.lstrip())]
            body: list[str] = []
            j = i + 1
            while j < n:
                inner = lines[j]
                if inner.strip().endswith('"""'):
                    head = ln.strip()[3:]
                    tail = inner.strip()[:-3]
                    body_text = " ".join((head, tail)).strip()
                    if body_text:
                        first_sent = body_text.split(". ")[0]
                        if not first_sent.endswith("."):
                            first_sent += "."
                        out.append(f'{indent}"""{first_sent}"""')
                    else:
                        # body was only on the closing line already handled
                        pass
                    i = j + 1
                    break
                body.append(inner.strip())
                j += 1
            else:
                # unterminated — shouldn't happen; keep verbatim
                out.append(ln)
                i += 1
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)


def main() -> int:
    before = len(P.read_text().split("\n"))
    text = P.read_text()
    for _ in range(5):
        collapsed = collapse(text)
        if collapsed == text:
            break
        text = collapsed
    P.write_text(text)
    after = len(text.split("\n"))
    print(f"phase13.py: {before} -> {after} lines")
    return 0 if after <= 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
