"""strip_comments.py — removes comments and docstrings from all .py files in scripts/"""
import ast
import io
import os
import tokenize

SCRIPTS_DIR = "scripts"


def strip_comments(source: str) -> str:
    result = []
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0

    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for tok in tokens:
        token_type, token_string, start, end, line = tok
        start_line, start_col = start

        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            result.append(" " * (start_col - last_col))

        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING and prev_toktype in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL) and start_col == 0:
            pass
        else:
            result.append(token_string)

        prev_toktype = token_type
        last_col = end[1]
        last_lineno = end[0]

    return "".join(result)


def strip_docstrings(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body[0].value.value = ""
    return ast.unparse(tree)


def clean_blank_lines(source: str) -> str:
    lines = source.split("\n")
    cleaned = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        cleaned.append(line)
    return "\n".join(cleaned)


def main():
    for filename in sorted(os.listdir(SCRIPTS_DIR)):
        if not filename.endswith(".py"):
            continue
        path = os.path.join(SCRIPTS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()

        try:
            no_docstrings = strip_docstrings(source)
            cleaned = strip_comments(no_docstrings)
            cleaned = clean_blank_lines(cleaned)
        except Exception as e:
            print(f"[SKIPPED] {filename}: {e}")
            continue

        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"[CLEANED] {filename}")


if __name__ == "__main__":
    main()
