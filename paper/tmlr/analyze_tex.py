"""Structure analysis of the canonical ACL LaTeX source.

Prints section word counts and a float inventory with orphan detection,
so the TMLR restructure works from measured lengths rather than impressions.
"""
import re
from pathlib import Path

SRC = Path(__file__).parent / "acl_source" / "acl_latex.tex"
src = SRC.read_text(encoding="utf-8")
body = src[src.find(r"\maketitle"):]

SEC = re.compile(r"\\(sub)?section\*?\{([^}]*)\}")

def visible_words(chunk):
    chunk = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "", chunk, flags=re.S)
    chunk = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "", chunk, flags=re.S)
    chunk = re.sub(r"%.*", "", chunk)
    chunk = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", chunk)
    return [w for w in chunk.split() if any(c.isalpha() for c in w)]

parts = SEC.split(body)
print("%-50s %6s" % ("SECTION", "words"))
print("-" * 58)
cur, total = "ABSTRACT (pre-section)", 0
i = 0
while i < len(parts):
    p = parts[i]
    if i % 3 == 0:
        n = len(visible_words(p))
        if n > 3:
            print("%-50s %6d" % (cur[:50], n))
            total += n
        i += 1
    else:
        sub, name = parts[i], parts[i + 1]
        cur = ("    " if sub else "") + re.sub(r"\\[a-zA-Z]+", "", name).strip()
        i += 2
print("-" * 58)
print("%-50s %6d" % ("TOTAL BODY", total))

print("\nFLOAT INVENTORY (label -> in-text \\ref count)")
print("-" * 58)
for m in re.finditer(r"\\label\{((?:tab|fig|sec|app):[^}]*)\}", body):
    lab = m.group(1)
    n = len(re.findall(r"\\ref\{" + re.escape(lab) + r"\}", body))
    flag = "   <-- ORPHAN (never referenced)" if n == 0 else ""
    print("  %-38s %d%s" % (lab, n, flag))

print("\nAPPENDIX SECTIONS")
print("-" * 58)
app = body[body.find(r"\appendix"):] if r"\appendix" in body else ""
for m in SEC.finditer(app):
    print("  %s%s" % ("    " if m.group(1) else "", m.group(2)))
print("\nappendix present:", bool(app), "| appendix words:", len(visible_words(app)))
