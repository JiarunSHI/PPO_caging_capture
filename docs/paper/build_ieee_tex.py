"""Convert the English Markdown draft to an IEEEtran conference source file."""

import re
from pathlib import Path

from build_docx import parse_markdown


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Dynamic_Cage_IEEE_Conference_Paper.md"
OUTPUT = ROOT / "Dynamic_Cage_IEEE_Conference_Paper.tex"


def strip_markdown(text):
    return text.replace("**", "").strip()


def citation_range(match):
    start, end = int(match.group(1)), int(match.group(2))
    return "\\cite{" + ",".join("ref{}".format(i) for i in range(start, end + 1)) + "}"


def escape_plain(text):
    text = re.sub(r"\[(\d+)\][–-]\[(\d+)\]", citation_range, text)
    text = re.sub(r"\[(\d+)\]", lambda m: "\\cite{ref" + m.group(1) + "}", text)
    protected = {}

    def protect(match):
        key = "@@PROTECTED{}@@".format(len(protected))
        protected[key] = match.group(0)
        return key

    text = re.sub(r"\\cite\{[^}]+\}", protect, text)
    text = text.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    text = text.replace("°", r"\textdegree{}").replace("±", r"$\pm$").replace("²", r"$^2$")
    text = text.replace("–", "--").replace("—", "---").replace("−", "-")
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def inline(text):
    pattern = re.compile(r"(\\\(.+?\\\)|`[^`]+`|\*[^*]+\*)")
    parts = []
    cursor = 0
    for match in pattern.finditer(text):
        parts.append(escape_plain(text[cursor:match.start()]))
        token = match.group(0)
        if token.startswith(r"\("):
            parts.append(token)
        elif token.startswith("`"):
            parts.append(r"\texttt{" + escape_plain(token[1:-1]).replace("_", r"\_") + "}")
        else:
            parts.append(r"\emph{" + escape_plain(token[1:-1]) + "}")
        cursor = match.end()
    parts.append(escape_plain(text[cursor:]))
    return "".join(parts)


def heading_text(text):
    return re.sub(r"^(?:[IVXLCDM]+\.|[A-Z]\.|\d+\))\s*", "", text).strip()


def main():
    blocks = parse_markdown(SOURCE)
    title = blocks[0][2]
    abstract = ""
    keywords = ""
    for block in blocks:
        if block[0] == "paragraph" and block[1].startswith("**Abstract—**"):
            abstract = strip_markdown(block[1]).replace("Abstract—", "", 1).strip()
        if block[0] == "paragraph" and block[1].startswith("**Index Terms—**"):
            keywords = strip_markdown(block[1]).replace("Index Terms—", "", 1).strip()

    out = [
        r"\documentclass[conference]{IEEEtran}",
        r"\IEEEoverridecommandlockouts",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{textcomp}",
        r"\usepackage{cite}",
        r"\graphicspath{{figures/}}",
        "",
        r"\title{" + escape_plain(title) + "}",
        r"\author{\IEEEauthorblockN{\mbox{}}\IEEEauthorblockA{\mbox{}}}",
        "",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        inline(abstract),
        r"\end{abstract}",
        r"\begin{IEEEkeywords}",
        inline(keywords),
        r"\end{IEEEkeywords}",
        "",
    ]

    pending_image = None
    table_label = None
    table_title = None
    in_references = False
    skip_front = True

    for block in blocks:
        kind = block[0]
        if skip_front:
            if kind == "heading" and block[1] == 2 and block[2].startswith("I. "):
                skip_front = False
            else:
                continue

        if kind == "heading":
            level, text = block[1], block[2]
            clean = heading_text(text)
            if level == 2 and clean == "REFERENCES":
                in_references = True
                out.extend([r"\begin{thebibliography}{00}", ""])
            elif level == 2:
                out.extend([r"\section{" + escape_plain(clean.title()) + "}", ""])
            elif level == 3:
                out.extend([r"\subsection{" + escape_plain(clean) + "}", ""])
            else:
                out.extend([r"\subsubsection{" + escape_plain(clean) + "}", ""])
        elif kind == "image":
            pending_image = block[2]
        elif kind == "paragraph":
            text = block[1]
            if pending_image and text.startswith("**Fig. "):
                caption = re.sub(r"^\*\*Fig\.\s*\d+\.\*\*\s*", "", text)
                compact_names = {
                    "figures/system_pipeline_en.png": "0.84",
                    "figures/est_expansion_flow_en.png": "0.90",
                    "figures/ppo_architecture_en.png": "0.82",
                    "figures/inference_state_machine.png": "0.90",
                }
                figure_width = compact_names.get(pending_image, "0.94")
                out.extend([
                    r"\begin{figure*}[t]",
                    r"\centering",
                    r"\includegraphics[width=" + figure_width + r"\textwidth]{" + pending_image + "}",
                    r"\caption{" + inline(caption) + "}",
                    r"\end{figure*}",
                    "",
                ])
                pending_image = None
            elif text.startswith("**TABLE "):
                table_label = strip_markdown(text)
            elif table_label and table_title is None:
                table_title = strip_markdown(text)
            elif in_references:
                match = re.match(r"^\[(\d+)\]\s*(.*)$", text)
                if match:
                    out.append(r"\bibitem{ref" + match.group(1) + "}" + inline(match.group(2)))
            else:
                out.extend([inline(text), ""])
        elif kind in {"number", "bullet"}:
            marker = r"\noindent\textbullet\ " if kind == "bullet" else ""
            out.extend([marker + inline(block[1]) + r"\par", ""])
        elif kind == "equation":
            equation = "\n".join(block[1])
            equation = re.sub(r"\\tag\{[^}]+\}", "", equation)
            out.extend([r"\begin{equation}", equation, r"\end{equation}", ""])
        elif kind == "table":
            rows = block[1]
            columns = "l" + "r" * (len(rows[0]) - 1)
            out.extend([
                r"\begin{table}[t]",
                r"\caption{" + escape_plain(table_title or table_label or "") + "}",
                r"\centering",
                r"\footnotesize",
                r"\begin{tabular}{" + columns + "}",
                r"\toprule",
            ])
            for row_index, row in enumerate(rows):
                out.append(" & ".join(inline(cell) for cell in row) + r" \\")
                if row_index == 0:
                    out.append(r"\midrule")
            out.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
            table_label = None
            table_title = None

    if in_references:
        out.append(r"\end{thebibliography}")
    out.append(r"\end{document}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
