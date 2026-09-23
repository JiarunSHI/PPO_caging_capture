from pathlib import Path
import argparse

import markdown


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "动态Cage论文初稿.md"
OUTPUT = ROOT / "动态Cage论文初稿.html"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    first_line = source.read_text(encoding="utf-8").splitlines()[0]
    title = first_line[2:] if first_line.startswith("# ") else first_line
    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    body = body.replace(
        "<!-- IEEE作者信息区域预留：姓名、单位、邮箱暂不填写 -->",
        '<div class="author-placeholder" aria-label="Author information reserved"></div>',
    ).replace(
        "<!-- IEEE author names, affiliations, and e-mail addresses intentionally left blank -->",
        '<div class="author-placeholder" aria-label="Author information reserved"></div>',
    )
    main_heading = next(
        (heading for heading in ("<h2>I. INTRODUCTION</h2>", "<h2>I. 引言</h2>") if heading in body),
        None,
    )
    if main_heading:
        body = body.replace(main_heading, f'<div class="ieee-columns">{main_heading}', 1)
        body += "</div>"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
@page {{ size: Letter; margin: 19mm 15.9mm 25.4mm; }}
body {{
  max-width: 7.25in; margin: 0 auto; color: #000;
  font-family: "Times New Roman", "Noto Serif CJK SC", "SimSun", serif;
  font-size: 10pt; line-height: 1.12; text-align: justify;
}}
h1 {{ font-size: 24pt; font-weight: 400; text-align: center; margin: .35in 0 .08in; line-height: 1.08; }}
.author-placeholder {{ height: .42in; }}
.ieee-columns {{ column-count: 2; column-gap: .25in; column-fill: auto; }}
h2 {{ font-size: 10pt; text-transform: uppercase; text-align: center; margin: 1.2em 0 .45em; break-after: avoid; }}
h3 {{ font-size: 10pt; font-style: italic; margin: .85em 0 .25em; break-after: avoid; }}
h4 {{ font-size: 10pt; font-style: italic; margin: .65em 0 .2em; break-after: avoid; }}
p {{ margin: .2em 0; text-indent: 1em; }}
body > p:nth-of-type(-n+2) {{ text-indent: 0; }}
p:has(> img) {{ column-span: all; break-inside: avoid; text-indent: 0; }}
img {{ display: block; max-width: 7.1in; max-height: 7.4in; margin: .8em auto .2em; object-fit: contain; }}
img[src*="system_pipeline_en"], img[src*="ppo_architecture_en"] {{ max-width: 84%; }}
img[src*="est_expansion_flow_en"], img[src*="inference_state_machine"] {{ max-width: 90%; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8pt; margin: .55em 0; break-inside: avoid; }}
th, td {{ border: 1px solid #7f8c99; padding: .38em .5em; text-align: center; }}
th {{ background: #fff; font-weight: bold; }}
code {{ font-family: "DejaVu Sans Mono", monospace; font-size: .92em; }}
blockquote {{ border-left: 4px solid #7c93aa; margin-left: 0; padding-left: 1em; color: #425466; }}
ol, ul {{ padding-left: 1.6em; }}
@media screen and (max-width: 850px) {{
  body {{ padding: 0 18px; }}
  .ieee-columns {{ column-count: 1; }}
  img {{ max-width: 100%; }}
}}
</style>
<script>
window.MathJax = {{tex: {{tags: 'ams', inlineMath: [['$', '$']], displayMath: [['$$','$$']]}}}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>{body}</body>
</html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
