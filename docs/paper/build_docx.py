"""Build an editable DOCX draft without external Python dependencies.

The Markdown source remains authoritative for LaTeX equations.  Display
equations are inserted as centered Cambria Math text so they stay editable.
"""

from __future__ import annotations

import argparse
import html
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "动态Cage论文初稿.md"
OUTPUT = ROOT / "build" / "动态Cage论文初稿.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC = "http://purl.org/dc/elements/1.1/"
DCT = "http://purl.org/dc/terms/"
XSI = "http://www.w3.org/2001/XMLSchema-instance"

for prefix, uri in (("w", W), ("r", R), ("wp", WP), ("a", A), ("pic", PIC), ("m", M)):
    ET.register_namespace(prefix, uri)


def q(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}"


def add_run(paragraph, text: str, bold=False, italic=False, code=False, size=None):
    run = ET.SubElement(paragraph, q(W, "r"))
    props = ET.SubElement(run, q(W, "rPr"))
    fonts = ET.SubElement(props, q(W, "rFonts"))
    fonts.set(q(W, "ascii"), "Consolas" if code else "Times New Roman")
    fonts.set(q(W, "hAnsi"), "Consolas" if code else "Times New Roman")
    fonts.set(q(W, "eastAsia"), "宋体")
    if bold:
        ET.SubElement(props, q(W, "b"))
    if italic:
        ET.SubElement(props, q(W, "i"))
    if size:
        sz = ET.SubElement(props, q(W, "sz"))
        sz.set(q(W, "val"), str(size))
        szcs = ET.SubElement(props, q(W, "szCs"))
        szcs.set(q(W, "val"), str(size))
    node = ET.SubElement(run, q(W, "t"))
    if text.startswith(" ") or text.endswith(" "):
        node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    node.text = text
    return run


def add_inline(paragraph, text: str):
    token = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)")
    cursor = 0
    for match in token.finditer(text):
        if match.start() > cursor:
            add_run(paragraph, text[cursor : match.start()])
        value = match.group(0)
        if value.startswith("`"):
            add_run(paragraph, value[1:-1], code=True)
        elif value.startswith("**"):
            add_run(paragraph, value[2:-2], bold=True)
        else:
            add_run(paragraph, value[1:-1], italic=True)
        cursor = match.end()
    if cursor < len(text):
        add_run(paragraph, text[cursor:])


def add_paragraph(body, text="", style=None, align=None, first_line=True):
    p = ET.SubElement(body, q(W, "p"))
    ppr = ET.SubElement(p, q(W, "pPr"))
    if style:
        pstyle = ET.SubElement(ppr, q(W, "pStyle"))
        pstyle.set(q(W, "val"), style)
    if align:
        jc = ET.SubElement(ppr, q(W, "jc"))
        jc.set(q(W, "val"), align)
    if first_line and not style:
        ind = ET.SubElement(ppr, q(W, "ind"))
        ind.set(q(W, "firstLine"), "200")
    spacing = ET.SubElement(ppr, q(W, "spacing"))
    spacing.set(q(W, "line"), "240")
    spacing.set(q(W, "lineRule"), "auto")
    if text:
        add_inline(p, text)
    return p


def math_run(text, plain=False, bold=False):
    run = ET.Element(q(M, "r"))
    props = ET.SubElement(run, q(M, "rPr"))
    style = ET.SubElement(props, q(M, "sty"))
    style.set(q(M, "val"), "b" if bold else ("p" if plain else "i"))
    node = ET.SubElement(run, q(M, "t"))
    node.text = text
    return run


def append_math(parent, elements):
    for element in elements:
        parent.append(element)


class LatexOmmlParser:
    SYMBOLS = {
        "Delta": "Δ", "gamma": "γ", "lambda": "λ", "theta": "θ",
        "pi": "π", "rho": "ρ", "sigma": "σ", "epsilon": "ε",
        "phi": "φ", "mu": "μ", "xi": "ξ", "nu": "ν", "kappa": "κ",
        "alpha": "α", "beta": "β", "delta": "δ", "ell": "ℓ",
        "in": "∈", "leq": "≤", "geq": "≥", "times": "×",
        "sim": "∼", "sum": "∑", "pm": "±", "quad": "  ",
        "qquad": "    ", "cdot": "·", "top": "⊤",
    }
    FUNCTIONS = {"clip", "interp", "LPF", "Std", "diag", "exp", "log", "tanh", "atanh", "min", "max", "argmin"}

    def __init__(self, source):
        self.source = source
        self.index = 0

    def parse(self, stop=None, plain=False, bold=False):
        result = []
        while self.index < len(self.source):
            if stop and self.source.startswith(stop, self.index):
                self.index += len(stop)
                break
            atom = self.parse_atom(plain=plain, bold=bold)
            if not atom:
                continue
            subscript = None
            superscript = None
            while self.index < len(self.source) and self.source[self.index] in "_^":
                marker = self.source[self.index]
                self.index += 1
                script = self.parse_group()
                if marker == "_":
                    subscript = script
                else:
                    superscript = script
            if subscript is not None or superscript is not None:
                atom = [self.make_script(atom, subscript, superscript)]
            result.extend(atom)
        return result

    def parse_group(self):
        while self.index < len(self.source) and self.source[self.index].isspace():
            self.index += 1
        if self.index < len(self.source) and self.source[self.index] == "{":
            self.index += 1
            return self.parse(stop="}")
        return self.parse_atom()

    def parse_atom(self, plain=False, bold=False):
        if self.index >= len(self.source):
            return []
        if self.source.startswith(r"\begin{bmatrix}", self.index):
            return [self.parse_matrix()]
        char = self.source[self.index]
        if char == "{":
            self.index += 1
            return self.parse(stop="}", plain=plain, bold=bold)
        if char == "}":
            self.index += 1
            return []
        if char != "\\":
            self.index += 1
            if char == "&":
                char = " "
            return [math_run(char, plain=plain, bold=bold)]

        self.index += 1
        if self.index < len(self.source) and self.source[self.index] == "\\":
            self.index += 1
            return [math_run(" ", plain=True)]
        start = self.index
        while self.index < len(self.source) and self.source[self.index].isalpha():
            self.index += 1
        if start == self.index:
            command = self.source[self.index] if self.index < len(self.source) else ""
            self.index += int(bool(command))
            return [] if command in "!,; " else [math_run("‖" if command == "|" else command, plain=True)]
        command = self.source[start:self.index]
        if command in {"left", "right", "big", "Big", "bigl", "bigr"}:
            return []
        if command == "frac":
            numerator = self.parse_group()
            denominator = self.parse_group()
            fraction = ET.Element(q(M, "f"))
            num = ET.SubElement(fraction, q(M, "num"))
            den = ET.SubElement(fraction, q(M, "den"))
            append_math(num, numerator)
            append_math(den, denominator)
            return [fraction]
        if command == "bar":
            bar = ET.Element(q(M, "bar"))
            bar_props = ET.SubElement(bar, q(M, "barPr"))
            position = ET.SubElement(bar_props, q(M, "pos"))
            position.set(q(M, "val"), "top")
            element = ET.SubElement(bar, q(M, "e"))
            append_math(element, self.parse_group())
            return [bar]
        if command in {"mathbf", "boldsymbol"}:
            group = self.extract_group_text()
            return LatexOmmlParser(group).parse(bold=True)
        if command == "mathcal":
            group = self.extract_group_text()
            return LatexOmmlParser(group).parse()
        if command in {"mathrm", "mathsf", "operatorname", "text"}:
            group = self.extract_group_text()
            return LatexOmmlParser(group).parse(plain=True)
        if command == "mathbb":
            group = self.extract_group_text()
            mapped = {"R": "ℝ", "N": "ℕ", "Z": "ℤ"}.get(group, group)
            return [math_run(mapped, plain=True)]
        if command in self.FUNCTIONS:
            return [math_run(command, plain=True)]
        if command in self.SYMBOLS:
            return [math_run(self.SYMBOLS[command], plain=True)]
        return [math_run(command, plain=True)]

    def extract_group_text(self):
        while self.index < len(self.source) and self.source[self.index].isspace():
            self.index += 1
        if self.index >= len(self.source) or self.source[self.index] != "{":
            return ""
        self.index += 1
        start = self.index
        depth = 1
        while self.index < len(self.source) and depth:
            if self.source[self.index] == "{":
                depth += 1
            elif self.source[self.index] == "}":
                depth -= 1
            self.index += 1
        return self.source[start:self.index - 1]

    def parse_matrix(self):
        self.index += len(r"\begin{bmatrix}")
        end = self.source.find(r"\end{bmatrix}", self.index)
        if end < 0:
            end = len(self.source)
        content = self.source[self.index:end]
        self.index = end + len(r"\end{bmatrix}")
        delimiter = ET.Element(q(M, "d"))
        delimiter_props = ET.SubElement(delimiter, q(M, "dPr"))
        begin_char = ET.SubElement(delimiter_props, q(M, "begChr"))
        begin_char.set(q(M, "val"), "[")
        end_char = ET.SubElement(delimiter_props, q(M, "endChr"))
        end_char.set(q(M, "val"), "]")
        element = ET.SubElement(delimiter, q(M, "e"))
        matrix = ET.SubElement(element, q(M, "m"))
        for row_text in re.split(r"\\\\", content):
            row = ET.SubElement(matrix, q(M, "mr"))
            for cell_text in row_text.split("&"):
                cell = ET.SubElement(row, q(M, "e"))
                append_math(cell, LatexOmmlParser(cell_text.strip()).parse())
        return delimiter

    @staticmethod
    def make_script(base, subscript, superscript):
        if subscript is not None and superscript is not None:
            node = ET.Element(q(M, "sSubSup"))
            base_node = ET.SubElement(node, q(M, "e"))
            sub_node = ET.SubElement(node, q(M, "sub"))
            sup_node = ET.SubElement(node, q(M, "sup"))
            append_math(base_node, base)
            append_math(sub_node, subscript)
            append_math(sup_node, superscript)
            return node
        tag = "sSub" if subscript is not None else "sSup"
        node = ET.Element(q(M, tag))
        base_node = ET.SubElement(node, q(M, "e"))
        script_node = ET.SubElement(node, q(M, "sub" if subscript is not None else "sup"))
        append_math(base_node, base)
        append_math(script_node, subscript if subscript is not None else superscript)
        return node


def add_equation(body, lines):
    source = " ".join(part.strip() for part in lines if part.strip())
    tag_match = re.search(r"\\tag\{([^}]+)\}", source)
    number = tag_match.group(1) if tag_match else None
    source = re.sub(r"\\tag\{[^}]+\}", "", source).strip().rstrip(".")
    paragraph = ET.SubElement(body, q(M, "oMathPara"))
    paragraph_props = ET.SubElement(paragraph, q(M, "oMathParaPr"))
    justify = ET.SubElement(paragraph_props, q(M, "jc"))
    justify.set(q(M, "val"), "centerGroup")
    equation = ET.SubElement(paragraph, q(M, "oMath"))
    append_math(equation, LatexOmmlParser(source).parse())
    if number:
        equation.append(math_run("    ({})".format(number), plain=True))


def add_image(body, image_path: Path, rid: str, image_index: int, wide: bool = False):
    with Image.open(image_path) as image:
        width_px, height_px = image.size
    if wide:
        scientific_sizes = {
            "system_pipeline_en.png": (5.9, 3.5),
            "est_expansion_flow_en.png": (6.2, 3.0),
            "ppo_architecture_en.png": (5.6, 3.2),
            "inference_state_machine.png": (6.3, 3.0),
            "training_overview.png": (6.45, 4.9),
            "stage_summary.png": (6.25, 4.9),
            "state_arc_prior_relationships.png": (5.8, 4.6),
        }
        max_width, max_height = scientific_sizes.get(image_path.name, (6.3, 4.8))
    else:
        max_width, max_height = 3.35, 4.7
    scale = min(max_width / width_px, max_height / height_px)
    width_in = width_px * scale
    height_in = height_px * scale
    cx = int(width_in * 914400)
    cy = int(height_in * 914400)

    p = add_paragraph(body, align="center", first_line=False)
    run = ET.SubElement(p, q(W, "r"))
    drawing = ET.SubElement(run, q(W, "drawing"))
    inline = ET.SubElement(drawing, q(WP, "inline"))
    for name in ("distT", "distB", "distL", "distR"):
        inline.set(name, "0")
    extent = ET.SubElement(inline, q(WP, "extent"))
    extent.set("cx", str(cx))
    extent.set("cy", str(cy))
    effect = ET.SubElement(inline, q(WP, "effectExtent"))
    for name in ("l", "t", "r", "b"):
        effect.set(name, "0")
    doc_pr = ET.SubElement(inline, q(WP, "docPr"))
    doc_pr.set("id", str(image_index))
    doc_pr.set("name", image_path.name)
    frame_pr = ET.SubElement(inline, q(WP, "cNvGraphicFramePr"))
    locks = ET.SubElement(frame_pr, q(A, "graphicFrameLocks"))
    locks.set("noChangeAspect", "1")
    graphic = ET.SubElement(inline, q(A, "graphic"))
    graphic_data = ET.SubElement(graphic, q(A, "graphicData"))
    graphic_data.set("uri", "http://schemas.openxmlformats.org/drawingml/2006/picture")
    pic = ET.SubElement(graphic_data, q(PIC, "pic"))
    nv = ET.SubElement(pic, q(PIC, "nvPicPr"))
    cnv = ET.SubElement(nv, q(PIC, "cNvPr"))
    cnv.set("id", "0")
    cnv.set("name", image_path.name)
    ET.SubElement(nv, q(PIC, "cNvPicPr"))
    fill = ET.SubElement(pic, q(PIC, "blipFill"))
    blip = ET.SubElement(fill, q(A, "blip"))
    blip.set(q(R, "embed"), rid)
    stretch = ET.SubElement(fill, q(A, "stretch"))
    ET.SubElement(stretch, q(A, "fillRect"))
    shape = ET.SubElement(pic, q(PIC, "spPr"))
    transform = ET.SubElement(shape, q(A, "xfrm"))
    off = ET.SubElement(transform, q(A, "off"))
    off.set("x", "0")
    off.set("y", "0")
    ext = ET.SubElement(transform, q(A, "ext"))
    ext.set("cx", str(cx))
    ext.set("cy", str(cy))
    geom = ET.SubElement(shape, q(A, "prstGeom"))
    geom.set("prst", "rect")
    ET.SubElement(geom, q(A, "avLst"))


def add_table(body, rows):
    table = ET.SubElement(body, q(W, "tbl"))
    props = ET.SubElement(table, q(W, "tblPr"))
    style = ET.SubElement(props, q(W, "tblStyle"))
    style.set(q(W, "val"), "TableGrid")
    width = ET.SubElement(props, q(W, "tblW"))
    width.set(q(W, "w"), "0")
    width.set(q(W, "type"), "auto")
    for row_index, row in enumerate(rows):
        tr = ET.SubElement(table, q(W, "tr"))
        for cell_text in row:
            tc = ET.SubElement(tr, q(W, "tc"))
            p = add_paragraph(tc, align="center", first_line=False)
            add_run(p, cell_text.strip(), bold=row_index == 0, size=16)


def parse_markdown(source: Path):
    lines = source.read_text(encoding="utf-8").splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.strip().startswith("<!--"):
            while index < len(lines) and "-->" not in lines[index]:
                index += 1
            blocks.append(("author_placeholder",))
            index += 1
            continue
        if line.strip() == "$$":
            equation = []
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                equation.append(lines[index])
                index += 1
            blocks.append(("equation", equation))
            index += 1
            continue
        image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line.strip())
        if image_match:
            blocks.append(("image", image_match.group(1), image_match.group(2)))
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            blocks.append(("table", rows))
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            blocks.append(("heading", len(heading.group(1)), heading.group(2)))
            index += 1
            continue
        if re.match(r"^\d+\.\s+", line):
            blocks.append(("number", re.sub(r"^(\d+)\.\s+", r"\1. ", line)))
            index += 1
            continue
        if line.startswith("- "):
            blocks.append(("bullet", line[2:]))
            index += 1
            continue
        paragraph = [line.strip()]
        index += 1
        while index < len(lines) and lines[index].strip() and not re.match(
            r"^(#{1,4})\s+|^\$\$$|^!\[|^\||^\d+\.\s+|^- ", lines[index]
        ):
            paragraph.append(lines[index].strip())
            index += 1
        blocks.append(("paragraph", " ".join(paragraph)))
    return blocks


def build_document(source: Path):
    document = ET.Element(q(W, "document"))
    body = ET.SubElement(document, q(W, "body"))
    relationships = []
    media = []
    image_index = 1
    main_columns_started = False
    wide_figure_open = False
    in_references = False
    table_caption_pending = False

    for block in parse_markdown(source):
        kind = block[0]
        if kind == "heading":
            level, text = block[1], block[2]
            if level == 2 and text.startswith("I. ") and not main_columns_started:
                add_section_break(body, columns=1)
                main_columns_started = True
            if level == 2 and text == "REFERENCES":
                in_references = True
            style = "Title" if level == 1 else f"Heading{min(level - 1, 3)}"
            add_paragraph(body, text, style=style, align="center" if level == 1 else None, first_line=False)
        elif kind == "author_placeholder":
            p = add_paragraph(body, align="center", first_line=False)
            ppr = p.find(q(W, "pPr"))
            spacing = ppr.find(q(W, "spacing"))
            spacing.set(q(W, "after"), "480")
            add_run(p, "\u00a0", size=20)
        elif kind == "paragraph":
            text = block[1]
            is_abstract = text.startswith("**Abstract—**") or text.startswith("**Index Terms—**")
            is_figure_caption = text.startswith("**Fig. ")
            is_table_label = text.startswith("**TABLE ")
            if is_abstract:
                add_paragraph(body, text, style="Abstract", first_line=False)
            elif is_figure_caption:
                add_paragraph(body, text, style="FigureCaption", align="center", first_line=False)
                if wide_figure_open:
                    add_section_break(body, columns=1)
                    wide_figure_open = False
            elif is_table_label or table_caption_pending:
                add_paragraph(body, text, style="TableCaption", align="center", first_line=False)
                table_caption_pending = is_table_label
            elif in_references:
                add_paragraph(body, text, style="Reference", first_line=False)
            else:
                no_indent = text.startswith("**摘要—**") or text.startswith("**关键词—**")
                add_paragraph(body, text, first_line=not no_indent)
        elif kind in {"number", "bullet"}:
            prefix = "• " if kind == "bullet" else ""
            add_paragraph(body, prefix + block[1], first_line=False)
        elif kind == "equation":
            add_equation(body, block[1])
        elif kind == "table":
            add_table(body, block[1])
        elif kind == "image":
            path = (source.parent / block[2]).resolve()
            extension = path.suffix.lower().lstrip(".")
            extension = "jpeg" if extension in {"jpg", "jpeg"} else extension
            target_name = f"image{image_index}.{extension}"
            rid = f"rId{image_index + 1}"
            relationships.append((rid, f"media/{target_name}"))
            media.append((target_name, path.read_bytes()))
            if main_columns_started:
                add_section_break(body, columns=2)
                wide_figure_open = True
            add_image(body, path, rid, image_index, wide=main_columns_started)
            image_index += 1

    section = ET.SubElement(body, q(W, "sectPr"))
    configure_section(section, columns=2)
    return ET.tostring(document, encoding="utf-8", xml_declaration=True), relationships, media


def configure_section(section, columns: int, continuous: bool = False):
    if continuous:
        section_type = ET.SubElement(section, q(W, "type"))
        section_type.set(q(W, "val"), "continuous")
    size = ET.SubElement(section, q(W, "pgSz"))
    size.set(q(W, "w"), "12240")
    size.set(q(W, "h"), "15840")
    margin = ET.SubElement(section, q(W, "pgMar"))
    for name, value in {
        "top": "1080", "right": "900", "bottom": "1440", "left": "900",
        "header": "360", "footer": "720", "gutter": "0",
    }.items():
        margin.set(q(W, name), value)
    cols = ET.SubElement(section, q(W, "cols"))
    cols.set(q(W, "num"), str(columns))
    if columns > 1:
        cols.set(q(W, "space"), "360")


def add_section_break(body, columns: int):
    paragraph = ET.SubElement(body, q(W, "p"))
    paragraph_props = ET.SubElement(paragraph, q(W, "pPr"))
    section = ET.SubElement(paragraph_props, q(W, "sectPr"))
    configure_section(section, columns=columns, continuous=True)


def styles_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:rPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:line="240" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="80"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:sz w:val="48"/><w:szCs w:val="48"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:jc w:val="center"/><w:spacing w:before="180" w:after="80"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:rFonts w:eastAsia="黑体"/><w:b/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="140" w:after="50"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:rFonts w:eastAsia="黑体"/><w:i/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="100" w:after="40"/><w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:rFonts w:eastAsia="宋体"/><w:i/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Abstract"><w:name w:val="Abstract"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="both"/><w:spacing w:line="216" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="FigureCaption"><w:name w:val="Figure Caption"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:spacing w:before="20" w:after="90"/></w:pPr><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="TableCaption"><w:name w:val="Table Caption"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:spacing w:before="70" w:after="25"/></w:pPr><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Reference"><w:name w:val="Reference"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:hanging="240"/><w:spacing w:line="192" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Equation"><w:name w:val="Equation"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:spacing w:before="100" w:after="100"/></w:pPr><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr></w:style>
</w:styles>""".encode("utf-8")


def package_document(source: Path = SOURCE, output: Path = OUTPUT, title=None):
    document, image_relationships, media = build_document(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    if title is None:
        first_line = source.read_text(encoding="utf-8").splitlines()[0]
        title = first_line[2:] if first_line.startswith("# ") else first_line
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Default Extension="jpeg" ContentType="image/jpeg"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    rel_rows = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    ]
    for rid, target in image_relationships:
        rel_rows.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>'
        )
    document_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rel_rows)
        + "</Relationships>"
    )
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>{html.escape(title)}</dc:title><dc:creator></dc:creator><cp:lastModifiedBy></cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">2026-09-17T00:00:00Z</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">2026-09-17T00:00:00Z</dcterms:modified></cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Office Word</Application><AppVersion>16.0000</AppVersion></Properties>"""

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", styles_xml())
        archive.writestr("word/_rels/document.xml.rels", document_rels)
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)
        for name, payload in media:
            archive.writestr(f"word/media/{name}", payload)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--title")
    args = parser.parse_args()
    package_document(args.source.resolve(), args.output.resolve(), args.title)
