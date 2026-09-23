"""Compose the three inference screenshots into an IEEE two-column figure."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
OUTPUT = FIGURES / "inference_state_machine.png"


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_image(path: Path, width: int, height: int):
    source = Image.open(path).convert("RGB")
    scale = min(width / source.width, height / source.height)
    resized = source.resize((int(source.width * scale), int(source.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def arrow(draw, start, end, color="#374151", width=10):
    draw.line((start, end), fill=color, width=width)
    x, y = end
    draw.polygon([(x, y), (x - 28, y - 18), (x - 28, y + 18)], fill=color)


def main():
    canvas = Image.new("RGB", (3600, 1500), "white")
    draw = ImageDraw.Draw(canvas)
    panel_font = font(39, bold=True)
    body_font = font(30)
    small_font = font(27)

    margin, gap = 55, 35
    panel_w = (3600 - 2 * margin - 2 * gap) // 3
    panel_h = 820
    panels = [
        ("(a) Visual bootstrap", "inference_warmup.jpg", "#2563eb"),
        ("(b) Bounded Jacobian approach", "inference_jacobian.jpg", "#d97706"),
        ("(c) PPO local tracking", "inference_ppo.jpg", "#059669"),
    ]
    positions = [(margin + i * (panel_w + gap), 30) for i in range(3)]
    for (label, filename, color), (x, y) in zip(panels, positions):
        draw.text((x + 8, y), label, font=panel_font, fill="#111827")
        image = fit_image(FIGURES / filename, panel_w, panel_h - 60)
        canvas.paste(image, (x, y + 55))
        draw.rectangle((x, y + 55, x + panel_w, y + panel_h), outline="#64748b", width=3)

    y = 930
    box_gap = 75
    box_w = (3600 - 2 * margin - 2 * box_gap) // 3
    boxes = [
        ("Frames 1–5", "Raw RGB-D measurements", "Hold observation pose", "#eaf2fb", "#4f6f8f"),
        ("Frame 6 onward", "Kalman correction + causal spline", "Reconstruct Cage; Jacobian at 100 Hz", "#fbf2e8", "#9a6a32"),
        ("Local-entry event", "e_p < 15 mm and e_R < 45°", "Latch PPO tracking at 100 Hz", "#eaf6ef", "#4f8064"),
    ]
    for index, (head, line1, line2, fill, outline) in enumerate(boxes):
        bx = margin + index * (box_w + box_gap)
        by, bh = y, 310
        draw.rectangle((bx, by, bx + box_w, by + bh), fill=fill, outline=outline, width=4)
        draw.text((bx + 25, by + 28), head, font=panel_font, fill=outline)
        draw.text((bx + 25, by + 105), line1, font=body_font, fill="#111827")
        draw.text((bx + 25, by + 175), line2, font=small_font, fill="#374151")
        if index < 2:
            arrow(draw, (bx + box_w + 12, by + bh // 2),
                  (bx + box_w + box_gap - 12, by + bh // 2), color="#64748b", width=7)

    draw.text((1800, 1370),
              "Target motion and RGB-D acquisition remain active in all controller modes.",
              font=body_font, fill="#334155", anchor="ma")
    canvas.save(OUTPUT, quality=95, dpi=(400, 400))
    print(OUTPUT)


if __name__ == "__main__":
    main()
