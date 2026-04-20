from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CARD_BACKGROUND = "#fffdfa"
CANVAS_BACKGROUND = "#f6ede2"
CARD_BORDER = "#e1d4c2"
PRIMARY_TEXT = "#1f2732"
SECONDARY_TEXT = "#657384"
COMPACT_MARK = "ŷᶜᵒᵐᵖᵃᶜᵗ"
GRID = "#e6ddd0"
ACCENT = "#375f8b"
ACCENT_SOFT = "#90acc8"


@dataclass(frozen=True)
class EquationSpec:
    intercept: float
    slope: float
    row_count: int

    @property
    def upper_index(self) -> int:
        return max(0, self.row_count - 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render Euclid's sample exact-closure equation into standalone PNGs "
            "without plotting the time series."
        )
    )
    parser.add_argument(
        "--intercept",
        type=float,
        default=0.000072,
        help="Compact deterministic intercept term.",
    )
    parser.add_argument(
        "--slope",
        type=float,
        default=0.000001,
        help="Compact deterministic slope term multiplying t.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1255,
        help="Observed row count used by the exact-closure term.",
    )
    parser.add_argument(
        "--analysis-path",
        type=Path,
        help=(
            "Optional workbench analysis.json path. When provided, the script "
            "renders the full holistic equation curve from holistic_equation.chart.equation_curve."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/equation_visuals"),
        help="Directory where the rendered PNGs will be written.",
    )
    return parser.parse_args(argv)


def font(size: int, *, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    candidates: list[str]
    if bold and italic:
        candidates = [
            "/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf",
            "/System/Library/Fonts/Supplemental/Georgia Bold Italic.ttf",
            "/System/Library/Fonts/Supplemental/STIXGeneralBolIta.otf",
        ]
    elif bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/System/Library/Fonts/Supplemental/STIXGeneralBol.otf",
        ]
    elif italic:
        candidates = [
            "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
            "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
            "/System/Library/Fonts/Supplemental/STIXGeneralItalic.otf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/Supplemental/STIXGeneral.otf",
        ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def measure(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=text_font)
    return right - left, bottom - top


def draw_rounded_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(
        box,
        radius=34,
        fill=CARD_BACKGROUND,
        outline=CARD_BORDER,
        width=3,
    )


def compact_rhs(spec: EquationSpec) -> str:
    return f"{spec.intercept:.6f} + {spec.slope:.6f} · t"


def draw_text(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    text: str,
    text_font: ImageFont.ImageFont,
    fill: str = PRIMARY_TEXT,
) -> int:
    draw.text((x, y), text, font=text_font, fill=fill)
    width, _height = measure(draw, text, text_font)
    return x + width


def draw_bound_operator(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    baseline_y: int,
    upper: str,
    symbol: str,
    lower: str,
    symbol_font: ImageFont.ImageFont,
    bound_font: ImageFont.ImageFont,
    fill: str = PRIMARY_TEXT,
) -> int:
    symbol_width, symbol_height = measure(draw, symbol, symbol_font)
    upper_width, upper_height = measure(draw, upper, bound_font)
    lower_width, lower_height = measure(draw, lower, bound_font)
    block_width = max(symbol_width, upper_width, lower_width)
    symbol_x = x + (block_width - symbol_width) // 2
    upper_x = x + (block_width - upper_width) // 2
    lower_x = x + (block_width - lower_width) // 2
    symbol_y = baseline_y - int(symbol_height * 0.56)
    upper_y = symbol_y - upper_height - 8
    lower_y = symbol_y + symbol_height - 18
    draw.text((upper_x, upper_y), upper, font=bound_font, fill=fill)
    draw.text((symbol_x, symbol_y), symbol, font=symbol_font, fill=fill)
    draw.text((lower_x, lower_y), lower, font=bound_font, fill=fill)
    return x + block_width


def draw_fraction(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    baseline_y: int,
    numerator: str,
    denominator: str,
    fraction_font: ImageFont.ImageFont,
    fill: str = PRIMARY_TEXT,
) -> int:
    num_width, num_height = measure(draw, numerator, fraction_font)
    den_width, den_height = measure(draw, denominator, fraction_font)
    content_width = max(num_width, den_width) + 28
    num_x = x + (content_width - num_width) // 2
    den_x = x + (content_width - den_width) // 2
    line_y = baseline_y - 4
    numerator_y = line_y - num_height - 12
    denominator_y = line_y + 10
    draw.text((num_x, numerator_y), numerator, font=fraction_font, fill=fill)
    draw.line((x + 4, line_y, x + content_width - 4, line_y), fill=fill, width=3)
    draw.text((den_x, denominator_y), denominator, font=fraction_font, fill=fill)
    return x + content_width


def draw_full_equation(spec: EquationSpec, output_path: Path) -> None:
    image = Image.new("RGB", (3200, 760), CANVAS_BACKGROUND)
    draw = ImageDraw.Draw(image)

    title_font = font(72, bold=True)
    main_font = font(74, italic=True)
    operator_font = font(126)
    bound_font = font(42, italic=True)
    fraction_font = font(48, italic=True)
    note_font = font(54, italic=True)

    draw_rounded_card(draw, (66, 136, 3136, 626))
    draw.text((190, 182), "Descriptive unified equation", font=title_font, fill=PRIMARY_TEXT)

    baseline_y = 402
    x = 190
    x = draw_text(
        draw,
        x=x,
        y=baseline_y - 58,
        text=f"y(t) = ({compact_rhs(spec)}) + ",
        text_font=main_font,
    )
    x = draw_bound_operator(
        draw,
        x=x + 18,
        baseline_y=baseline_y,
        upper=str(spec.upper_index),
        symbol="Σ",
        lower="i = 0",
        symbol_font=operator_font,
        bound_font=bound_font,
    )
    x = draw_text(
        draw,
        x=x + 18,
        y=baseline_y - 52,
        text="cᵢ",
        text_font=main_font,
    )
    x = draw_bound_operator(
        draw,
        x=x + 18,
        baseline_y=baseline_y,
        upper="",
        symbol="∏",
        lower="j ≠ i",
        symbol_font=operator_font,
        bound_font=bound_font,
    )
    x = draw_fraction(
        draw,
        x=x + 16,
        baseline_y=baseline_y + 18,
        numerator="τ(t) − j",
        denominator="i − j",
        fraction_font=fraction_font,
    )
    draw_text(
        draw,
        x=x + 26,
        y=baseline_y - 58,
        text=",  τ(tᵢ) = i",
        text_font=main_font,
    )

    draw.text((190, 536), f"cᵢ = yᵢ − {COMPACT_MARK}ᵢ", font=note_font, fill=SECONDARY_TEXT)
    image.save(output_path)


def draw_breakdown(spec: EquationSpec, output_path: Path) -> None:
    image = Image.new("RGB", (2200, 1280), CANVAS_BACKGROUND)
    draw = ImageDraw.Draw(image)

    title_font = font(76, bold=True)
    section_font = font(44)
    main_font = font(78, italic=True)
    operator_font = font(128)
    bound_font = font(44, italic=True)
    fraction_font = font(50, italic=True)
    note_font = font(60, italic=True)

    draw_rounded_card(draw, (96, 100, 2104, 1188))
    draw.text((200, 168), "Equation-only breakdown", font=title_font, fill=PRIMARY_TEXT)

    draw.text((200, 346), "Compact deterministic law", font=section_font, fill=SECONDARY_TEXT)
    draw.text(
        (200, 422),
        f"{COMPACT_MARK}(t) = {compact_rhs(spec)}",
        font=main_font,
        fill=PRIMARY_TEXT,
    )

    draw.text((200, 668), "Exact residual closure", font=section_font, fill=SECONDARY_TEXT)
    baseline_y = 866
    x = 200
    x = draw_bound_operator(
        draw,
        x=x,
        baseline_y=baseline_y,
        upper=str(spec.upper_index),
        symbol="Σ",
        lower="i = 0",
        symbol_font=operator_font,
        bound_font=bound_font,
    )
    x = draw_text(
        draw,
        x=x + 20,
        y=baseline_y - 56,
        text="cᵢ",
        text_font=main_font,
    )
    x = draw_bound_operator(
        draw,
        x=x + 22,
        baseline_y=baseline_y,
        upper="",
        symbol="∏",
        lower="j ≠ i",
        symbol_font=operator_font,
        bound_font=bound_font,
    )
    draw_fraction(
        draw,
        x=x + 16,
        baseline_y=baseline_y + 18,
        numerator="τ(t) − j",
        denominator="i − j",
        fraction_font=fraction_font,
    )

    draw.text((200, 1008), "τ(tᵢ) = i", font=note_font, fill=SECONDARY_TEXT)
    draw.text((200, 1100), f"cᵢ = yᵢ − {COMPACT_MARK}ᵢ", font=note_font, fill=SECONDARY_TEXT)
    image.save(output_path)


def draw_compact_graph(spec: EquationSpec, output_path: Path) -> None:
    image = Image.new("RGB", (2200, 1400), CANVAS_BACKGROUND)
    draw = ImageDraw.Draw(image)

    title_font = font(76, bold=True)
    body_font = font(38)
    body_italic = font(44, italic=True)
    tick_font = font(30)
    label_font = font(36, bold=True)
    note_font = font(32)

    draw_rounded_card(draw, (90, 90, 2110, 1310))
    draw.text((190, 160), "Graph of the displayed equation", font=title_font, fill=PRIMARY_TEXT)
    draw.text(
        (190, 270),
        "The exact sample-closure term needs the residual coefficients cᵢ.",
        font=body_font,
        fill=SECONDARY_TEXT,
    )
    draw.text(
        (190, 320),
        f"This graph shows the visible compact law: {COMPACT_MARK}(t) = {compact_rhs(spec)}",
        font=body_italic,
        fill=PRIMARY_TEXT,
    )

    chart_left = 230
    chart_top = 430
    chart_right = 1980
    chart_bottom = 1100

    draw.rounded_rectangle(
        (chart_left, chart_top, chart_right, chart_bottom),
        radius=18,
        fill="#fffefe",
        outline=CARD_BORDER,
        width=2,
    )

    start_x = 0
    end_x = spec.upper_index
    start_y = spec.intercept
    end_y = spec.intercept + spec.slope * spec.upper_index
    min_y = min(start_y, end_y)
    max_y = max(start_y, end_y)
    y_pad = max((max_y - min_y) * 0.12, 0.00008)
    plot_min_y = min_y - y_pad
    plot_max_y = max_y + y_pad

    def px_x(value: float) -> int:
        return int(chart_left + 80 + (value - start_x) * (chart_right - chart_left - 150) / max(1, end_x - start_x))

    def px_y(value: float) -> int:
        return int(chart_bottom - 70 - (value - plot_min_y) * (chart_bottom - chart_top - 140) / (plot_max_y - plot_min_y))

    # Grid and ticks
    x_ticks = [0, 250, 500, 750, 1000, spec.upper_index]
    y_ticks = [plot_min_y + (plot_max_y - plot_min_y) * step / 5 for step in range(6)]
    for y_tick in y_ticks:
        y = px_y(y_tick)
        draw.line((chart_left + 80, y, chart_right - 40, y), fill=GRID, width=2)
        label = f"{y_tick:.6f}"
        w, h = measure(draw, label, tick_font)
        draw.text((chart_left + 20 - w, y - h // 2), label, font=tick_font, fill=SECONDARY_TEXT)

    for x_tick in x_ticks:
        x = px_x(x_tick)
        draw.line((x, chart_top + 30, x, chart_bottom - 70), fill=GRID, width=2)
        label = str(x_tick)
        w, _h = measure(draw, label, tick_font)
        draw.text((x - w // 2, chart_bottom - 50), label, font=tick_font, fill=SECONDARY_TEXT)

    # Axes
    draw.line((chart_left + 80, chart_top + 30, chart_left + 80, chart_bottom - 70), fill=PRIMARY_TEXT, width=4)
    draw.line((chart_left + 80, chart_bottom - 70, chart_right - 40, chart_bottom - 70), fill=PRIMARY_TEXT, width=4)

    # Line
    points: list[tuple[int, int]] = []
    samples = max(64, spec.upper_index)
    for step in range(samples + 1):
        t = spec.upper_index * step / samples
        y = spec.intercept + spec.slope * t
        points.append((px_x(t), px_y(y)))
    draw.line(points, fill=ACCENT, width=7)

    # End markers
    start_pt = (px_x(start_x), px_y(start_y))
    end_pt = (px_x(end_x), px_y(end_y))
    for px, py in (start_pt, end_pt):
        draw.ellipse((px - 9, py - 9, px + 9, py + 9), fill=ACCENT, outline="#ffffff", width=2)

    start_label = f"t=0, y={start_y:.6f}"
    end_label = f"t={spec.upper_index}, y={end_y:.6f}"
    draw.text((start_pt[0] + 18, start_pt[1] + 10), start_label, font=note_font, fill=PRIMARY_TEXT)
    end_w, end_h = measure(draw, end_label, note_font)
    draw.text((end_pt[0] - end_w - 20, end_pt[1] - end_h - 14), end_label, font=note_font, fill=PRIMARY_TEXT)

    draw.text((1040, 1170), "t (sample index)", font=label_font, fill=PRIMARY_TEXT)
    draw.text((190, 390), "y(t)", font=label_font, fill=PRIMARY_TEXT)
    draw.text(
        (190, 1238),
        "If you want the exact full curve, give me the matching analysis.json so I can recover the residual coefficients.",
        font=note_font,
        fill=SECONDARY_TEXT,
    )

    image.save(output_path)


def load_holistic_curve(analysis_path: Path) -> tuple[list[dict[str, float | str]], str]:
    data = json.loads(analysis_path.read_text())
    holistic = data.get("holistic_equation") or {}
    equation = holistic.get("equation") or {}
    chart = holistic.get("chart") or {}
    curve = chart.get("equation_curve") or equation.get("curve") or []
    if not isinstance(curve, list) or not curve:
        raise SystemExit("analysis does not include holistic_equation.chart.equation_curve")
    label = equation.get("label") or "Holistic equation"
    return curve, label


def draw_holistic_graph(
    *,
    curve: list[dict[str, float | str]],
    equation_label: str,
    output_path: Path,
) -> None:
    image = Image.new("RGB", (2200, 1400), CANVAS_BACKGROUND)
    draw = ImageDraw.Draw(image)

    title_font = font(76, bold=True)
    body_font = font(38)
    body_italic = font(34, italic=True)
    tick_font = font(30)
    label_font = font(36, bold=True)
    note_font = font(32)

    draw_rounded_card(draw, (90, 90, 2110, 1310))
    draw.text((190, 160), "Graph of the full holistic equation", font=title_font, fill=PRIMARY_TEXT)
    draw.text(
        (190, 270),
        "This is the actual holistic equation curve stored in the analysis, not the compact term alone.",
        font=body_font,
        fill=SECONDARY_TEXT,
    )
    draw.text((190, 320), "For sample exact closure, this equals the observed path on-sample.", font=body_font, fill=SECONDARY_TEXT)
    preview = "y(t) = (compact law) + exact residual-closure term, with τ(tᵢ) = i"
    draw.text((190, 378), preview, font=body_italic, fill=PRIMARY_TEXT)

    chart_left = 230
    chart_top = 470
    chart_right = 1980
    chart_bottom = 1100

    draw.rounded_rectangle(
        (chart_left, chart_top, chart_right, chart_bottom),
        radius=18,
        fill="#fffefe",
        outline=CARD_BORDER,
        width=2,
    )

    values = [float(point["fitted_value"]) for point in curve]
    x_start = 0
    x_end = len(curve) - 1
    min_y = min(values)
    max_y = max(values)
    y_pad = max((max_y - min_y) * 0.12, 0.001)
    plot_min_y = min_y - y_pad
    plot_max_y = max_y + y_pad

    def px_x(value: float) -> int:
        return int(chart_left + 80 + (value - x_start) * (chart_right - chart_left - 150) / max(1, x_end - x_start))

    def px_y(value: float) -> int:
        return int(chart_bottom - 70 - (value - plot_min_y) * (chart_bottom - chart_top - 140) / (plot_max_y - plot_min_y))

    x_ticks = sorted(set([0, len(curve) // 4, len(curve) // 2, (3 * len(curve)) // 4, x_end]))
    y_ticks = [plot_min_y + (plot_max_y - plot_min_y) * step / 5 for step in range(6)]
    for y_tick in y_ticks:
        y = px_y(y_tick)
        draw.line((chart_left + 80, y, chart_right - 40, y), fill=GRID, width=2)
        label = f"{y_tick:.4f}"
        w, h = measure(draw, label, tick_font)
        draw.text((chart_left + 20 - w, y - h // 2), label, font=tick_font, fill=SECONDARY_TEXT)

    for x_tick in x_ticks:
        x = px_x(x_tick)
        draw.line((x, chart_top + 30, x, chart_bottom - 70), fill=GRID, width=2)
        label = str(x_tick)
        w, _h = measure(draw, label, tick_font)
        draw.text((x - w // 2, chart_bottom - 50), label, font=tick_font, fill=SECONDARY_TEXT)

    draw.line((chart_left + 80, chart_top + 30, chart_left + 80, chart_bottom - 70), fill=PRIMARY_TEXT, width=4)
    draw.line((chart_left + 80, chart_bottom - 70, chart_right - 40, chart_bottom - 70), fill=PRIMARY_TEXT, width=4)

    points: list[tuple[int, int]] = []
    for idx, value in enumerate(values):
        points.append((px_x(idx), px_y(value)))
    draw.line(points, fill=ACCENT, width=4)

    first = curve[0]
    last = curve[-1]
    first_pt = points[0]
    last_pt = points[-1]
    for px, py in (first_pt, last_pt):
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=ACCENT, outline="#ffffff", width=2)

    first_label = f"{first.get('event_time', '')[:10]}  y={values[0]:.4f}"
    last_label = f"{last.get('event_time', '')[:10]}  y={values[-1]:.4f}"
    draw.text((first_pt[0] + 18, first_pt[1] + 10), first_label, font=note_font, fill=PRIMARY_TEXT)
    last_w, last_h = measure(draw, last_label, note_font)
    draw.text((last_pt[0] - last_w - 18, last_pt[1] - last_h - 12), last_label, font=note_font, fill=PRIMARY_TEXT)

    draw.text((960, 1170), "τ(t) sample index", font=label_font, fill=PRIMARY_TEXT)
    draw.text((190, 430), "y(t)", font=label_font, fill=PRIMARY_TEXT)
    draw.text(
        (190, 1238),
        "Because this is an exact closure over the observed sample, the holistic curve interpolates the sample exactly.",
        font=note_font,
        fill=SECONDARY_TEXT,
    )

    image.save(output_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rows < 1:
        raise SystemExit("--rows must be at least 1")

    spec = EquationSpec(
        intercept=args.intercept,
        slope=args.slope,
        row_count=args.rows,
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    full_path = output_root / "holistic_equation_full.png"
    breakdown_path = output_root / "holistic_equation_breakdown.png"
    graph_path = output_root / "holistic_equation_compact_graph.png"
    draw_full_equation(spec, full_path)
    draw_breakdown(spec, breakdown_path)
    draw_compact_graph(spec, graph_path)

    holistic_graph_path = None
    if args.analysis_path is not None:
        curve, equation_label = load_holistic_curve(args.analysis_path.expanduser().resolve())
        holistic_graph_path = output_root / "holistic_equation_full_graph.png"
        draw_holistic_graph(
            curve=curve,
            equation_label=equation_label,
            output_path=holistic_graph_path,
        )

    print(full_path)
    print(breakdown_path)
    print(graph_path)
    if holistic_graph_path is not None:
        print(holistic_graph_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
