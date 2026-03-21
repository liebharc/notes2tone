from __future__ import annotations

import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil

import converter21
import music21 as m21


def _find_gt_for_pred(pred_name: str, gt_dir: Path) -> Path | None:
    exact = gt_dir / pred_name
    if exact.is_file():
        return exact

    stem = Path(pred_name).stem
    matches = [p for p in gt_dir.iterdir() if p.is_file() and p.stem == stem]
    if len(matches) == 1:
        return matches[0]
    return None


def _parse_score(path: Path) -> m21.stream.Score | None:
    try:
        score = m21.converter.parse(str(path), forceSource=True, acceptSyntaxErrors=False)
        if isinstance(score, m21.stream.Opus):
            if not score.scores:
                return None
            score = score.scores[0]
        if not isinstance(score, m21.stream.Score):
            return None
        return score
    except Exception:
        return None


def _measure_count(path: Path) -> int | None:
    score = _parse_score(path)
    if score is None or len(score.parts) == 0:
        return None
    return len(score.parts[0].getElementsByClass(m21.stream.Measure))


def _trim_musicxml_to_measure_count(src: Path, dst: Path, max_measures: int) -> tuple[int, int]:
    tree = ET.parse(src)
    root = tree.getroot()
    parts = root.findall('.//part')
    if not parts:
        raise ValueError('No <part> elements found in MusicXML file.')

    original_counts: list[int] = []
    kept_counts: list[int] = []
    for part in parts:
        measures = list(part.findall('measure'))
        original_counts.append(len(measures))
        for measure in measures[max_measures:]:
            part.remove(measure)
        kept_counts.append(min(len(measures), max_measures))

    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding='utf-8', xml_declaration=True)
    return max(original_counts), max(kept_counts)


def prepare_trimmed_predictions(
    gt_dir: Path,
    pred_dir: Path,
    trimmed_pred_dir: Path,
    only_stems: set[str] | None = None,
    keep_untrimmed: bool = False,
) -> tuple[int, int, int, list[str]]:
    trimmed_pred_dir.mkdir(parents=True, exist_ok=True)

    for old_file in trimmed_pred_dir.glob('*.xml'):
        old_file.unlink()

    processed = 0
    trimmed = 0
    skipped = 0
    messages: list[str] = []

    for pred_path in sorted(pred_dir.iterdir(), key=lambda p: p.name.lower()):
        if not pred_path.is_file():
            continue
        if pred_path.suffix.lower() != '.xml':
            continue
        if only_stems is not None and pred_path.stem not in only_stems:
            continue

        gt_path = _find_gt_for_pred(pred_path.name, gt_dir)
        if gt_path is None:
            messages.append(f'[SKIP] {pred_path.name}: no matching GT file found')
            skipped += 1
            continue

        gt_measures = _measure_count(gt_path)
        if gt_measures is None:
            messages.append(f'[SKIP] {pred_path.name}: GT parse failed for {gt_path.name}')
            skipped += 1
            continue

        dst_path = trimmed_pred_dir / pred_path.name
        try:
            pred_measures_before, pred_measures_after = _trim_musicxml_to_measure_count(
                pred_path,
                dst_path,
                gt_measures,
            )
        except Exception as exc:
            messages.append(f'[SKIP] {pred_path.name}: trim failed ({exc})')
            skipped += 1
            continue

        processed += 1
        if pred_measures_before > gt_measures:
            trimmed += 1
            messages.append(
                f'[TRIM] {pred_path.name}: pred {pred_measures_before} -> {pred_measures_after} measures '
                f'(GT={gt_measures}, file={gt_path.name})'
            )
        else:
            if keep_untrimmed:
                shutil.copy2(pred_path, dst_path)
            messages.append(
                f'[KEEP] {pred_path.name}: pred {pred_measures_before} measures <= GT {gt_measures}'
            )

    return processed, trimmed, skipped, messages

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trim predicted MusicXML files to the measure count of the matching GT file, "
            "without running musicdiff evaluation."
        )
    )
    parser.add_argument(
        "--gt",
        default="generated/gt",
        help="Ground-truth folder used to determine the relevant measure count (default: generated/gt).",
    )
    parser.add_argument(
        "--pred",
        default="generated/xml_test",
        help="Predicted MusicXML folder to trim (default: generated/xml_test).",
    )
    parser.add_argument(
        "--out",
        default="generated/pred_trimmed",
        help="Output folder for trimmed MusicXML files (default: generated/pred_trimmed).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of stems to process, e.g. sample_000003.",
    )
    parser.add_argument(
        "--keep-untrimmed",
        action="store_true",
        help=(
            "Copy files unchanged when prediction measure count is already less than or equal "
            "to the GT measure count."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    converter21.register()

    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)
    out_dir = Path(args.out)
    only_stems = set(args.only) if args.only else None

    if not gt_dir.exists():
        print(f"[ERROR] GT folder not found: {gt_dir}", file=sys.stderr)
        return 1
    if not pred_dir.exists():
        print(f"[ERROR] prediction folder not found: {pred_dir}", file=sys.stderr)
        return 1

    processed, trimmed, skipped, messages = prepare_trimmed_predictions(
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        trimmed_pred_dir=out_dir,
        only_stems=only_stems,
        keep_untrimmed=args.keep_untrimmed,
    )

    for line in messages:
        print(line)

    if processed == 0:
        print("[ERROR] No prediction files were prepared.", file=sys.stderr)
        return 2

    print()
    print(f"Prepared files : {processed}")
    print(f"Trimmed files  : {trimmed}")
    print(f"Skipped files  : {skipped}")
    print(f"Output folder  : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

