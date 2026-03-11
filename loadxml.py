from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET
import zipfile

# Treat .mxl as compressed MusicXML input next to plain .xml files.
XML_EXTENSIONS = {".xml", ".mxl"}


def collect_xml_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []

    files = [
        item
        for item in source_dir.iterdir()
        if item.is_file() and item.suffix.lower() in XML_EXTENSIONS
    ]
    return sorted(files, key=lambda path: path.name.lower())


def _target_filename(src: Path) -> str:
    if src.suffix.lower() == ".mxl":
        return f"{src.stem}.xml"
    return src.name


def _load_musicxml_bytes(src: Path) -> bytes:
    if src.suffix.lower() == ".xml":
        return src.read_bytes()

    if src.suffix.lower() != ".mxl":
        raise ValueError(f"Unsupported extension: {src.suffix}")

    with zipfile.ZipFile(src, "r") as archive:
        namelist = archive.namelist()

        # MusicXML .mxl usually points to the score via META-INF/container.xml.
        if "META-INF/container.xml" in namelist:
            container_xml = archive.read("META-INF/container.xml")
            root = ET.fromstring(container_xml)
            rootfile = root.find(".//{*}rootfile")
            full_path = rootfile.attrib.get("full-path") if rootfile is not None else None
            if full_path and full_path in namelist:
                return archive.read(full_path)

        xml_entries = [name for name in namelist if name.lower().endswith(".xml") and not name.startswith("META-INF/")]
        if not xml_entries:
            raise ValueError(f"No score XML found inside archive: {src}")

        return archive.read(sorted(xml_entries, key=lambda name: name.lower())[0])


def copy_xml_batch(files: Iterable[Path], target_dir: Path, dry_run: bool = False) -> tuple[int, int, int]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    failed = 0

    for src in files:
        try:
            xml_bytes = _load_musicxml_bytes(src)
            dst = target_dir / _target_filename(src)

            if dst.exists() and dst.read_bytes() == xml_bytes:
                skipped += 1
                continue

            if not dry_run:
                dst.write_bytes(xml_bytes)
            copied += 1
        except (OSError, ValueError, zipfile.BadZipFile, ET.ParseError) as exc:
            failed += 1
            print(f"[WARN] Failed to process {src}: {exc}")

    return copied, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy Audiveris XML/MXL output into generated/xml and generated/xml_scaled."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root path (defaults to this file's folder).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing files.",
    )
    return parser.parse_args()


def run(root: Path, dry_run: bool = False) -> int:
    source_regular = root / "audiveris_output"
    source_scaled = root / "audiveris_output_scaled"

    target_regular = root / "generated" / "xml"
    target_scaled = root / "generated" / "xml_scaled"

    regular_files = collect_xml_files(source_regular)
    scaled_files = collect_xml_files(source_scaled)

    copied_regular, skipped_regular, failed_regular = copy_xml_batch(regular_files, target_regular, dry_run=dry_run)
    copied_scaled, skipped_scaled, failed_scaled = copy_xml_batch(scaled_files, target_scaled, dry_run=dry_run)

    mode = "DRY-RUN" if dry_run else "DONE"
    print(
        f"[{mode}] {source_regular} -> {target_regular}: "
        f"copied={copied_regular}, skipped={skipped_regular}, failed={failed_regular}"
    )
    print(
        f"[{mode}] {source_scaled} -> {target_scaled}: "
        f"copied={copied_scaled}, skipped={skipped_scaled}, failed={failed_scaled}"
    )

    return 0 if (failed_regular + failed_scaled) == 0 else 1


def main() -> int:
    args = parse_args()
    return run(root=args.root.resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

