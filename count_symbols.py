"""
Einfaches Skript zum Zählen von Symbolen in MusicXML-Dateien.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def load_xml(file_path: Path) -> ET.Element:
    """Lädt eine .xml Datei und gibt das Root-Element zurück."""
    if file_path.suffix.lower() != ".xml":
        raise ValueError(f"Nur .xml Dateien werden unterstützt: {file_path}")

    tree = ET.parse(file_path)
    return tree.getroot()


def count_symbols(root: ET.Element) -> dict:
    """Zählt verschiedene musikalische Symbole in einem MusicXML-Dokument."""
    counts = {
        "notes": 0,
        "rests": 0,
        "clefs": 0,
        "key_signatures": 0,
        "time_signatures": 0,
        "barlines": 0,
        "directions": 0,
        "dynamics": 0,
        "slurs": 0,
        "ties": 0,
        "articulations": 0,
        "ornaments": 0,
        "words": 0,
    }

    # Zähle Noten und Pausen
    for note in root.findall(".//{*}note"):
        if note.find("{*}rest") is not None:
            counts["rests"] += 1
        else:
            counts["notes"] += 1

        # Zähle Bindebögen (ties)
        if note.find("{*}tie") is not None:
            counts["ties"] += 1

        # Zähle Artikulationen
        if note.find("{*}notations/{*}articulations") is not None:
            counts["articulations"] += 1

        # Zähle Ornamente
        if note.find("{*}notations/{*}ornaments") is not None:
            counts["ornaments"] += 1

    # Zähle weitere Elemente
    counts["clefs"] = len(root.findall(".//{*}clef"))
    counts["key_signatures"] = len(root.findall(".//{*}key"))
    counts["time_signatures"] = len(root.findall(".//{*}time"))
    counts["barlines"] = len(root.findall(".//{*}barline"))
    counts["directions"] = len(root.findall(".//{*}direction"))
    counts["dynamics"] = len(root.findall(".//{*}dynamics"))
    counts["slurs"] = len(root.findall(".//{*}slur"))
    counts["words"] = len(root.findall(".//{*}words"))

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Zählt Symbole in einer MusicXML-Datei"
    )
    parser.add_argument("file", type=Path, help="Pfad zur .xml Datei")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Zeige detaillierte Informationen"
    )

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Fehler: Datei nicht gefunden: {args.file}")
        return

    # Lade und zähle
    root = load_xml(args.file)
    counts = count_symbols(root)

    # Gesamtsumme
    total = sum(counts.values())

    # Ausgabe
    print(f"\nDatei: {args.file.name}")
    print(f"{'='*60}")

    if args.verbose:
        for key, value in counts.items():
            if value > 0:
                print(f"{key:20s}: {value:6d}")
        print(f"{'='*60}")

    print(f"{'Gesamt':20s}: {total:6d}")


if __name__ == "__main__":
    main()
