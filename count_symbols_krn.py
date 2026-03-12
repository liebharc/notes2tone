"""
Einfaches Skript zum Zählen von Symbolen in Humdrum Kern (.krn) Dateien.
"""

import argparse
import re
from pathlib import Path


def load_krn(file_path: Path) -> list[str]:
    """Lädt eine .krn Datei und gibt die Zeilen zurück."""
    if file_path.suffix.lower() != ".krn":
        raise ValueError(f"Nur .krn Dateien werden unterstützt: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()


def count_symbols(lines: list[str]) -> dict:
    """Zählt verschiedene musikalische Symbole in einem Humdrum Kern-Dokument."""
    counts = {
        "notes": 0,
        "rests": 0,
        "clefs": 0,
        "key_signatures": 0,
        "time_signatures": 0,
        "barlines": 0,
        "ties": 0,
        "slurs": 0,
        "articulations": 0,
        "fermatas": 0,
        "dynamics": 0,
        "ornaments": 0,
    }

    # Pattern für Noten (Buchstaben für Tonhöhen)
    note_pattern = re.compile(r"[A-Ga-g]+")

    for line in lines:
        line = line.strip()

        # Überspringe Kommentare und leere Zeilen
        if not line or line.startswith("!"):
            continue

        # Taktstriche
        if line.startswith("="):
            counts["barlines"] += 1
            continue

        # Interpretationen (beginnen mit *)
        if line.startswith("*"):
            if "clef" in line.lower():
                counts["clefs"] += 1
            elif line.startswith("*k["):
                counts["key_signatures"] += 1
            elif line.startswith("*M"):
                counts["time_signatures"] += 1
            continue

        # Daten-Tokens (durch Tabs getrennt)
        tokens = line.split("\t")

        for token in tokens:
            token = token.strip()
            if not token or token == ".":
                continue

            # Pausen
            if "r" in token:
                counts["rests"] += 1
            # Noten
            elif note_pattern.search(token):
                counts["notes"] += 1

                # Bindebögen (ties): [ = Start, ] = Ende, _ = Fortsetzung
                if "[" in token or "]" in token or "_" in token:
                    counts["ties"] += 1

                # Legatobögen (slurs): ( = Start, ) = Ende
                if "(" in token or ")" in token:
                    counts["slurs"] += 1

                # Artikulationen
                if "'" in token or '"' in token or "`" in token or "~" in token:
                    counts["articulations"] += 1

                # Fermaten
                if ";" in token:
                    counts["fermatas"] += 1

                # Ornamente (Triller, Mordent, etc.)
                if (
                    "t" in token
                    or "T" in token
                    or "M" in token
                    or "W" in token
                    or "S" in token
                ):
                    # Vorsichtig: könnte auch Teil einer Note sein
                    # Prüfe auf spezielle Ornament-Marker
                    if re.search(r"[tTMWS](?![a-gA-G])", token):
                        counts["ornaments"] += 1

            # Dynamik (piano, forte, etc.)
            if any(
                dyn in token for dyn in ["pp", "p", "mp", "mf", "f", "ff", "sfz", "fp"]
            ):
                counts["dynamics"] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Zählt Symbole in einer Humdrum Kern (.krn) Datei"
    )
    parser.add_argument("file", type=Path, help="Pfad zur .krn Datei")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Zeige detaillierte Informationen"
    )

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Fehler: Datei nicht gefunden: {args.file}")
        return

    # Lade und zähle
    lines = load_krn(args.file)
    counts = count_symbols(lines)

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
