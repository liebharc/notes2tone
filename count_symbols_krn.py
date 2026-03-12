"""
Skript zum Zählen von Symbolen in Humdrum Kern (.krn) Dateien.
Verwendet dieselbe Methode wie musicdiff (AnnScore.notation_size()),
sodass die Ausgabe mit den gt numsyms in den CSV-Dateien übereinstimmt.
"""

import argparse
from pathlib import Path

import converter21
import music21 as m21

from musicdiff.annotation import AnnScore

converter21.register()


def count_symbols(file_path: Path) -> int:
    """
    Zählt die Symbole in einer .krn Datei auf dieselbe Art wie musicdiff.

    Args:
        file_path: Pfad zur .krn Datei

    Returns:
        int: Anzahl der Symbole (notation_size gemäß musicdiff)
    """
    if file_path.suffix.lower() != ".krn":
        raise ValueError(f"Nur .krn Dateien werden unterstützt: {file_path}")

    score = m21.converter.parse(
        str(file_path), forceSource=True, acceptSyntaxErrors=False
    )
    if isinstance(score, m21.stream.Opus):
        score = score.scores[0]

    ann_score = AnnScore(score)
    return ann_score.notation_size()


def main():
    parser = argparse.ArgumentParser(
        description="Zählt Symbole in einer Humdrum Kern (.krn) Datei (kompatibel mit musicdiff)"
    )
    parser.add_argument("file", type=Path, help="Pfad zur .krn Datei")

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Fehler: Datei nicht gefunden: {args.file}")
        return

    total = count_symbols(args.file)

    print(f"\nDatei: {args.file.name}")
    print(f"{'='*60}")
    print(f"{'Gesamt':20s}: {total:6d}")


if __name__ == "__main__":
    main()
