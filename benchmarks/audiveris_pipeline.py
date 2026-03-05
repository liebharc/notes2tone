import argparse
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from dotenv import load_dotenv

from audiveris.AudiverisProcessor import AudiverisProcessor
from benchmarks.converters.musicxml_to_kern import convert_musicxml_to_kern
from benchmarks.datasets import SMBDataset

logger = logging.getLogger(__name__)


def _resolve_musicxml_member(mxl_path: Path) -> str:
    with zipfile.ZipFile(mxl_path, "r") as zf:
        members = [name for name in zf.namelist() if not name.endswith("/")]
        for name in members:
            if name.lower().endswith(".xml") and not name.lower().endswith("container.xml"):
                return name

        container_name = "META-INF/container.xml"
        if container_name not in zf.namelist():
            raise ValueError(f"No score XML found in {mxl_path}")

        root = ET.fromstring(zf.read(container_name))
        full_path = root.find(".//{*}rootfile")
        if full_path is None:
            raise ValueError(f"container.xml missing rootfile entry in {mxl_path}")
        media_path = full_path.attrib.get("full-path")
        if not media_path:
            raise ValueError(f"container.xml has empty full-path in {mxl_path}")
        return media_path


def _read_musicxml_from_mxl(mxl_path: Path) -> str:
    member_name = _resolve_musicxml_member(mxl_path)
    with zipfile.ZipFile(mxl_path, "r") as zf:
        raw_xml = zf.read(member_name)
    return raw_xml.decode("utf-8", errors="replace")


def _copy_ground_truth(samples: list[dict], gt_dir: Path) -> None:
    gt_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        sample_id = sample["sample_id"]
        gt_text = sample.get("ground_truth", "")
        (gt_dir / f"{sample_id}.krn").write_text(gt_text, encoding="utf-8")


def _convert_audiveris_output_to_pred(output_dir: Path, pred_dir: Path) -> int:
    pred_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for mxl_path in sorted(output_dir.glob("*.mxl")):
        try:
            xml_content = _read_musicxml_from_mxl(mxl_path)
            kern_content = convert_musicxml_to_kern(xml_content)
            (pred_dir / f"{mxl_path.stem}.krn").write_text(kern_content, encoding="utf-8")
            converted += 1
        except Exception as exc:
            logger.warning(f"Skipping {mxl_path.name}: {exc}")
    return converted


def run_pipeline(
    audiveris_path: str,
    audiveris_output: Path,
    benchmark_output: Path,
    split: str = "test",
    limit: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> None:
    dataset = SMBDataset(split=split, limit=limit, token=hf_token)
    processor = AudiverisProcessor(audiveris_path=audiveris_path, output_dir=str(audiveris_output))

    samples = processor.process_dataset(dataset, limit=limit)

    pred_dir = benchmark_output / "pred"
    gt_dir = benchmark_output / "gt"

    converted = _convert_audiveris_output_to_pred(audiveris_output, pred_dir)
    _copy_ground_truth(samples, gt_dir)

    logger.info(f"Converted {converted} .mxl files into {pred_dir}")
    logger.info(f"Copied {len(samples)} ground-truth files into {gt_dir}")


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Run Audiveris on SMB and build pred/gt **kern folders")
    parser.add_argument("--audiveris-path", required=True, help="Path to Audiveris executable")
    parser.add_argument("--audiveris-output", type=Path, default=Path("audiveris_output"))
    parser.add_argument("--benchmark-output", type=Path, default=Path("benchmarks/generated"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--clean", action="store_true", help="Delete old output folders before running")

    args = parser.parse_args()

    if args.clean:
        if args.audiveris_output.exists():
            shutil.rmtree(args.audiveris_output)
        if args.benchmark_output.exists():
            shutil.rmtree(args.benchmark_output)

    run_pipeline(
        audiveris_path=args.audiveris_path,
        audiveris_output=args.audiveris_output,
        benchmark_output=args.benchmark_output,
        split=args.split,
        limit=args.limit,
        hf_token=args.hf_token,
    )

