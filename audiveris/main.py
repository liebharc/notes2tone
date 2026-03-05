from dotenv import load_dotenv
import os

from audiveris.AudiverisProcessor import AudiverisProcessor

load_dotenv()

if __name__ == "__main__":
    from benchmarks.datasets import SMBDataset

    # Initialize dataset
    dataset = SMBDataset()

    audiveris_path = os.getenv("AUDIVERIS_PATH")
    if not audiveris_path:
        raise RuntimeError(
            "AUDIVERIS_PATH environment variable not set. Please set it to the path of the Audiveris executable."
        )

    output_dir = os.getenv("AUDIVERIS_OUTPUT", "./audiveris_output")

    # Process dataset with Audiveris
    processor = AudiverisProcessor(audiveris_path, output_dir)
    processor.process_dataset(dataset, limit=None)  # Set limit for testing, None for all
