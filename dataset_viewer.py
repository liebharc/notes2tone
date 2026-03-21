"""Dataset viewer for the PRAIG/SMB dataset.

Allows browsing and inspecting images with their **kern ground truth annotations.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.datasets.smb_loader import SMBDataset

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')  # Use Tk backend for Windows
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️  matplotlib not installed - using text-only mode")
    print("   Install with: uv pip install matplotlib")


def display_sample_gui(sample, index, total):
    """Display sample with matplotlib GUI."""
    if not HAS_PLOTTING:
        print("❌ Plotting not available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    filename = sample.get('metadata', {}).get('filename', f'sample_{index}')
    fig.suptitle(f"Sample {index}/{total} - {filename}", fontsize=12)
    
    # Display image
    image = sample['image']
    ax1.imshow(image, cmap='gray')
    ax1.axis('off')
    ax1.set_title(f"Image ({image.width}x{image.height})")
    
    # Display ground truth as text
    gt = sample['ground_truth']
    ax2.axis('off')
    ax2.set_title(f"Ground Truth ({len(gt)} chars)")
    
    # Wrap text for readability and remove problematic characters
    # Replace tabs and special unicode with readable equivalents
    cleaned_gt = gt.replace('\t', '  ').replace('\r', '')
    wrapped_text = '\n'.join(
        cleaned_gt[i:i+60] for i in range(0, len(cleaned_gt), 60)
    )
    
    # Show first 2000 chars to avoid clutter
    if len(wrapped_text) > 2000:
        wrapped_text = wrapped_text[:2000] + "\n\n[... truncated]"
    
    ax2.text(0.05, 0.95, wrapped_text, 
             verticalalignment='top',
             fontfamily='monospace',
             fontsize=8,
             wrap=True)
    
    plt.tight_layout(pad=2.0)
    plt.show()


def display_sample_text(sample, index, total):
    """Display sample in text-only mode."""
    print("\n" + "=" * 80)
    print(f"Sample {index}/{total}")
    print("=" * 80)
    filename = sample.get('metadata', {}).get('filename', f'sample_{index}')
    print(f"Filename: {filename}")
    print(f"Image size: {sample['image'].width}x{sample['image'].height}")
    print(f"Ground truth length: {len(sample['ground_truth'])} characters")
    print("\nGround truth (first 500 chars):")
    print("-" * 80)
    print(sample['ground_truth'][:500])
    if len(sample['ground_truth']) > 500:
        print("\n[... truncated]")
    print("-" * 80)


def save_sample(sample, output_dir):
    """Save sample to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = sample.get('metadata', {}).get('filename', 'sample')
    base_name = Path(filename).stem if filename else 'sample'
    
    # Save image
    image_path = output_dir / f"{base_name}_image.png"
    sample['image'].save(image_path)
    print(f"✅ Saved image to {image_path}")
    
    # Save ground truth
    gt_path = output_dir / f"{base_name}_groundtruth.txt"
    with open(gt_path, 'w', encoding='utf-8') as f:
        f.write(sample['ground_truth'])
    print(f"✅ Saved ground truth to {gt_path}")


def browse_dataset(dataset, start_index=0, gui_mode=True):
    """Interactive dataset browser."""
    current_idx = start_index
    total = len(dataset)
    
    print(f"\nDataset loaded: {total} samples")
    print("\nControls:")
    print("  n - next sample")
    print("  p - previous sample")
    print("  j <num> - jump to sample number")
    print("  s - save current sample to disk")
    print("  q - quit")
    print()
    
    while True:
        # Load and display current sample
        print(f"\nLoading sample {current_idx}/{total}...")
        sample = dataset[current_idx]
        
        if gui_mode and HAS_PLOTTING:
            display_sample_gui(sample, current_idx, total)
        else:
            display_sample_text(sample, current_idx, total)
        
        # Get user input
        try:
            command = input(f"\n[{current_idx}/{total}] Command (n/p/j/s/q): ").strip().lower()
            
            if command == 'q':
                print("Exiting...")
                break
            elif command == 'n':
                current_idx = min(current_idx + 1, total - 1)
            elif command == 'p':
                current_idx = max(current_idx - 1, 0)
            elif command.startswith('j'):
                try:
                    parts = command.split()
                    if len(parts) == 2:
                        jump_idx = int(parts[1])
                        if 0 <= jump_idx < total:
                            current_idx = jump_idx
                        else:
                            print(f"❌ Invalid index: must be 0-{total-1}")
                    else:
                        print("❌ Usage: j <number>")
                except ValueError:
                    print("❌ Invalid number")
            elif command == 's':
                output_dir = Path("data/samples")
                save_sample(sample, output_dir)
            else:
                print("❌ Unknown command")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Browse the PRAIG/SMB dataset interactively"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to browse (default: test)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting sample index (default: 0)"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Use text-only mode (no GUI)"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Show random samples"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    print("=" * 80)
    print("Notes2Tone Dataset Viewer")
    print("=" * 80)
    print(f"\nLoading {args.split} split...")
    
    try:
        # Load dataset
        dataset = SMBDataset(split=args.split)
        print(f"✅ Loaded {len(dataset)} samples")
        
        # Start browsing
        if args.random:
            import random
            start_idx = random.randint(0, len(dataset) - 1)
            print(f"Starting at random sample: {start_idx}")
        else:
            start_idx = args.start
        
        gui_mode = HAS_PLOTTING and not args.text_only
        browse_dataset(dataset, start_index=start_idx, gui_mode=gui_mode)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
