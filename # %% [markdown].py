# %% [markdown]
# ## 1. Setup & Dependencies
# 
# **Installation Instructions (Jupyter Lab/Hub):**
# 
# ```bash
# # Navigate to project directory
# cd notes2tone
# 
# # Install dependencies
# pip install -e '.[models,viz]'
# 
# # Install and register Jupyter kernel (IMPORTANT!)
# pip install ipykernel
# python -m ipykernel install --user --name notes2tone --display-name "notes2tone"
# ```
# 
# **After installation:**
# 1. Click on kernel selector (top right corner of notebook)
# 2. Select "notes2tone" from the list
# 3. If not visible, restart Jupyter Lab and try again
# 
# **Verify installation:**
# Run the cells below to check if everything is set up correctly.

# %%
# Verify and install dependencies
import sys
import subprocess

# Check if benchmark framework is available
try:
    from benchmarks.datasets import SMBDataset
    print("Benchmark framework is available")
except ImportError:
    print("Installing benchmark framework...")
    print("-" * 60)
    
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[models,viz]'],
                            capture_output=True, text=True)
    if result.returncode == 0:
        print("Installation completed with pip")
        print("Note: This creates notes2tone.egg-info directory (normal behavior)")
    else:
        print(f"Error during pip installation: {result.stderr}")
        print("\nPlease run manually in terminal:")
        print("   pip install -e '.[models,viz]'")
    
    print("-" * 60)

# %%
# Check GPU availability
import subprocess

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("GPU detected:")
        print(result.stdout.strip())
    else:
        print("No GPU detected - using CPU")
except Exception:
    print("nvidia-smi not available - using CPU")

# %% [markdown]
# ## 2. HuggingFace Authentication

# %%
# Configure HuggingFace authentication (required for PRAIG/SMB dataset)
import os
from getpass import getpass

if 'HF_TOKEN' not in os.environ:
    print("HuggingFace Token Required")
    print("Token URL: https://huggingface.co/settings/tokens")
    print("Dataset access: https://huggingface.co/datasets/PRAIG/SMB")
    hf_token = getpass("Enter HF_TOKEN: ")
    os.environ['HF_TOKEN'] = hf_token
    print("Token configured successfully")
else:
    print("HF_TOKEN already configured")

# %% [markdown]
# ## 3. Import Benchmark Framework

# %%
# Import benchmark framework modules
from benchmarks.datasets import SMBDataset
from benchmarks.models import OemerModel
from benchmarks.benchmark import BenchmarkRunner
from benchmarks.eval import omr_ned

print("Benchmark framework imported successfully")

# %% [markdown]
# ## 4. Load Dataset

# %%
# Load PRAIG/SMB dataset
print("Loading PRAIG/SMB dataset...")

dataset = SMBDataset(
    split="test",
    limit=None,  # Set to None for full dataset, or e.g. 10 for testing
    token=os.environ.get('HF_TOKEN')
)

print(f"Dataset loaded: {len(dataset)} samples")

# %% [markdown]
# ## 6. Initialize Models

# %%
# Initialize OeMeR model
import sys
import os
from pathlib import Path

print("Initializing OeMeR model...")

# Try to find oemer executable
oemer_path = "oemer"  # default
possible_paths = [
    Path(sys.prefix) / "bin" / "oemer",  # venv location
    Path.home() / ".local" / "bin" / "oemer",  # user install
]

for path in possible_paths:
    if path.exists():
        oemer_path = str(path)
        print(f"Found oemer at: {oemer_path}")
        break

oemer = OemerModel(
    oemer_path=oemer_path,
    disable_deskew=True,
    save_cache=False,
    use_tf=False
)

print("Model initialized successfully")
print(f"Model name: {oemer.name}")
print(f"Configuration: {oemer.config}")

# %% [markdown]
# ## 8. Run Full Benchmark

# %%
# Configure benchmark parameters
from pathlib import Path

# Configuration
NUM_SAMPLES = 10  # Change to None for full dataset evaluation
SAVE_PREDICTIONS = True  # Save individual predictions for detailed analysis

# Update dataset with specified limit
if NUM_SAMPLES:
    dataset = SMBDataset(
        split="test",
        limit=NUM_SAMPLES,
        token=os.environ.get('HF_TOKEN')
    )
    print(f"Dataset limited to {NUM_SAMPLES} samples for testing")
else:
    print(f"Using full dataset: {len(dataset)} samples")

# Initialize benchmark runner
output_dir = Path("benchmark_results")
output_dir.mkdir(exist_ok=True)

runner = BenchmarkRunner(
    dataset=dataset,
    output_dir=output_dir,
    save_predictions=SAVE_PREDICTIONS
)

print("Benchmark runner initialized")
print(f"Output directory: {output_dir}")
print(f"Save predictions: {SAVE_PREDICTIONS}")

# %% [markdown]
# ## 9. Run Evaluation

# %%
import time
from datetime import datetime

print("Starting benchmark evaluation...")
print(f"Model: {oemer.name}")
print(f"Sample count: {len(dataset)}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

start_time = time.time()

# Execute evaluation
results = runner.evaluate_model(oemer)

elapsed_time = time.time() - start_time

print("\nEvaluation completed")
print(f"Total duration: {elapsed_time/60:.1f} minutes")
print(f"Average per sample: {elapsed_time/len(dataset):.1f} seconds")

# %% [markdown]
# ## 10. Detailed Metrics Summary

# %%
# Display comprehensive results summary
metrics = results['metrics']

print("=" * 70)
print("BENCHMARK RESULTS")
print("=" * 70)
print(f"Model:               {results['model_name']}")
print(f"Timestamp:           {results['timestamp']}")
print(f"Total Samples:       {metrics['total_samples']}")
print(f"Errors:              {metrics['num_errors']} ({metrics['error_rate']*100:.1f}%)")
print("\n" + "-" * 70)
print("OMR-NED Metrics (lower is better):")
print("-" * 70)
print(f"  Mean NED:          {metrics['mean_ned']:.4f}")
print(f"  Median NED:        {metrics['median_ned']:.4f}")
print(f"  Std Dev:           {metrics['std_ned']:.4f}")
print(f"  Min (Best):        {metrics['min_ned']:.4f}")
print(f"  Max (Worst):       {metrics['max_ned']:.4f}")
# print(f"  25th Percentile:   {metrics['percentile_25']:.4f}")
# print(f"  75th Percentile:   {metrics['percentile_75']:.4f}")
print("\n" + "-" * 70)
print("Perfect Matches:")
print("-" * 70)
print(f"  Count:             {metrics['perfect_matches']}")
print(f"  Percentage:        {metrics['perfect_matches']/metrics['total_samples']*100:.1f}%")
print("=" * 70)

# %% [markdown]
# ## 11. Visualization: NED Distribution

# %%
# Visualization: NED distribution and trends analysis

import numpy as np

# Extract NED scores from results
if SAVE_PREDICTIONS and 'predictions' in results:
    ned_scores = [omr_ned(p['prediction'], p['ground_truth']) 
                  for p in results['predictions']]
else:
    # Recalculate if predictions were not saved
    ned_scores = []
    for item in dataset:
        try:
            pred = oemer.predict(item['image'])
            ned_scores.append(omr_ned(pred, item['ground_truth']))
        except Exception:
            ned_scores.append(None)

# Filter out failed predictions
valid_scores = [s for s in ned_scores if s is not None]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. NED Distribution (Histogram)
axes[0, 0].hist(valid_scores, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(np.mean(valid_scores), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(valid_scores):.4f}')
axes[0, 0].axvline(np.median(valid_scores), color='green', linestyle='--', 
                    linewidth=2, label=f'Median: {np.median(valid_scores):.4f}')
axes[0, 0].set_xlabel('OMR-NED Score', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('NED Score Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. NED over sample index
axes[0, 1].plot(range(len(valid_scores)), valid_scores, marker='o', 
                linestyle='-', alpha=0.6, markersize=4)
axes[0, 1].axhline(np.mean(valid_scores), color='red', linestyle='--', 
                    linewidth=2, label='Mean')
axes[0, 1].set_xlabel('Sample Index', fontsize=12)
axes[0, 1].set_ylabel('OMR-NED Score', fontsize=12)
axes[0, 1].set_title('NED Scores by Sample', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Box plot for quartile visualization
axes[1, 0].boxplot(valid_scores, vert=True)
axes[1, 0].set_ylabel('OMR-NED Score', fontsize=12)
axes[1, 0].set_title('NED Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='y')

# 4. Cumulative distribution function
sorted_scores = np.sort(valid_scores)
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
axes[1, 1].plot(sorted_scores, cumulative, linewidth=2)
axes[1, 1].set_xlabel('OMR-NED Score', fontsize=12)
axes[1, 1].set_ylabel('Cumulative Probability', fontsize=12)
axes[1, 1].set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Length Analysis

# %%
# Analyze prediction vs ground truth length characteristics
if SAVE_PREDICTIONS and 'predictions' in results:
    pred_lengths = [len(p['prediction']) for p in results['predictions']]
    gt_lengths = [len(p['ground_truth']) for p in results['predictions']]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Length comparison scatter plot
    axes[0].scatter(gt_lengths, pred_lengths, alpha=0.6, s=50)
    
    # Add diagonal reference line (perfect prediction)
    max_len = max(max(gt_lengths), max(pred_lengths))
    axes[0].plot([0, max_len], [0, max_len], 'r--', linewidth=2, 
                 label='Perfect Match', alpha=0.7)
    
    axes[0].set_xlabel('Ground Truth Length (chars)', fontsize=12)
    axes[0].set_ylabel('Prediction Length (chars)', fontsize=12)
    axes[0].set_title('Prediction vs Ground Truth Length', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Length difference histogram
    length_diffs = [p - g for p, g in zip(pred_lengths, gt_lengths)]
    axes[1].hist(length_diffs, bins=30, edgecolor='black', alpha=0.7, color='coral')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[1].set_xlabel('Length Difference (pred - gt)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Prediction Length Bias', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical summary
    print("\nLength Analysis:")
    print(f"  Mean GT length:    {np.mean(gt_lengths):.1f} chars")
    print(f"  Mean Pred length:  {np.mean(pred_lengths):.1f} chars")
    print(f"  Mean difference:   {np.mean(length_diffs):+.1f} chars")
    print(f"  Std difference:    {np.std(length_diffs):.1f} chars")
else:
    print("Predictions not saved - enable SAVE_PREDICTIONS for detailed analysis")

# %% [markdown]
# ## 13. Error Patterns by Complexity

# %%
# Analyze error patterns by score complexity
if SAVE_PREDICTIONS and 'predictions' in results:
    print("Analyzing error patterns by ground truth complexity...\n")
    
    # Define complexity bins based on ground truth length
    complexity_bins = [0, 100, 500, 1000, 5000, float('inf')]
    complexity_labels = ['Tiny\n(0-100)', 'Small\n(100-500)', 'Medium\n(500-1k)', 
                         'Large\n(1k-5k)', 'Huge\n(5k+)']
    
    binned_neds = {label: [] for label in complexity_labels}
    
    for pred_item in results['predictions']:
        gt_len = len(pred_item['ground_truth'])
        ned = omr_ned(pred_item['prediction'], pred_item['ground_truth'])
        
        for i in range(len(complexity_bins) - 1):
            if complexity_bins[i] <= gt_len < complexity_bins[i+1]:
                binned_neds[complexity_labels[i]].append(ned)
                break
    
    # Generate box plot by complexity category
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_to_plot = [binned_neds[label] for label in complexity_labels if binned_neds[label]]
    labels_to_plot = [label for label in complexity_labels if binned_neds[label]]
    
    bp = ax.boxplot(data_to_plot, tick_labels=labels_to_plot, patch_artist=True)
    
    # Apply styling
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_ylabel('OMR-NED Score', fontsize=12)
    ax.set_xlabel('Ground Truth Complexity', fontsize=12)
    ax.set_title('Model Performance by Score Complexity', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistical breakdown
    print("\nPerformance by Complexity Category:")
    print("-" * 60)
    for label in complexity_labels:
        if binned_neds[label]:
            mean_ned = np.mean(binned_neds[label])
            count = len(binned_neds[label])
            print(f"  {label:15s}  |  Mean NED: {mean_ned:.4f}  |  Samples: {count}")
    
else:
    print("Predictions not saved - enable SAVE_PREDICTIONS for error pattern analysis")

# %% [markdown]
# ## 14. Best & Worst Samples

# %%
# Identify and analyze best and worst performing samples
if SAVE_PREDICTIONS and 'predictions' in results:
    # Calculate NED for each sample with index
    samples_with_ned = []
    for idx, pred_item in enumerate(results['predictions']):
        ned = omr_ned(pred_item['prediction'], pred_item['ground_truth'])
        samples_with_ned.append({
            'index': idx,
            'ned': ned,
            'pred_len': len(pred_item['prediction']),
            'gt_len': len(pred_item['ground_truth'])
        })
    
    # Sort by NED score
    samples_with_ned.sort(key=lambda x: x['ned'])
    
    # Display top 5 best and worst samples
    print("=" * 70)
    print("TOP 5 BEST SAMPLES (Lowest NED)")
    print("=" * 70)
    for i, s in enumerate(samples_with_ned[:5], 1):
        print(f"{i}. Sample #{s['index']:3d}  |  NED: {s['ned']:.4f}  |  "
              f"Lengths: GT={s['gt_len']:4d}, Pred={s['pred_len']:4d}")
    
    print("\n" + "=" * 70)
    print("TOP 5 WORST SAMPLES (Highest NED)")
    print("=" * 70)
    for i, s in enumerate(samples_with_ned[-5:][::-1], 1):
        print(f"{i}. Sample #{s['index']:3d}  |  NED: {s['ned']:.4f}  |  "
              f"Lengths: GT={s['gt_len']:4d}, Pred={s['pred_len']:4d}")
    
    # Visualize extreme cases
    best_idx = samples_with_ned[0]['index']
    worst_idx = samples_with_ned[-1]['index']
    
    print(f"\nVisualizing best (#{best_idx}) and worst (#{worst_idx}) samples...")
    
    # Retrieve samples from dataset
    dataset_list = list(dataset)
    
    # Display best performing sample
    print(f"\nBEST SAMPLE (Index {best_idx}, NED: {samples_with_ned[0]['ned']:.4f})")
    show_sample(dataset_list[best_idx])
    
    # Display worst performing sample
    print(f"\nWORST SAMPLE (Index {worst_idx}, NED: {samples_with_ned[-1]['ned']:.4f})")
    show_sample(dataset_list[worst_idx])
    
else:
    print("Predictions not saved - enable SAVE_PREDICTIONS for detailed analysis")

# %% [markdown]
# ## 15. Export Results

# %%
from datetime import datetime
import csv

# Save results using benchmark runner
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"oemer_results_{timestamp}.json"

runner.save_results(results, filename)

print(f"Results saved to: {output_dir / filename}")

# Create summary CSV for analysis
if SAVE_PREDICTIONS and 'predictions' in results:
    csv_filename = filename.replace('.json', '.csv')
    csv_path = output_dir / csv_filename
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'NED', 'Pred_Length', 'GT_Length', 'Length_Diff'])
        
        for idx, pred_item in enumerate(results['predictions']):
            ned = omr_ned(pred_item['prediction'], pred_item['ground_truth'])
            pred_len = len(pred_item['prediction'])
            gt_len = len(pred_item['ground_truth'])
            
            writer.writerow([idx, f"{ned:.6f}", pred_len, gt_len, pred_len - gt_len])
    
    print(f"CSV summary saved to: {csv_path}")

# Display download instructions
if IN_COLAB:
    print("\nDownload Results:")
    print("  1. Open Files panel (left sidebar)")
    print(f"  2. Navigate to: {output_dir}")
    print("  3. Right-click and select Download")
else:
    print(f"\nResults saved to: {output_dir.absolute()}")

# %% [markdown]
# ## 16. Final Summary Report

# %%
# Generate comprehensive summary report
print("=" * 80)
print(" " * 25 + "FINAL BENCHMARK REPORT")
print("=" * 80)
print(f"\nModel:               {results['model_name']}")
print(f"Date:                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Dataset:             PRAIG/SMB (test split)")
print(f"Sample Count:        {metrics['total_samples']}")
print(f"Total Time:          {elapsed_time/60:.1f} minutes")
print(f"Avg Time/Sample:     {elapsed_time/len(dataset):.1f} seconds")

print("\n" + "-" * 80)
print("PERFORMANCE METRICS")
print("-" * 80)
print(f"Mean NED:            {metrics['mean_ned']:.4f}")
print(f"Median NED:          {metrics['median_ned']:.4f}")
print(f"Std Dev:             {metrics['std_ned']:.4f}")
print(f"Best Score:          {metrics['min_ned']:.4f}")
print(f"Worst Score:         {metrics['max_ned']:.4f}")

print("\n" + "-" * 80)
print("QUALITY INDICATORS")
print("-" * 80)
print(f"Perfect Matches:     {metrics['perfect_matches']} ({metrics['perfect_matches']/metrics['total_samples']*100:.1f}%)")
print(f"Errors:              {metrics['num_errors']} ({metrics['error_rate']*100:.1f}%)")
print(f"Success Rate:        {(1-metrics['error_rate'])*100:.1f}%")

# Qualitative interpretation
print("\n" + "-" * 80)
print("INTERPRETATION")
print("-" * 80)

if metrics['mean_ned'] < 0.1:
    quality = "EXCELLENT"
elif metrics['mean_ned'] < 0.3:
    quality = "GOOD"
elif metrics['mean_ned'] < 0.5:
    quality = "FAIR"
else:
    quality = "POOR"

print(f"Overall Quality:     {quality}")
print(f"\nThe model achieved a mean NED of {metrics['mean_ned']:.4f}, indicating that")
print(f"predictions differ from ground truth by {metrics['mean_ned']*100:.1f}% on average")
print("(measured by character-level edit distance normalized by ground truth length).")

print("\n" + "=" * 80)
print(f"Benchmark Complete. Results saved to: {output_dir}")
print("=" * 80)


