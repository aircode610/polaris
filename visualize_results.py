"""Visualize research results by topic and category scores.

Usage:
    python visualize_results.py

To add another version for comparison:
    1. Place your results file in the results/ directory (e.g., raw_results_other.jsonl)
    2. Add a new entry to the 'versions' list in the main() function:
       ("raw_results_other.jsonl", "Other Version")
    3. Run the script - it will create individual visualizations for each version
       and a comparison chart if multiple versions are present.

The script expects JSONL files with the following structure per line:
    {
        "id": <topic_id>,
        "prompt": "<topic description>",
        "comprehensiveness": <score 0-1>,
        "insight": <score 0-1>,
        "instruction_following": <score 0-1>,
        "readability": <score 0-1>,
        "overall_score": <score 0-1>
    }
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_results(file_path: str) -> List[Dict]:
    """Load results from a JSONL file."""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def prepare_data(results: List[Dict], version_name: str = "Version 1") -> pd.DataFrame:
    """Convert results to a DataFrame suitable for visualization."""
    data = []
    for result in results:
        # Extract topic (use first 60 chars of prompt as topic identifier)
        prompt = result.get('prompt', 'Unknown')
        topic = prompt[:60] + "..." if len(prompt) > 60 else prompt
        topic_id = result.get('id', 'Unknown')
        
        # Add all score categories
        data.append({
            'topic': topic,
            'topic_id': topic_id,
            'topic_label': f"ID {topic_id}",  # For easier labeling in plots
            'full_prompt': prompt,  # Keep full prompt for reference
            'version': version_name,
            'Comprehensiveness': result.get('comprehensiveness', 0),
            'Insight': result.get('insight', 0),
            'Instruction Following': result.get('instruction_following', 0),
            'Readability': result.get('readability', 0),
            'Overall Score': result.get('overall_score', 0),
        })
    
    return pd.DataFrame(data)


def visualize_results(df: pd.DataFrame, output_path: Optional[str] = None):
    """Create visualizations for the results."""
    # Categories to visualize (excluding overall_score for detailed view)
    categories = ['Comprehensiveness', 'Insight', 'Instruction Following', 'Readability']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Research Results by Topic and Category', fontsize=16, fontweight='bold')
    
    # Get unique topics
    topics = df['topic'].unique()
    n_topics = len(topics)
    x_pos = range(n_topics)
    
    # Plot each category
    for idx, category in enumerate(categories):
        ax = axes[idx // 2, idx % 2]
        
        # Get data for this category
        category_data = []
        topic_labels = []
        colors = []
        
        for topic in topics:
            topic_df = df[df['topic'] == topic]
            for _, row in topic_df.iterrows():
                category_data.append(row[category])
                topic_labels.append(row['topic_label'])
                colors.append(f"C{len(category_data) % 10}")
        
        # Create bar plot
        bars = ax.bar(range(len(category_data)), category_data, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
        
        ax.set_title(f'{category} Scores', fontweight='bold')
        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Score')
        ax.set_xticks(range(len(topic_labels)))
        ax.set_xticklabels(topic_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def visualize_comparison(df_list: List[pd.DataFrame], output_path: Optional[str] = None):
    """Create side-by-side comparison visualization for multiple versions."""
    categories = ['Comprehensiveness', 'Insight', 'Instruction Following', 'Readability', 'Overall Score']
    
    fig, axes = plt.subplots(len(categories), 1, figsize=(14, 4 * len(categories)))
    fig.suptitle('Results Comparison Across Versions', fontsize=16, fontweight='bold')
    
    # Get all unique topics across all versions
    all_topics = set()
    for df in df_list:
        all_topics.update(df['topic'].unique())
    all_topics = sorted(list(all_topics))
    
    for cat_idx, category in enumerate(categories):
        ax = axes[cat_idx]
        
        # Prepare data for grouped bar chart
        x = range(len(all_topics))
        width = 0.8 / len(df_list)
        
        for version_idx, df in enumerate(df_list):
            version_name = df['version'].iloc[0]
            values = []
            
            for topic in all_topics:
                topic_df = df[df['topic'] == topic]
                if not topic_df.empty:
                    values.append(topic_df[category].iloc[0])
                else:
                    values.append(0)
            
            offset = (version_idx - len(df_list) / 2 + 0.5) * width
            bars = ax.bar([xi + offset for xi in x], values, width, 
                         label=version_name, alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}',
                           ha='center', va='bottom', fontsize=7)
        
        ax.set_title(f'{category} Scores', fontweight='bold')
        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        # Get topic labels from first available dataframe
        topic_labels = []
        for topic in all_topics:
            found = False
            for df in df_list:
                topic_df = df[df['topic'] == topic]
                if not topic_df.empty:
                    topic_labels.append(topic_df['topic_label'].iloc[0])
                    found = True
                    break
            if not found:
                topic_labels.append('N/A')
        ax.set_xticklabels(topic_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison visualization to {output_path}")
    else:
        plt.show()


def visualize_average_by_topic(df_list: List[pd.DataFrame], output_path: Optional[str] = None):
    """Create visualization averaging all criteria per topic."""
    # Categories to average (excluding Overall Score to avoid double counting)
    categories = ['Comprehensiveness', 'Insight', 'Instruction Following', 'Readability']
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle('Average Score Across All Criteria by Topic', fontsize=16, fontweight='bold')
    
    # Get all unique topics across all versions
    all_topics = set()
    for df in df_list:
        all_topics.update(df['topic'].unique())
    all_topics = sorted(list(all_topics))
    
    # Prepare data for grouped bar chart
    x = range(len(all_topics))
    width = 0.8 / len(df_list)
    
    for version_idx, df in enumerate(df_list):
        version_name = df['version'].iloc[0]
        avg_scores = []
        
        for topic in all_topics:
            topic_df = df[df['topic'] == topic]
            if not topic_df.empty:
                # Calculate average across all criteria for this topic
                topic_avg = topic_df[categories].mean(axis=1).iloc[0]
                avg_scores.append(topic_avg)
            else:
                avg_scores.append(0)
        
        offset = (version_idx - len(df_list) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], avg_scores, width, 
                     label=version_name, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, avg_scores):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=9)
    
    # Get topic labels from first available dataframe
    topic_labels = []
    for topic in all_topics:
        found = False
        for df in df_list:
            topic_df = df[df['topic'] == topic]
            if not topic_df.empty:
                topic_labels.append(topic_df['topic_label'].iloc[0])
                found = True
                break
        if not found:
            topic_labels.append('N/A')
    
    ax.set_xlabel('Topic ID', fontweight='bold')
    ax.set_ylabel('Average Score (All Criteria)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(topic_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved average by topic visualization to {output_path}")
    else:
        plt.show()


def visualize_average_by_category(df_list: List[pd.DataFrame], output_path: Optional[str] = None):
    """Create visualization averaging all topics per category."""
    categories = ['Comprehensiveness', 'Insight', 'Instruction Following', 'Readability', 'Overall Score']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle('Average Score Across All Topics by Category', fontsize=16, fontweight='bold')
    
    # Prepare data for grouped bar chart
    x = range(len(categories))
    width = 0.8 / len(df_list)
    
    for version_idx, df in enumerate(df_list):
        version_name = df['version'].iloc[0]
        avg_scores = []
        
        for category in categories:
            # Calculate average across all topics for this category
            if category in df.columns:
                category_avg = df[category].mean()
                avg_scores.append(category_avg)
            else:
                avg_scores.append(0)
        
        offset = (version_idx - len(df_list) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], avg_scores, width, 
                     label=version_name, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, avg_scores):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Category', fontweight='bold')
    ax.set_ylabel('Average Score (All Topics)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved average by category visualization to {output_path}")
    else:
        plt.show()


def visualize_overall_average(df_list: List[pd.DataFrame], output_path: Optional[str] = None):
    """Create visualization with overall average score (all topics and all categories)."""
    # Categories to average (excluding Overall Score to avoid double counting)
    categories = ['Comprehensiveness', 'Insight', 'Instruction Following', 'Readability']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Overall Average Score (All Topics & All Categories)', fontsize=16, fontweight='bold')
    
    # Calculate overall average for each version
    version_names = []
    overall_averages = []
    
    for df in df_list:
        version_name = df['version'].iloc[0]
        version_names.append(version_name)
        
        # Calculate average across all categories and all topics
        overall_avg = df[categories].mean().mean()
        overall_averages.append(overall_avg)
    
    # Create bar chart
    bars = ax.bar(version_names, overall_averages, alpha=0.7, edgecolor='black', width=0.6)
    
    # Add value labels on bars
    for bar, val in zip(bars, overall_averages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Overall Average Score', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved overall average visualization to {output_path}")
    else:
        plt.show()


def main():
    """Main function to visualize results."""
    results_dir = Path("results")
    
    # Define versions to visualize
    # Format: (file_path, version_name)
    versions = [
        ("raw_results_open.jsonl", "Open Deep Research"),
        ("raw_results_polaris.jsonl", "Polaris"),
    ]
    
    all_dataframes = []
    
    # Load all versions
    for file_name, version_name in versions:
        results_file = results_dir / file_name
        if not results_file.exists():
            print(f"Warning: {results_file} not found, skipping...")
            continue
        
        print(f"Loading results from {results_file}...")
        results = load_results(str(results_file))
        print(f"Loaded {len(results)} results for {version_name}")
        
        # Prepare data
        df = prepare_data(results, version_name=version_name)
        all_dataframes.append(df)
        
        # Create individual visualization for this version
        print(f"Creating visualization for {version_name}...")
        output_name = f"visualization_{version_name.lower().replace(' ', '_')}.png"
        visualize_results(df, output_path=results_dir / output_name)
    
    # If we have multiple versions, create comparison
    if len(all_dataframes) > 1:
        print("Creating comparison visualization...")
        visualize_comparison(all_dataframes, output_path=results_dir / "comparison.png")
        print("Creating average by topic visualization...")
        visualize_average_by_topic(all_dataframes, output_path=results_dir / "comparison_avg_by_topic.png")
        print("Creating average by category visualization...")
        visualize_average_by_category(all_dataframes, output_path=results_dir / "comparison_avg_by_category.png")
        print("Creating overall average visualization...")
        visualize_overall_average(all_dataframes, output_path=results_dir / "comparison_overall_average.png")
    elif len(all_dataframes) == 0:
        print("Error: No results files found!")
        return
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
