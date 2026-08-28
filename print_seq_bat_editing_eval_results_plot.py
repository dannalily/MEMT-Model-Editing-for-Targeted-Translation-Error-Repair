import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob


def _get_x_label(input_dir: str) -> str:
    """Decide x-axis label based on input_dir contents."""
    low = (input_dir or "").lower()
    if "batch" in low:
        return "Batch Size"
    if "seq_" in low:
        return "Step"
    return "Step"  # default


# 定义固定的方法顺序
METHOD_ORDER = ['pre-edit', 'FT', 'ROME', 'MEMIT', 'AlphaEdit', 'UNKE', 'GRACE', 'WISE']

# 定义固定的方法颜色
METHOD_COLORS = {
    'pre-edit': '#000000',      # 黑色
    'FT': '#1E88E5',           # 蓝色
    'ROME': '#D81B60',         # 粉红色
    'MEMIT': '#FFC107',        # 黄色
    'AlphaEdit': '#004D40',    # 深青色
    'UNKE': '#8E24AA',         # 紫色
    'GRACE': '#43A047',        # 绿色
    'WISE': '#F4511E',         # 橙红色
}


def sort_methods(methods):
    """Sort methods according to METHOD_ORDER"""
    sorted_methods = []
    for method in METHOD_ORDER:
        if method in methods:
            sorted_methods.append(method)
    # Add any methods not in METHOD_ORDER at the end
    for method in methods:
        if method not in sorted_methods:
            sorted_methods.append(method)
    return sorted_methods


def get_method_color(method):
    """Get color for a method, with default fallback"""
    return METHOD_COLORS.get(method, '#666666')  # Default gray for unknown methods


def load_averaged_results(base_dir, translator, slang, tlang, metric, test_types, steps):
    """Load previously saved averaged results"""
    averaged_results = {}  # {test_type: {step: df}}
    
    for test_type in test_types:
        averaged_results[test_type] = {}
        
        for step in steps:
            file_path = (f'{base_dir}/{translator}/{slang}2{tlang}/'
                        f'{metric}_{test_type}_step{step}_avg.xlsx')
            
            if os.path.exists(file_path):
                df = pd.read_excel(file_path, index_col=0)
                averaged_results[test_type][step] = df
                print(f"✓ Loaded: {file_path}")
            else:
                print(f"✗ Not found: {file_path}")
    
    return averaged_results


def plot_score_combined(averaged_results, args, out_pair_langs, markers):
    """Plot Score in 2x3 grid: top row in-pair, bottom row out-of-pair"""
    test_types = [t for t in args.test_types if t in averaged_results and averaged_results[t]]
    if not test_types:
        print("No data available for score plot")
        return None
    
    x_label = _get_x_label(args.input_dir)
    
    # Create 2x3 grid
    # fig, axes = plt.subplots(2, len(test_types), figsize=(12, 6.5))
    fig, axes = plt.subplots(2, len(test_types), figsize=(4*len(test_types), 6.5))
    fig.suptitle(f'Score vs {x_label} - {args.translator} ({args.slang}→{args.tlang})', 
                 fontsize=20, y=0.98)

    if len(test_types) == 1:
        axes = axes.reshape(2, 1)

    # Get all editing methods and sort them (INCLUDING pre-edit)
    first_step_df = next(iter(next(iter(averaged_results.values())).values()))
    editing_methods = sort_methods(list(first_step_df.index))
    x_labels = [str(int(np.log2(s))) if s > 0 and s & (s-1) == 0 else str(s) for s in args.steps]
    
    # Plot each test type
    for idx, test_type in enumerate(test_types):
        # --- Top row: In-pair ---
        ax_top = axes[0, idx]
        
        for edit_idx, editing in enumerate(editing_methods):
            in_pair_scores = []
            
            for step in args.steps:
                if step in averaged_results[test_type]:
                    df = averaged_results[test_type][step]
                    if editing in df.index:
                        in_pair_col = f'{args.tlang}_{test_type[:3]}_score'
                        if in_pair_col in df.columns:
                            val = df.loc[editing, in_pair_col]
                            in_pair_scores.append(np.nan if val == -100 else val)
                        else:
                            in_pair_scores.append(np.nan)
                    else:
                        in_pair_scores.append(np.nan)
                else:
                    in_pair_scores.append(np.nan)
            
            color = get_method_color(editing)
            
            # pre-edit: dashed line without marker
            if editing == 'pre-edit':
                ax_top.plot(args.steps, in_pair_scores,
                           linestyle='--', color=color, label=editing,
                           linewidth=2.5, zorder=10)
            else:
                ax_top.plot(args.steps, in_pair_scores, 
                           marker=markers[edit_idx % len(markers)], 
                           color=color, 
                           label=editing, 
                           linewidth=2.5, 
                           markersize=8)
        
        ax_top.set_ylabel('Score', fontsize=16, fontweight='bold')
        if test_type == 'generality':
            ax_top.set_title(f'Generalization (In)', fontsize=18, fontweight='bold')
        else:
            ax_top.set_title(f'{test_type.capitalize()} (In)', fontsize=18, fontweight='bold')
        ax_top.grid(True, alpha=0.3)
        ax_top.set_xscale('log')
        ax_top.set_xticks(args.steps)
        ax_top.set_xticklabels(x_labels, fontsize=16)
        ax_top.tick_params(axis='both', labelsize=16)
        # No y-limit for score plots
        
        # --- Bottom row: Out-of-pair ---
        ax_bottom = axes[1, idx]
        
        for edit_idx, editing in enumerate(editing_methods):
            out_pair_scores = []
            
            for step in args.steps:
                if step in averaged_results[test_type]:
                    df = averaged_results[test_type][step]
                    if editing in df.index:
                        out_pair_cols = [f'{lang}_{test_type[:3]}_score' 
                                        for lang in out_pair_langs 
                                        if f'{lang}_{test_type[:3]}_score' in df.columns]
                        if out_pair_cols:
                            out_pair_avg = df.loc[editing, out_pair_cols].mean()
                            out_pair_scores.append(np.nan if out_pair_avg == -100 else out_pair_avg)
                        else:
                            out_pair_scores.append(np.nan)
                    else:
                        out_pair_scores.append(np.nan)
                else:
                    out_pair_scores.append(np.nan)
            
            color = get_method_color(editing)
            
            # pre-edit: dashed line without marker
            if editing == 'pre-edit':
                ax_bottom.plot(args.steps, out_pair_scores,
                              linestyle='--', color=color, label=editing,
                              linewidth=2.5,zorder=10)
            else:
                ax_bottom.plot(args.steps, out_pair_scores, 
                              marker=markers[edit_idx % len(markers)], 
                              color=color, 
                              label=editing, 
                              linewidth=2.5, 
                              markersize=8)
        
        ax_bottom.set_xlabel(x_label, fontsize=16, fontweight='bold')
        ax_bottom.set_ylabel('Score', fontsize=16, fontweight='bold')
        if test_type == 'generality':
            ax_bottom.set_title(f'Generalization (Out)', fontsize=18, fontweight='bold')
        else:
            ax_bottom.set_title(f'{test_type.capitalize()} (Out)', fontsize=18, fontweight='bold')
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.set_xscale('log')
        ax_bottom.set_xticks(args.steps)
        ax_bottom.set_xticklabels(x_labels, fontsize=16)
        ax_bottom.tick_params(axis='both', labelsize=16)
        # No y-limit for score plots

    # Get handles and labels from any subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Shared legend below all subplots
    fig.legend(handles, labels, loc='lower center', 
              ncol=4, fontsize=18,
              bbox_to_anchor=(0.5, -0.1), frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    output_path = f'{args.output_dir}/plot_score_combined_{args.translator}_{args.slang}2{args.tlang}_{args.metric}.pdf'
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Basic parameters
    parser.add_argument('--translator', type=str, default='Qwen2-5-3B-Instruct',
                        help='Name of the translator model')
    parser.add_argument('--slang', type=str, default='en',
                        help='Source language')
    parser.add_argument('--tlang', type=str, default='zh',
                        help='Target language')
    parser.add_argument('--metric', type=str, default='bleurt',
                        help='Evaluation metric')
    parser.add_argument(
        '--editing-type',
        type=str,
        default='sequential',
        choices=['sequential', 'batch'],
        help='Type of editing process'
    )
    
    # Test types
    parser.add_argument('--test_types', type=str, nargs='+',
                        default=['reliability', 'generality','locality', 'flores' ,'mmlu'],
                        help='List of test types')
    
    # Languages
    parser.add_argument('--tlangs', type=str, nargs='+',
                        default=['en', 'zh', 'de', 'fr', 'ja', 'ar'],
                        help='List of all target languages')
    
    # Steps
    parser.add_argument('--steps', type=int, nargs='+',
                        default=[],
                        help='List of editing steps or batch sizes (x-axis)')
    
    # Input/Output directories
    parser.add_argument('--input_dir', type=str, 
                        help='Input directory with collected results')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for plots')
    
    # Plot settings
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved plots')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.steps:
        if args.editing_type=='batch':
            args.steps = [1,4,16,32,64,128,256]
        else:
            args.steps = [1,4,16,64,256,1024]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_tlangs = [l for l in args.tlangs if l != args.slang]
    out_pair_langs = [l for l in all_tlangs if l != args.tlang]
    
    print("=" * 80)
    print("PLOTTING SCORE RESULTS (2x3 Grid)")
    print("=" * 80)
    print(f"Translator: {args.translator}")
    print(f"Language pair: {args.slang} -> {args.tlang}")
    print(f"In-pair: {args.slang}-{args.tlang}")
    print(f"Out-pair: {args.slang}-{out_pair_langs} (average)")
    print(f"Metric: {args.metric}")
    print(f"Test types: {args.test_types}")
    print(f"Steps: {args.steps}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    print()

    # Load averaged results
    print("Loading data...")
    print("-" * 80)
    averaged_results = load_averaged_results(
        base_dir=args.input_dir,
        translator=args.translator,
        slang=args.slang,
        tlang=args.tlang,
        metric=args.metric,
        test_types=args.test_types,
        steps=args.steps
    )
    print()

    # Check if we have any data
    if not any(averaged_results.values()):
        print("ERROR: No data loaded.")
        return

    # Set plot style
    plt.style.use('default')
    plt.rcParams.update({
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'grid.color': 'lightgray'
    })
    
    # Define markers
    markers = ['o', 's', '^', 'D', 'v', 'p', 'P', 'X']

    # Generate combined score plot
    print("Generating score plot (2x3)...")
    print("-" * 80)
    
    plot_score_combined(averaged_results, args, out_pair_langs, markers)

    print()
    print("=" * 80)
    print("PLOTTING COMPLETE!")
    print("=" * 80)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()