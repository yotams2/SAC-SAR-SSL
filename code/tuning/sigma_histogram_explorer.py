import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=str, required=True, help="Path to npz cache file")
    args = parser.parse_args()

    # Load cache
    data = np.load(args.cache)
    sigmas = data['sigmas']
    bin_edges = data['bin_edges']
    counts = data['counts']

    num_sigmas, num_batches, num_bins = counts.shape
    
    if num_batches != 4:
        print(f"Warning: Expected 4 batches, but found {num_batches} in cache.")
        
    global_max = counts.max()
    bin_width = bin_edges[1] - bin_edges[0]

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(bottom=0.25, hspace=0.3, wspace=0.2)
    
    axs = axs.flatten()
    
    # Store bar containers for updating
    bars_list = []
    
    # Initial sigma index
    init_idx = 0
    
    for i in range(min(num_batches, 4)):
        ax = axs[i]
        # Align 'edge' with bin_edges[:-1] exactly matches the bin intervals
        bars = ax.bar(bin_edges[:-1], counts[init_idx, i], width=bin_width, align='edge', edgecolor='black', alpha=0.7)
        bars_list.append(bars)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, global_max * 1.05)
        ax.set_title(f"Batch {i+1}")
        ax.set_xlabel("w_norm")
        ax.set_ylabel("Count")
        
    fig.suptitle(f"Sigma Explorer - Sigma: {sigmas[init_idx]:.4f}", fontsize=14)

    # Create slider
    ax_slider = fig.add_axes([0.15, 0.1, 0.65, 0.03])
    sigma_slider = Slider(
        ax=ax_slider,
        label='Sigma Index',
        valmin=0,
        valmax=num_sigmas - 1,
        valinit=init_idx,
        valstep=1,
        valfmt='%d'
    )
    
    # Add annotation for actual sigma value next to slider
    val_text = fig.text(0.82, 0.11, f"Sigma = {sigmas[init_idx]:.4f}", verticalalignment='bottom')

    # Add toggles
    ax_check_logx = fig.add_axes([0.02, 0.5, 0.12, 0.1])
    check_logx = CheckButtons(ax_check_logx, ['Log X-Axis'], [False])
    
    ax_check_y = fig.add_axes([0.02, 0.35, 0.12, 0.1])
    check_y = CheckButtons(ax_check_y, ['Autoscale Y'], [False])
    
    state = {'autoscale_y': False}

    # Update function
    def update(val):
        idx = int(sigma_slider.val)
        val_text.set_text(f"Sigma = {sigmas[idx]:.4f}")
        fig.suptitle(f"Sigma Explorer - Sigma: {sigmas[idx]:.4f}", fontsize=14)
        
        for i in range(min(num_batches, 4)):
            for rect, h in zip(bars_list[i], counts[idx, i]):
                rect.set_height(h)
                
        # Optional: autoscale Y if toggle is active
        if state['autoscale_y']:
            current_max = counts[idx].max()
            new_ylim = current_max * 1.05 if current_max > 0 else 1.0
            for i in range(min(num_batches, 4)):
                axs[i].set_ylim(0, new_ylim)
        else:
            for i in range(min(num_batches, 4)):
                axs[i].set_ylim(0, global_max * 1.05)
                
        fig.canvas.draw_idle()

    sigma_slider.on_changed(update)

    def toggle_logx(label):
        is_log = check_logx.get_status()[0]
        for i in range(min(num_batches, 4)):
            if is_log:
                axs[i].set_xscale('log')
                # when log scale is active, 0 is invalid
                axs[i].set_xlim(1e-3, 1.0)
            else:
                axs[i].set_xscale('linear')
                axs[i].set_xlim(0, 1.0)
        fig.canvas.draw_idle()
        
    def toggle_y(label):
        state['autoscale_y'] = check_y.get_status()[0]
        update(sigma_slider.val)

    check_logx.on_clicked(toggle_logx)
    check_y.on_clicked(toggle_y)

    plt.show()

if __name__ == "__main__":
    main()
