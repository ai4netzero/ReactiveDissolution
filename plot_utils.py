import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def matshow(data, title=None, figsize=(10, 10), scale=False,
            cmap='coolwarm', save_as=None):
    if scale:
        vmin = 0
        vmax = 1
    else:
        vmin = None
        vmax = None
    
    plt.figure(figsize=figsize)
    hm = sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=16)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as)

    plt.show()
    return hm


def plot_prediction(pred, true, title=None, figsize=(16, 6), scale=False,
                    diff=False, cmap='coolwarm', save_as=None):
    if scale:
        vmin = 0
        vmax = 1
    else:
        vmin = true.min()
        vmax = true.max()

    if diff:
        n_plots = 3
        figsize = (24, 6)
    else:
        n_plots = 2
        
    _, axes = plt.subplots(1, n_plots, figsize=figsize)
    sns.heatmap(pred, ax=axes[0], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_title(f"Predicted ({title})")

    sns.heatmap(true, ax=axes[1], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_title(f"Reference ({title})")

    if diff:
        sns.heatmap(pred-true, ax=axes[2], center=0, cmap="seismic_r")
        axes[2].set_title(f"Difference (Pred - True) ({title})")
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(wspace=0.01)

    if save_as:
        plt.savefig(save_as)

    plt.show()
    return axes
    

def plot_correlations(pred, true, title=None, figsize=(8, 8), save_as=None):
    pred = pred.flatten()
    true = true.flatten()
    
    plt.figure(figsize=figsize)
    corr_plot = sns.scatterplot(x=pred, y=true, color='g')
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.title(title, fontsize=16)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as)

    plt.show()
    return corr_plot
    

def plot_residuals(pred, true, title=None, figsize=(10, 10), save_as=None):
    pred = pred.flatten()
    true = true.flatten()
    
    scatter_kws = {
        's': np.abs(pred-true)+1,  
    }
    
    plt.figure(figsize=figsize)
    res_plot = sns.residplot(x=pred, y=true, color='r', scatter_kws=scatter_kws)
    
    plt.xlabel("Pred")
    plt.ylabel("Residual")
    plt.title(title, fontsize=16)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as)

    plt.show()
    return res_plot


def plot_diff(pred, true, title=None, figsize=(16, 16),
              cmap='seismic', save_as=None):
        
    plt.figure(figsize=figsize)
    diff = sns.heatmap(pred-true, center=0, cmap=cmap)
    diff.set_title(f"Difference ({title})")
    diff.set_xticks([])
    diff.set_yticks([])
    
    if save_as:
        plt.savefig(save_as)

    plt.show()
    return diff


def plot_features(images, title=None, cols=3, save_as=None):
    n = len(images)
    rows = int(np.ceil(n / cols))
    
    _, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes_flat = np.ravel(axes)
    plt.suptitle(title, fontsize=16)
    
    for i in range(n, len(axes_flat)):
        axes_flat[i].axis('off')
    
    # Display each plot in the grid
    for i, (name, image) in enumerate(images.items()):
        sns.heatmap(image, cmap='viridis', ax=axes_flat[i])
        axes_flat[i].set_title(name)
        axes_flat[i].set_xticks([])
        axes_flat[i].set_yticks([])
    
    plt.tight_layout()        
    
    if save_as:
        plt.savefig(save_as)

    plt.show()
    

def plot_corrcoef(results_dict, n_files, mode="train", timesteps=100,
                  skip=1, save_files=False):
    preds_key = f"preds_{mode}"
    masks_key = f"masks_{mode}"

    if mode == "train":
        mode = 'Training'
    elif mode == "val":
        mode = 'Validation'

    for file in range(n_files):
        plt.figure(figsize=(8, 6))
        
        for key in results_dict:    
            preds = results_dict[key][preds_key]
            masks = results_dict[key][masks_key]
        
            min_r = file * timesteps
            max_r = min_r + timesteps
            
            scores = [np.corrcoef(masks[i], preds[i])[0, 1]
                      for i in range(min_r, max_r, skip)]           
            sns.lineplot(
                x=np.arange(0, timesteps, skip),
                y=np.clip(scores, -1, 1),
                label=key)
            
        plt.title(f"$R^2$-Scores for {mode} Model {file + 1} (Pe = 1 | K = 1)")
        plt.legend()
        plt.tight_layout()

        if save_files:
            plt.savefig(f"r2_scores_{mode.lower()}_model_{file + 1}.png")

        plt.show()