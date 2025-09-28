import matplotlib.pyplot as plt
import numpy as np

def regime_heatmap(dates, regimes, title="Regime Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 2))
    regimes = np.asarray(regimes)
    ax.imshow(regimes.reshape(1, -1), aspect="auto")
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, len(dates)-1, 6, dtype=int))
    ax.set_xticklabels([dates[i].strftime("%Y-%m") for i in ax.get_xticks()])
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax
