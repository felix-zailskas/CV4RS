import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


# Define a custom formatting function to display percentages
def percentage_formatter(x, pos):
    return f"{100 * x:.0f}%"


def create_plot(data: dict, axis):
    # extract needed data
    f1_score = data["f1-score"]
    precision = data["precision"]
    recall = data["recall"]

    epochs = np.arange(len(f1_score)) + 1
    # Set the y-axis tick formatter to use the custom formatting function
    axis.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Value in %")
    axis.set_ylim(-0.05, 1.05)

    axis.plot(epochs, f1_score, label="F1-Score")
    axis.plot(epochs, precision, label="Precision")
    axis.plot(epochs, recall, label="recall")

    axis.grid()
    axis.legend()


def create_micro_macro_plot(
    data: dict, show: bool = True, save_micro: str = None, save_macro: str = None
):
    micro_fig, micro_ax = plt.subplots(figsize=(6, 4))
    macro_fig, macro_ax = plt.subplots(figsize=(6, 4))

    # Create plots for micro and macro data
    create_plot(data["micro avg"], axis=micro_ax)
    create_plot(data["macro avg"], axis=macro_ax)

    # adjust figure name etc.
    micro_ax.set_title("Micro Average")
    macro_ax.set_title("Macro Average")

    plt.tight_layout()

    if save_micro is not None:
        micro_fig.savefig(save_micro)
    if save_macro is not None:
        macro_fig.savefig(save_macro)

    if show:
        plt.show()
