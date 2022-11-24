import glob
import os
import csv
import matplotlib.pyplot as plt
from typing import Any, Optional
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


from typing import TypeVar
import numpy as np
from numpy import typing as npt
import pandas as pd


plt.rcParams.update({
    "text.usetex": True,
    #"font.family": "Helvetica"
})



"""Credit to https://rob-hall.com/articles/plot-ewma/
"""
B = TypeVar("B", bound=npt.NBitBase)


def numpy_ewma_vectorized(data, alpha):
    df = pd.DataFrame(data.copy(), columns=["A"])
    return df.ewm(alpha=alpha).mean()["A"].to_numpy()


XDT = TypeVar("XDT", bound=np.number)
YDT = TypeVar("YDT", bound=np.number)


def plot_smoothed(
    *,
    ax: Optional[Axes] = None,
    x: Optional[npt.NDArray[XDT]] = None,
    y: npt.NDArray[YDT],
    label: Optional[str] = None,
    ewma_alpha: float = 0.1,
    subplot_kwds={},
    plot_kwds={}
):
    """Plot a line as well as its exponentially weighted moving average

    Args:
        ax (Axes, optional): The axes upon which to plot, otherwise a new
            figure will be created passing `subplot_kwds` to `plt.subplots`

        x (npt.NDArray[XDT], optional): sample x values. If not supplied an
            integer sequence will be used.

        y (npt.NDArray[YDT]): sample y values

        label (str, optional): The label to be applied to the smoothed line

        ewma_alpha (float, optional): the smoothing factor to be applied.

        subplot_kwds (dict[str, Any], optional): Keyword arguments supplied to
            matplotlib when generating an axis if `ax` is not specified.

        plot_kwds (dict[str, Any], optional): Keyword arguments supplementing
            those used in this routine. For expected behavior you want to
            avoid: label, alpha, linewidth, zorder.

    Returns:
        tuple[Axes, Line2D, Line2D]: a tuple consisting of (1) the axis,
        (2) the smoothed line, and (3) the noisy line.
    """
    if "label" in plot_kwds:
        raise ValueError("You probably didn't mean to put 'label' in 'plot_kwds'")

    ln_args = {"alpha": 0.3}
    ln_args.update(plot_kwds)

    ax = ax if ax is not None else plt.subplots(**subplot_kwds)[-1]
    x = x if x is not None else np.arange(len(y))
    (line,) = ax.plot(x, y, **ln_args)

    sm_args = {
        "alpha": 1,
        "linewidth": line.get_linewidth() * 1,
        "color": line.get_color(),
        "zorder": line.get_zorder() + 1,
    }
    sm_args.update(plot_kwds)

    y_ewma = numpy_ewma_vectorized(y, ewma_alpha)
    (smoothed_line,) = ax.plot(x, y_ewma, label=label, **sm_args)
    return (ax, smoothed_line, line)


CSV_FOLDER = "experiments/learning/inputs/room_npt"
XLABEL = "Timestep"
YLABEL = "Average Training Reward"
NAME = "Single Agent Training Reward over 1M Timesteps"
LABELS = ["R0", "R1", "R2"]
FILENAME = "single_agent_room_training_reward"

paths = sorted(glob.glob(os.path.join(CSV_FOLDER, "*.csv")))
print(paths)

x = []
y = []
for i, path in enumerate(paths):
    filename = os.path.splitext(os.path.basename(path))[0]
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")

        timestamps = []
        x.append([])
        y.append([])
        iterator = iter(reader)

        # Skip headers
        for row in iterator:
            break

        for row in iterator:
            timestamps.append(row[0])
            x[i].append(int(row[1]))
            y[i].append(float(row[2]))
min_len = min([len(x_) for x_ in x])
# update x and y to have the same length
x = [x_[:min_len] for x_ in x]
y = [y_[:min_len] for y_ in y]

plt.rcParams.update({"font.size": 14})
fig, ax = plt.subplots()
for i in range(len(x)):
    # plt.plot(x[i], y[i], label=LABELS[i])
    plot_smoothed(x=x[i], y=y[i], ax=ax, label=LABELS[i], ewma_alpha=0.01)
plt.title(NAME)
plt.xlim([0, 1000000])
# plt.ylim([0, 120])
plt.legend()
plt.grid()
plt.xlabel(XLABEL)
ax.ticklabel_format(style="plain")
# ax.xaxis.major.formatter.set_useOffset(1000000)
# ax.xaxis.major.formatter._useMathText = True
plt.xticks([500000, 1000000])
plt.ylabel(YLABEL)
plt.savefig("{}.png".format(FILENAME), bbox_inches="tight", dpi=300)
plt.show()
