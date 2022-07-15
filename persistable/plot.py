import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm


def plot_hilbert_function(xs, ys, max_dim, dimensions, logscale=True):
    normalized_dimensions = np.minimum(max_dim, dimensions) / max_dim
    # normalized_dimensions = normalized_dimensions[:,::-1]
    colormap = cm.binary

    _, ax = plt.subplots()
    ax.set_xlim((xs[0], xs[-1]))
    ax.set_ylim((ys[0], ys[-1]))

    for i in range(xs.shape[0] - 1):
        for j in range(ys.shape[0] - 1):
            rect = patches.Rectangle(
                (xs[i], ys[j]),
                xs[i + 1] - xs[i],
                ys[j + 1] - ys[j],
                facecolor=colormap(normalized_dimensions[j, i]),
            )
            ax.add_patch(rect)
    if logscale:
        ax.set_xscale("log")
    return ax


# a combination of:
# https://stackoverflow.com/a/65604940/2171328
# https://stackoverflow.com/a/47166787/7128154
# https://matplotlib.org/3.3.3/api/collections_api.html#matplotlib.collections.PathCollection
# https://matplotlib.org/3.3.3/api/path_api.html#matplotlib.path.Path
# https://stackoverflow.com/questions/15876011/add-information-to-matplotlib-navigation-toolbar-status-bar
# https://stackoverflow.com/questions/36730261/matplotlib-path-contains-point
# https://stackoverflow.com/a/36335048/7128154
class StatusbarHoverManager:
    def __init__(self, ax, label):
        assert isinstance(ax, mpl.axes.Axes)

        def hover(event):
            if event.inaxes != ax:
                return
            info = "x={:.2f}, y={:.2f}".format(event.xdata, event.ydata)
            ax.format_coord = lambda x, y: info

        cid = ax.figure.canvas.mpl_connect("motion_notify_event", hover)

        self.ax = ax
        self.cid = cid
        self.artists = []
        self.labels = []
        self.label = label

    def add_artist_labels(self, artist, label):
        if isinstance(artist, list):
            assert len(artist) == 1
            artist = artist[0]

        self.artists += [artist]
        self.labels += [label]

        def hover(event):
            if event.inaxes != self.ax:
                return
            # info = (str(self.xlabel)+'={:.3e}, ' + str(self.ylabel)+'={:.3e}').format(event.xdata, event.ydata)
            info = self.label.format(event.xdata)
            for aa, artist in enumerate(self.artists):
                cont, dct = artist.contains(event)
                if not cont:
                    continue
                inds = dct.get("ind")
                lbl = self.labels[aa]
                info += ";   " + str(lbl)

            self.ax.format_coord = lambda x, y: info

        self.ax.figure.canvas.mpl_disconnect(self.cid)
        self.cid = self.ax.figure.canvas.mpl_connect("motion_notify_event", hover)
