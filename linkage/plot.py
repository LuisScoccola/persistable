import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# https://stackoverflow.com/a/65604940/2171328
# https://stackoverflow.com/a/47166787/7128154
# https://matplotlib.org/3.3.3/api/collections_api.html#matplotlib.collections.PathCollection
# https://matplotlib.org/3.3.3/api/path_api.html#matplotlib.path.Path
# https://stackoverflow.com/questions/15876011/add-information-to-matplotlib-navigation-toolbar-status-bar
# https://stackoverflow.com/questions/36730261/matplotlib-path-contains-point
# https://stackoverflow.com/a/36335048/7128154
class StatusbarHoverManager:
    """
    Manage hover information for mpl.axes.Axes object based on appearing
    artists.

    Attributes
    ----------
    ax : mpl.axes.Axes
        subplot to show status information
    artists : list of mpl.artist.Artist
        elements on the subplot, which react to mouse over
    labels : list (list of strings) or strings
        each element on the top level corresponds to an artist.
        if the artist has items
        (i.e. second return value of contains() has key 'ind'),
        the element has to be of type list.
        otherwise the element if of type string
    cid : to reconnect motion_notify_event
    """
    def __init__(self, ax, xlabel, ylabel):
        assert isinstance(ax, mpl.axes.Axes)

        def hover(event):
            if event.inaxes != ax:
                return
            info = 'x={:.2f}, y={:.2f}'.format(event.xdata, event.ydata)
            ax.format_coord = lambda x, y: info
        cid = ax.figure.canvas.mpl_connect("motion_notify_event", hover)

        self.ax = ax
        self.cid = cid
        self.artists = []
        self.labels = []
        self.xlabel = xlabel
        self.ylabel = ylabel

    def add_artist_labels(self, artist, label):
        if isinstance(artist, list):
            assert len(artist) == 1
            artist = artist[0]

        self.artists += [artist]
        self.labels += [label]

        def hover(event):
            if event.inaxes != self.ax:
                return
            info = (str(self.xlabel)+'={:.3e}, ' + str(self.ylabel)+'={:.3e}').format(event.xdata, event.ydata)
            for aa, artist in enumerate(self.artists):
                cont, dct = artist.contains(event)
                if not cont:
                    continue
                inds = dct.get('ind')
                lbl = self.labels[aa]
                info += ';   ' + str(lbl)

            self.ax.format_coord = lambda x, y: info

        self.ax.figure.canvas.mpl_disconnect(self.cid)
        self.cid = self.ax.figure.canvas.mpl_connect(
            "motion_notify_event", hover)
