import numpy as np
import matplotlib.pyplot as plt


def plotProminenceVineyard(pds, n_parameters, initial_parameter, final_parameter, s = 5, kappa = None, tau = None, y_lim = [0.1,10], log = True, height = 2, width = 3, decimals = 2):

    def prominences(bd) :
        TOL = 1e-15 ; INF = 1e15
        pers = np.abs(bd[:,0] - bd[:,1])
        pers = pers[pers > TOL]
        pers = pers[pers < INF]
        return pers

    pr = [ prominences(pd) for pd in pds]
    
    prominence_points = []
    for i in range(n_parameters) :
        for y in pr[i] :
            prominence_points.append([i,y])
    prominence_points = np.array(prominence_points)

    fig1, ax1 = plt.subplots() ; fig1.set_figheight(height) ; fig1.set_figwidth(width) ; plt.xlabel("density threshold")
    plt.ylabel("prominences") ; _ = plt.ylim(y_lim) ; _ = plt.xlim([0,n_parameters])

    if log :
        ax1.set_yscale('log')
    
    _ = plt.xticks(np.linspace(0, n_parameters, num=4), np.rint((10**decimals)*np.linspace(initial_parameter, final_parameter, num=4))/(10**decimals))
    _ = plt.scatter(prominence_points.T[0], prominence_points.T[1], s = s, alpha = 1, zorder=2, rasterized = True)

    if kappa != None and tau != None:
    
        x_,y_ = np.searchsorted(np.linspace(initial_parameter, final_parameter, num=n_parameters), kappa), tau
        arrowprops={'arrowstyle': '-', 'ls':'-', 'color':'grey'}
        _ = plt.annotate("$\\kappa$", xy=(x_,y_), xytext=(x_, -0.01), 
                     textcoords=plt.gca().get_xaxis_transform(),
                     arrowprops=arrowprops,
                     va='top', ha='center',c="black")
        _ = plt.annotate("$\\tau$", xy=(x_,y_), xytext=(-0.02, y_), 
                     textcoords=plt.gca().get_yaxis_transform(),
                     arrowprops=arrowprops,
                     va='center', ha='right',c="black")
        _  = plt.scatter([x_], [y_], marker='o',c="orange", s = 20)