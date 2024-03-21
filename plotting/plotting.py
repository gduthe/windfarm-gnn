import numpy as np
import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
import pandas as pd
from sklearn.metrics import r2_score
from graph_farms.utils import get_mean_values
from windrose import WindroseAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

#formatting settings
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['axes.unicode_minus'] = False

def set_ax_to_sci_format(ax_axis):
    formatter = ticker.FormatStrFormatter('%4.1e')
    ax_axis.set_major_formatter(formatter)

def _back_drop(df, x_prm, y_prm, bins, percentile=10):
    """
    Function to obtain from a dataframe the binned results
    :param df:
    :param x_prm:
    :param y_prm:
    :param bins:
    :param percentile:
    :return:
    """
    labels = range(len(bins) - 1)
    df['binned'] = pd.cut(df[x_prm], bins, labels=labels)

    result = []
    for label in labels:
        data = df[y_prm][df['binned'] == label]
        data = data[~np.isnan(data)]
        if len(data) > 0:
            result.append(np.percentile(data, percentile))
        else:
            result.append(np.nan)
    return result

def plot_shaded_percentile(df,
                           x,
                           y,
                           bins,
                           color=None,
                           shaded_surfaces=[(25, 75), (10, 90)],
                           alpha=0.3):
    """
    Plot a shaded percentile plot of the data inside the Dynawind frame.

    E.g. the example below provides a shaded percentile plot of the mean_power versus the mean windspeed

    .. code-block:: python

       dwf.plot_shaded_percentile('mean_windspeed', 'mean_power', bins = range(0,35))

    The center line is the median (50-th) percentile of the the data inside the bin.

    :param x: string specifying a parameter in the dwf
    :param y: string specifying a parameter in the dwf
    :param bins: Binedges (Note: Not centers!) to be used to bin the data into
    :param shaded_surfaces: List of tuples specifying the upper and lower bounds of the shaded areas. E.g by default
    [(25,75),(10,90)] indicates a surface between the 25th and 75th percentile and the 10th and 90th percentile
    :param alpha: alpha or opacity of the shaded area.
    :return:
    """

    self = df

    x_values = [(a + b) / 2.0 for a, b in zip(bins, bins[1:])]
    center_line = _back_drop(self, x, y, bins=bins, percentile=50)
    fig, ax = plt.subplots()

    if color is None:
        for surface in shaded_surfaces:
            bottom = np.array(_back_drop(self, x, y, bins, percentile=surface[0]))
            top = np.array(_back_drop(self, x, y, bins, percentile=surface[1]))

            plt.fill_between(x_values, bottom, top, facecolor=self.color, edgecolor=self.color, linestyle=':',
                             alpha=alpha)

        plt.plot(x_values, center_line, color=self.color, linestyle='-', marker='x', label=self.location)

        plt.grid(which='both', linestyle=':')
    else:
        for surface in shaded_surfaces:
            bottom = np.array(_back_drop(self, x, y, bins, percentile=surface[0]))
            top = np.array(_back_drop(self, x, y, bins, percentile=surface[1]))

            ax.fill_between(x_values, bottom, top, facecolor=color, edgecolor=color, linestyle=':', alpha=alpha)

            ax.plot(x_values, center_line, color=color, linestyle='-', marker='x', label=self.location)

        plt.grid(which='both', linestyle=':')
        return ax


def plot_vs_x(dataset:torch_geometric.data.Dataset, y:list, y_pred, x_axis:str, turb, layout_idx:int, length:int, dpi=600, 
              y_pred2=None, show_farm=False, ws_filter=None, wd_filter=None, labels=None, shaded_surfaces=[(25, 75), (10, 90)], plot_graph=True):
    """
    Plot the results of a machine learning model on a test dataset.

    Parameters
    ----------
    y : list
        A list of true values for the output of the model.
    y_pred : list
        A list of predicted values for the output of the model.
    x_axis : str
        A string specifying the type of x-axis to use in the plot. The options are 'ws' (wind speed) or 'wd' (wind direction).
    turb : int
        An integer representing the index of the turbine for which the results are being plotted.
    length : int
        The length of the test dataset.

    Returns
    -------
    fig, ax
        This function creates a subplot with 4 rows and 2 columns and plots the results of the model for each output variable.

    """
    matplotlib.rcParams['axes.linewidth'] = 1.5
    if labels is None:
        labels = ['True', 'Pred']
    ws = []
    wd = []
    start_idx = layout_idx*length
    end_idx = layout_idx*length + length
    idxs = []
    for d in range(start_idx, end_idx):
        if ws_filter is not None:
            if dataset.__getitem__(d).globals[0][0] < ws_filter[0] or dataset.__getitem__(d).globals[0][0] > ws_filter[1]:
                continue
        if wd_filter is not None:
            if dataset.__getitem__(d).globals[0][1] < wd_filter[0] or dataset.__getitem__(d).globals[0][1] > wd_filter[1]:
                continue
        ws += [((dataset.__getitem__(d).globals[0][0]).item())]
        wd += [((dataset.__getitem__(d).globals[0][1]).item())]
        idxs += [d]
    
    # adjust the wind direction for plotting (0° is north)
    polar_wd = (270 - np.array(wd) + 180) % 360
    
    if show_farm:                    
        # change the projection to polar if the x_axis is wind direction
        if x_axis == 'wd':
            fig, axs =  plt.subplot_mosaic('AAAA;BCDE;FGHI', per_subplot_kw={('B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'): {"projection": "polar"}}, 
                                           gridspec_kw={'hspace': 0.6}, height_ratios=[1.04, 1.02, 1.02], figsize=(8, 6), dpi=dpi)
            gen_marker_size = 40
            turb_marker_size = 10
        else:
            fig, axs =  plt.subplot_mosaic('AA;BC;DE;FG;HI', figsize=(8, 11), height_ratios=[1.03, 1.0, 1.0, 1.0, 1.0],
                                           dpi=dpi)
            gen_marker_size = 100
            turb_marker_size = 16
        coords = dataset.__getitem__(start_idx).pos
        graph = dataset.__getitem__(start_idx)
        ax_list = axs.values()
    else:
        fig, axs = plt.subplots(4, 2, figsize=(8, 12), dpi=dpi)
        ax_list = axs.flatten()

    string_list = ['Power [$kW$]', 'Wind speed [$m.s^{-1}$]', 'Turbulence intensity [$\%$]',
                   'DEL flap [$kN.m$]', 'DEL edge [$kN.m$]', 'DEL FA [$kN.m$]', 'DEL SS [$kN.m$]',
                   'DEL torsion [$kN.m$]']
    

    if not isinstance(turb, list):
        turb = [turb]
    assert len(turb) < 4

    # color_picker_pred = ['tab:blue', 'darkorange', 'limegreen']
    # color_picker_true = ['teal','lightcoral', 'forestgreen']
    wt_markers = ['^', 'o', '+']
    palette = sns.color_palette("Paired")
    color_picker_true = [palette[1], palette[7], palette[2]]
    color_picker_pred = [palette[0], palette[6], palette[3]]
    color_picker_pred2 = [palette[9], palette[4], palette[10]]
    
    # color_picker_pred = [palette[6], palette[6], palette[3]]
    # color_picker_pred2 = ['darkred','darkred']

    h1, h2, h3, h4, h5, h6 = [], [], [], [], [], []
    farm_legend = []

    for i, wt in enumerate(turb):
        color = color_picker_true[i]
        color2 = color_picker_pred[i]
        color3 = color_picker_pred2[i]
        marker = wt_markers[i]
        for r, ax in enumerate(ax_list):
            if show_farm:
                if i == 0 and r==0:
                    # normalize the coords
                    coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
                    coords[:, 1] = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())
                    ax.scatter(coords[:, 0], coords[:, 1], s=gen_marker_size, c='grey', marker="2",linewidth=1, alpha=0.7)
                    node_pos_dict = {}
                    for i in range(graph.num_nodes):
                        node_pos_dict[i] = coords[i, :].tolist()
                    if plot_graph:
                        G = torch_geometric.utils.to_networkx(graph, to_undirected=True)
                        nx.draw_networkx_edges(G, node_pos_dict, ax=ax, width=0.5, alpha=0.15, edge_color='grey')
                        ax.margins(x=0, y=0.05)
                    else:
                        ax.margins(x=0, y=0.15)

                    ax.set_aspect('equal',  'datalim')
                    ax.set_xticks([]), ax.set_yticks([]), ax.set_xticks([], minor=True), ax.set_yticks([], minor=True)

                    plt.setp(ax.spines.values(), linewidth=0.2, color='grey')
                    
                    #wind rose plot
                    if x_axis == 'ws':
                        ax.set_xlim(-0.0, None)
                        axins = inset_axes(ax, width="100%", height="100%", axes_class=WindroseAxes, bbox_to_anchor=(-0.1, 0.28, 0.6, 0.6 ), bbox_transform=ax.transAxes)
                    else:
                        ax.set_xlim(-0.0, None)
                        axins = inset_axes(ax, width="100%", height="100%", axes_class=WindroseAxes, bbox_to_anchor=(-0.1, 0.25, 0.65, 0.65 ), bbox_transform=ax.transAxes)
                    # axins.bar(wd, ws, normed=True, opening=1.0, edgecolor='white', nsector=11, bins=7, cmap=sns.color_palette("magma", as_cmap=True))
                    axins.contourf(wd, ws, bins=np.arange(0, 15, 1), nsector=16, cmap=sns.color_palette("magma", as_cmap=True))
                    xlabels = ('E', '', 'N', '',  'W', '', 'S', '',)
                    axins.set_xticklabels(xlabels)
                    axins.grid(linewidth=0.5)
                    axins.tick_params(axis='y', which='major',labelsize=6)
                    axins.tick_params(axis='x', which='major', pad=-3, labelsize=9)
                    # axins.tick_params(labelleft=False, labelright=True, labelbottom=True)
                    plt.setp(axins.spines.values(), linewidth=0.1)



                if r == 0:
                    farm_legend.append(ax.plot(coords[wt, 0], coords[wt, 1], marker="2",markersize=turb_marker_size, markeredgewidth=2.5, c=color,  linestyle='', label='WT '+str(wt+1))[0])
                    ha = 'right' if coords[wt,0] < 0.5 else 'left'
                    padding = -0.1  if coords[wt,0] < 0.5  else 0.1
                    ax.text(coords[wt, 0]+padding, coords[wt, 1], '$\mathbf{WT ' + str(wt+1) + '}$', c=color, fontsize=10 if x_axis=='ws' else 8 , ha=ha, bbox=dict(boxstyle='square', fc='white', ec='none', alpha=0.4))
                    continue
                else:
                    r=r-1
                
            y1 = np.array([np.array(y[i])[:, r] for i in idxs])
            if y_pred is not None:
                y2 = np.array([np.array(y_pred[i])[:, r] for i in idxs])
            if y_pred2 is not None:
                y3 = np.array([np.array(y_pred2[i])[:, r] for i in idxs])
            if r == 2:
                y1 *= 100
                if y_pred is not None:
                    y2 *= 100
                if y_pred2 is not None:
                    y3 *= 100

            if r == 0: # the power is in kW
                y1 /= 1000
                if y_pred is not None:
                    y2 /= 1000
                if y_pred2 is not None:
                    y3 /= 1000
                    
            df1 = pd.DataFrame()
            df1.index = pd.date_range("2018-01-01", periods=len(y1[:, wt]), freq="1S")
            df1['ws'] = ws
            df1['wd'] = polar_wd
            df1['true'] = y1[:, wt]
            if y_pred is not None:
                df1['pred'] = y2[:, wt]
            if y_pred2 is not None:
                df1['pred2'] = y3[:, wt]

            if x_axis.replace(" ", "").casefold() == 'wd':
                top = 360
                x = 'wd'
                x_label = 'Wind direction [$^\circ$]'
            elif x_axis.replace(" ", "").casefold() == 'ws':
                top = 30
                x = 'ws'
                x_label = 'Wind speed [$m.s^{-1}$]'
            else:
                raise ValueError('Please define a valid x_axis (available types: : \'ws\', \'wd\')')

            self = df1
            bins = np.linspace(-1, top, 18)
            y1 = 'true'
            y2 = 'pred'
            y3 = 'pred2'
            x_values = [(a + b) / 2.0 for a, b in zip(bins, bins[1:])]

            # ax.scatter(df1[x], df1[y1], s=1, c=color, marker=marker, alpha=0.5)
            center_line = _back_drop(self, x, y1, bins=bins, percentile=50)
            if y_pred is not None:
                center_line2 = _back_drop(self, x, y2, bins=bins, percentile=50)
            if y_pred2 is not None:
                center_line3 = _back_drop(self, x, y3, bins=bins, percentile=50)
            
            # close the circle if using wind direction plot
            if x_axis == 'wd':
                x_values = np.radians(x_values)
                x_values = np.append(x_values, x_values[0])    
                center_line = np.append(center_line, center_line[0])
                if y_pred is not None:
                    center_line2 = np.append(center_line2, center_line2[0])
                if y_pred2 is not None:
                    center_line3 = np.append(center_line3, center_line3[0]) 
            
            if shaded_surfaces is not None:
                for surface in shaded_surfaces:
                    bottom = np.array(_back_drop(self, x, y1, bins, percentile=surface[0]))
                    top = np.array(_back_drop(self, x, y1, bins, percentile=surface[1]))
                    if x_axis == 'wd':
                        bottom = np.append(bottom, bottom[0])
                        top = np.append(top, top[0])
                    
                    h1.append(ax.fill_between(x_values, bottom, top, facecolor=color, edgecolor='none',  linestyle=':', alpha=0.3))
                    h2.append(ax.plot(x_values, center_line, color=color, linestyle='-', marker=marker,markersize=2)[0])

                    if y_pred is not None:                    

                        bottom2 = np.array(_back_drop(self, x, y2, bins, percentile=surface[0]))
                        top2 = np.array(_back_drop(self, x, y2, bins, percentile=surface[1]))
                        if x_axis == 'wd':
                            top2 = np.append(top2, top[0])
                            bottom2 = np.append(bottom2, bottom[0])
                        h3.append(ax.fill_between(x_values, bottom2, top2, facecolor=color2, edgecolor='none',linestyle=':', alpha=0.3))
                        h4.append(ax.plot(x_values, center_line2, color=color2, linestyle='-', marker=marker, markersize=2)[0])

                    if y_pred2 is not None:
                        bottom3 = np.array(_back_drop(self, x, y3, bins, percentile=surface[0]))
                        top3 = np.array(_back_drop(self, x, y3, bins, percentile=surface[1]))
                        if x_axis == 'wd':
                            top3 = np.append(top3, top[0])
                            bottom3 = np.append(bottom3, bottom[0])
                        h5.append(ax.fill_between(x_values, bottom3, top3, facecolor=color3, edgecolor='none',linestyle=':', alpha=0.3))
                        h6.append(ax.plot(x_values, center_line3, color=color3, linestyle='-', marker=marker, markersize=2)[0])
                    
            else:
                h1.append(ax.plot(x_values, center_line, color=color, linestyle='-', marker=marker, markersize=2)[0])
                if y_pred is not None:
                    h2.append(ax.plot(x_values, center_line2, color=color2, linestyle='-', marker=marker, markersize=2)[0])
                if y_pred2 is not None:
                    h3.append(ax.plot(x_values, center_line3, color=color3, linestyle='-', marker=marker, markersize=2)[0])
                
            if x_axis == 'ws':
                ax.set_xlabel(x_label)
                ax.set_ylabel(string_list[r])
            else:
                ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
                xlabels = ('90°', '45°', '0°', '315°', '270°', '225°', '180°', '135°',)
                yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 4)
                # # round the ticks to the closest integer that is a multiple of 5
                if r==1 or r==2:
                    rounded_yticks = np.round(yticks)
                else:
                    rounded_yticks = np.round(yticks/5)*5
                ax.set_yticks(rounded_yticks)
                # print(rounded_yticks)
                ax.set_rlabel_position(30)
                formatter = ticker.FormatStrFormatter('%0.0f')
                ax.yaxis.set_major_formatter(formatter)
                # set_ax_to_sci_format(ax.yaxis)
                ax.set_xticklabels(xlabels)
                ax.set_title(string_list[r], fontsize=9)
                ax.grid(alpha=0.8, linewidth=0.5)
                ax.tick_params(axis='x', pad=-1, labelsize=7)
                ax.tick_params(axis='y', pad=0, labelsize=6)
                ax.set_axisbelow(False)
                plt.setp(ax.spines.values(), linewidth=0.2)

    # activate the grid for the ws plots
    if x_axis == 'ws':
        for k, ax in enumerate(ax_list):
            ax.grid()
    
    
    legend_tuples = []
    legend_names = []
    for i, wt in enumerate(turb):
        j = int(i*len(h1)/len(turb))
        if shaded_surfaces is not None:
            legend_tuples += [(h1[j], h2[j])]
            legend_names += ['WT {} {}'.format(wt+1, labels[0])]
            if y_pred is not None:
                legend_tuples += [(h3[j], h4[j])]
                legend_names += ['WT {} {}'.format(wt+1, labels[1])]
            if y_pred2 is not None:
                legend_tuples += [(h5[j], h6[j])]
                legend_names += ['WT {} {}'.format(wt+1, labels[2])]
        else:
            legend_tuples += [(h1[j])]
            legend_names += ['WT {} {}'.format(wt+1, labels[0])]
            if y_pred is not None:
                legend_tuples += [(h2[j])]
                legend_names += ['WT {} {}'.format(wt+1, labels[1])]
            if y_pred2 is not None:
                legend_tuples += [(h3[j])]
                legend_names += ['WT {} {}'.format(wt+1, labels[2])] 
    if x_axis == 'ws':
        legend = fig.legend(legend_tuples, legend_names, loc='upper center', fontsize=10, ncol=1, bbox_to_anchor=(0.8, 0.96))
    else:
        legend = fig.legend(legend_tuples, legend_names, loc='upper center', fontsize=8, ncol=1, bbox_to_anchor=(0.75, 0.855))

    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('none')
    if x_axis == 'ws':
        plt.tight_layout()
    return fig, axs

def plot_graph(g: torch_geometric.data.data.Data, ax=None, highlight:list=None, **kwargs):
    '''
    Plots graph using networkx.

    ...

    Arguments:
    ----------
    g : torch_geometric.data.data.Data
        a torch_geometric graph
    **kwargs : **kwargs
        networkx.draw keyboard arguments

    Returns
    -------
    fig, ax : fig, ax
        The figure and its axis
    '''

    graph = g
    G = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    node_pos_dict = {}
    for i in range(graph.num_nodes):
        node_pos_dict[i] = graph.pos[i, :].tolist()
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    nx.draw(G, pos=node_pos_dict, ax=ax, **kwargs)
    if highlight:
        for h in highlight:
            ax.plot(node_pos_dict[h][0], node_pos_dict[h][1], 'o', color='red')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_edgecolor('black')
    return ax

def plot_predictions(y, y_pred, turbines:list=None, mask=None, bounds:list=None):
    fig, ax = plt.subplot_mosaic([['upper left', 'upper right'], ['mid left', 'mid right'],
                                  ['low mid left', 'low mid right'], ['lower left', 'lower right']], figsize=(8, 12))

    # upper left
    y1 = np.array([np.array(i)[:, 1] for i in y])
    y2 = np.array([np.array(i)[:, 1] for i in y_pred])

    if mask is None:
        m = np.full(np.shape(np.array(y1)[:, 0]), True)
    else:
        m = (np.array(mask) < bounds[1]) & (np.array(mask) > bounds[0])

    if turbines is None:
        for i in range(np.shape(y1)[1]):
            ax['upper left'].plot(y1[:, i][m], y2[:, i][m], '.',
                                  label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                         y2[:, i][m]), 2)))
    else:
        for i in turbines:
            ax['upper left'].plot(np.array(y1)[:, i][m], np.array(y2)[:, i][m], '.',
                                  label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                         y2[:, i][m]), 2)))
    ax['upper left'].plot([0, 5e1], [0, 5e1], color='red')
    ax['upper left'].legend(loc='best', fontsize=10)
    ax['upper left'].set_title('Wind speed [$m.s^{-1}$], $R^2$: ' + str(np.round(r2_score(y1[m], y2[m]), 2)))
    ax['upper left'].grid()
    ax['upper left'].set_xlim([0, 30])
    ax['upper left'].set_ylim([0, 30])

    # upper right

    y1 = np.array([np.array(i)[:, 2] for i in y])
    y2 = np.array([np.array(i)[:, 2] for i in y_pred])
    if turbines is None:
        for i in range(np.shape(y1)[1]):
            ax['upper right'].plot(y1[:, i][m] * 100, y2[:, i][m] * 100, '.',
                                   label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                          y2[:, i][m]), 2)))
    else:
        for i in turbines:
            ax['upper right'].plot(y1[:, i][m] * 100, y2[:, i][m] * 100, '.',
                                   label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                          y2[:, i][m]), 2)))
    ax['upper right'].plot([0, 5e1], [0, 5e1], color='red')
    ax['upper right'].legend(loc='best', fontsize=10)
    ax['upper right'].set_title('Turbulence Intensity [\%], $R^2$: ' + str(np.round(r2_score(y1[m], y2[m]), 2)))
    ax['upper right'].grid()
    ax['upper right'].set_xlim([10, 40])
    ax['upper right'].set_ylim([10, 40])

    # mid left
    y1 = np.array([np.array(i)[:, 0] for i in y])
    y2 = np.array([np.array(i)[:, 0] for i in y_pred])
    if turbines is None:
        for i in range(np.shape(y1)[1]):
            ax['mid left'].plot(y1[:, i][m], y2[:, i][m], '.',
                                label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                       y2[:, i][m]), 2)))
    else:
        for i in turbines:
            ax['mid left'].plot(y1[:, i][m], y2[:, i][m], '.',
                                label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                       y2[:, i][m]), 2)))
    ax['mid left'].plot([0, 5e6], [0, 5e6], color='red')
    ax['mid left'].legend(loc='right', fontsize=10)
    ax['mid left'].set_title('Power [$W$], $R^2$: ' + str(np.round(r2_score(y1[m], y2[m]), 2)))
    ax['mid left'].grid()
    ax['mid left'].set_xlim([0, 5e6])
    ax['mid left'].set_ylim([0, 5e6])

    # mid right
    y1 = np.array([np.array(i)[:, 7] for i in y])
    y2 = np.array([np.array(i)[:, 7] for i in y_pred])
    if turbines is None:
        for i in range(np.shape(y1)[1]):
            ax['mid right'].plot(y1[:, i][m], y2[:, i][m], '.',
                                 label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                        y2[:, i][m]), 2)))
    else:
        for i in turbines:
            ax['mid right'].plot(y1[:, i][m], y2[:, i][m], '.',
                                 label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                        y2[:, i][m]), 2)))
    ax['mid right'].plot([0, 5e6], [0, 5e6], color='red')
    ax['mid right'].legend(loc='best', fontsize=10)
    ax['mid right'].set_title('Torsional DEL [$N.m$], $R^2$: ' + str(np.round(r2_score(y1[m], y2[m]), 2)))
    ax['mid right'].grid()
    ax['mid right'].set_xlim([0, .5e4])
    ax['mid right'].set_ylim([0, .5e4])

    # low mid left
    y1 = np.array([np.array(i)[:, 4] for i in y])
    y2 = np.array([np.array(i)[:, 4] for i in y_pred])
    if turbines is None:
        for i in range(np.shape(y1)[1]):
            ax['low mid left'].plot(y1[:, i][m], y2[:, i][m], '.',
                                    label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                           y2[:, i][m]), 2)))
    else:
        for i in turbines:
            ax['low mid left'].plot(y1[:, i][m], y2[:, i][m], '.',
                                    label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                           y2[:, i][m]), 2)))
    ax['low mid left'].plot([0, 5e6], [0, 5e6], color='red')
    ax['low mid left'].legend(loc='best', fontsize=10)
    ax['low mid left'].set_title('Edgewise DEL [$N.m$], $R^2$: ' + str(np.round(r2_score(y1[m], y2[m]), 2)))
    ax['low mid left'].grid()
    ax['low mid left'].set_xlim([0, .7e4])
    ax['low mid left'].set_ylim([0, .7e4])

    # low mid right
    y1 = np.array([np.array(i)[:, 3] for i in y])
    y2 = np.array([np.array(i)[:, 3] for i in y_pred])
    if turbines is None:
        for i in range(np.shape(y1)[1]):
            ax['low mid right'].plot(y1[:, i][m], y2[:, i][m], '.',
                                     label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                            y2[:, i][m]), 2)))
    else:
        for i in turbines:
            ax['low mid right'].plot(y1[:, i][m], y2[:, i][m], '.',
                                     label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                            y2[:, i][m]), 2)))
    ax['low mid right'].plot([0, 5e6], [0, 5e6], color='red')
    ax['low mid right'].legend(loc='best', fontsize=10)
    ax['low mid right'].set_title('Flapwise DEL [$N.m$], $R^2$: ' + str(np.round(r2_score(y1[m], y2[m]), 2)))
    ax['low mid right'].grid()
    ax['low mid right'].set_xlim([0, .7e4])
    ax['low mid right'].set_ylim([0, .7e4])

    # lower left
    y1 = np.array([np.array(i)[:, 5] for i in y])
    y2 = np.array([np.array(i)[:, 5] for i in y_pred])
    if turbines is None:
        for i in range(np.shape(y1)[1]):
            ax['lower left'].plot(y1[:, i][m], y2[:, i][m], '.',
                                  label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                         y2[:, i][m]), 2)))
    else:
        for i in turbines:
            ax['lower left'].plot(y1[:, i][m], y2[:, i][m], '.',
                                  label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                         y2[:, i][m]), 2)))
    ax['lower left'].plot([0, 5e6], [0, 5e6], color='red')
    ax['lower left'].legend(loc='best', fontsize=10)
    ax['lower left'].set_title('Fore-aft DEL [$N.m$], $R^2$: ' + str(np.round(r2_score(y1[m], y2[m]), 2)))
    ax['lower left'].grid()
    ax['lower left'].set_xlim([0, .3e5])
    ax['lower left'].set_ylim([0, .3e5])

    # lower right
    y1 = np.array([np.array(i)[:, 5] for i in y])
    y2 = np.array([np.array(i)[:, 5] for i in y_pred])
    if turbines is None:
        for i in range(np.shape(y1)[1]):
            ax['lower right'].plot(y1[:, i][m], y2[:, i][m], '.',
                                   label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                          y2[:, i][m]), 2)))
    else:
        for i in turbines:
            ax['lower right'].plot(y1[:, i][m], y2[:, i][m], '.',
                                   label='WT' + str(i) + ', R2: ' + str(np.round(r2_score(y1[:, i][m],
                                                                                          y2[:, i][m]), 2)))
    ax['lower right'].plot([0, 5e6], [0, 5e6], color='red')
    ax['lower right'].legend(loc='best', fontsize=10)
    ax['lower right'].set_title('Side-to-side DEL [$N.m$], $R^2$: ' + str(np.round(r2_score(y1[m], y2[m]), 2)))
    ax['lower right'].grid()
    ax['lower right'].set_xlim([0, .2e5])
    ax['lower right'].set_ylim([0, .2e5])

    # labels
    ax['lower right'].set_xlabel('Ground truth', fontsize=20)
    ax['lower left'].set_xlabel('Ground truth', fontsize=20)
    ax['upper left'].set_ylabel('Prediction', fontsize=20)
    ax['mid left'].set_ylabel('Prediction', fontsize=20)
    ax['low mid left'].set_ylabel('Prediction', fontsize=20)
    ax['lower left'].set_ylabel('Prediction', fontsize=20)

    if bounds is not None:
        plt.suptitle(f'Wind direction: ${bounds[0]}-{bounds[1]}^{{\circ}}$')
    else:
        plt.suptitle('')

    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_farm_qty(data, var_idx_to_plot=0, sensor_node_idx=None, show_max =False, cmap_label=None, figsize=(10, 8),
                  windrose=True,title=None,label=False, cmap='viridis', cmap_range=None, ax=None, show_cmap=False, show_graph=False):
    y = data.y[:, var_idx_to_plot].numpy()
    coords = data.pos
    # normalize the coords
    coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
    coords[:, 1] = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())
    
    wf_globals = data.globals
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
    # cmap = sns.color_palette(cmap, as_cmap=True)
    if show_graph:
        node_pos_dict = {}
        for i in range(data.num_nodes):
            node_pos_dict[i] = data.pos[i, :].tolist()
        G = torch_geometric.utils.to_networkx(data, to_undirected=True)
        nx.draw_networkx_edges(G, node_pos_dict, ax=ax, width=2, alpha=0.5)
    
    h1 = ax.scatter(coords[:,0], coords[:,1], c=y, s=150, marker='o', linewidth=1.5, edgecolors='black', cmap=cmap)
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1./20)
    if show_cmap:
        # pad = axes_size.Fraction(0.6, width)
        # cax = divider.append_axes("left", size=width, pad=pad)
        
        # uncomment to get figure 13 of the paper
        fig = ax.get_figure()
        cax = fig.add_axes([0.09, 0.56, 0.009, 0.32]) # [left, bottom, width, height]
        plt.colorbar(h1, cax=cax)
        cax.set_ylabel(cmap_label, fontsize=15)
        cax.yaxis.set_ticks_position('left')
        cax.yaxis.set_label_position('left')
        
        # clb.ax.set_title(cmap_label, fontsize=15)
        if cmap_range is not None:
            h1.set_clim(cmap_range[0], cmap_range[1])
           
    ax.set_title(title, fontsize=20)
    plt.tick_params(left=False, right=False, labelleft=True, labelbottom=False, bottom=False)
    
    if label:
        h_offset = 0.022
        v_offset = 0.022
        for i, txt in enumerate(range(len(coords))):
            ax.text(coords[i, 0]+h_offset, coords[i, 1]+v_offset, str(txt), fontsize=8, ha='left', va='bottom')
    
    if (wf_globals is not None) and (windrose is True):
        ws = wf_globals[0][0].repeat(6).numpy()
        wd = (wf_globals[0][1].repeat(6).numpy())
        
        # wind rose plot
        axins = inset_axes(ax, width="50%", height="50%", bbox_to_anchor=(0.58, 0.05, 0.4, 0.4), bbox_transform=ax.transAxes,
                   loc='lower right', axes_class=WindroseAxes)
        axins.bar(wd, ws, normed=True, opening=0.8, edgecolor='white', nsector=36, color='#0bb5a7')
        xlabels = ('E', '', 'N', '', 'W', '', 'S', '',)
        axins.set_xticklabels(xlabels)
        axins.grid(linewidth=0.5)
        axins.tick_params(axis='y', which='major')
        axins.set_yticklabels([])
        axins.tick_params(axis='x', which='major', pad=-4, labelsize=9)
        for spine in axins.spines.values():
            spine.set_linewidth(0.1)
        
    plt.tight_layout()
    ax.margins(x=0.05, y=0.1)
    ax.set_aspect('equal')
    
    for spine in ax.spines.values():
            spine.set_linewidth(0.0)
            
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    if show_max:
        ax.scatter(coords[np.argmax(y),0], coords[np.argmax(y),1], facecolors='none', s=100, linewidth=3, edgecolors='blue')
    
    if sensor_node_idx is not None:
        snode_plot = ax.scatter(coords[sensor_node_idx,0], coords[sensor_node_idx,1], facecolors='none', s=120, linewidth=3, edgecolors='red')


def plot_farm(y: list, y_pred: list, dataset: torch_geometric.data.Dataset, idx: int = 0, wf_globals:list = None,
              variable: str = 'ws', typ: str = 'pred', annotate: bool = True, title: str = None, **kwargs):
    df = get_mean_values(y, y_pred, dataset, idx)

    assert {typ}.issubset({'pred', 'true', 'diff'})

    mapping = {'power': 'Power [$kW$]', 'ws': 'Wind speed [$m.s^{-1}$]', 'ti': 'Turbulence Intensity [\%]',
               'del_flap': 'Flapwise DEL [$N.m$]', 'del_edge': 'Edgewise DEL [$N.m$]', 'del_fa': 'FA DEL [$N.m$]',
               'del_ss': 'SS DEL [$N.m$]', 'del_torsion': 'Torsional DEL [$N.m$]'}

    assert {variable}.issubset(mapping.keys())

    fig, ax = plt.subplots(figsize=(10, 8))

    if typ == 'diff':
        plt.scatter(df.x, df.y, c=df['_'.join([variable, 'true'])] - df['_'.join([variable, 'pred'])],
                    s=100, linewidth=2, edgecolors='black', **kwargs)
    else:
        plt.scatter(df.x, df.y, c=df['_'.join([variable, typ])], s=100, linewidth=2, edgecolors='black', **kwargs)
    if annotate:
        for i, v1 in enumerate(df.x):
            plt.annotate(i, (df.x[i] + 0.01 * df.x.max(), df.y[i] + .01 * df.x.max()))
    clb = plt.colorbar()
    clb.ax.set_title(mapping[variable], fontsize=15)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    # plt.xlim(df.x.min() - .1 * df.x.max(), df.x.max() + .1 * df.x.max())
    # plt.ylim(df.y.min() - .1 * df.y.max(), df.y.max() + .1 * df.y.max())
    
    if wf_globals is not None:
        ws = wf_globals[idx][0][0].repeat(6).numpy()
        wd = wf_globals[idx][0][1].repeat(6).numpy()
        
        # wind rose plot
        axins = inset_axes(ax, width="50%", height="50%", bbox_to_anchor=(0.8, 0.68, 0.3, 0.3), bbox_transform=ax.transAxes,
                   loc='upper left', axes_class=WindroseAxes)
        axins.bar(wd, ws, normed=True, opening=0.8, edgecolor='white', nsector=36)
        xlabels = ('E', '', 'N', '', 'W', '', 'S', '',)
        axins.set_xticklabels(xlabels)
        axins.grid(linewidth=0.5)
        axins.tick_params(axis='y', which='major', labelsize=7)
        axins.tick_params(axis='x', which='major', pad=-4, labelsize=9)
        plt.setp(axins.spines.values(), linewidth=0.1)
    
    
    plt.title(title, fontsize=20)
    return fig, ax


def plot_vs_x_model_comp(dataset: torch_geometric.data.Dataset, y:list, y_pred_dict: dict, x_axis: str, turb, layout_idx: int, length: int,
              dpi=200, show_farm=False, vars_idx_to_plot=[1,2,3,4]):
    """
    Plot the results of a machine learning model on a test dataset.

    Parameters
    ----------
    y_dict : dict
        A dict of results of 3 models to compare
    x_axis : str
        A string specifying the type of x-axis to use in the plot. The options are 'ws' (wind speed) or 'wd' (wind direction).
    turb : int
        An integer representing the index of the turbine for which the results are being plotted.
    length : int
        The length of the test dataset.

    Returns
    -------
    fig, ax
        This function creates a subplot with 4 rows and 2 columns and plots the results of the model for each output variable.

    """

    matplotlib.rcParams['axes.linewidth'] = 1.5

    ws = []
    wd = []
    start_idx = layout_idx * length
    end_idx = layout_idx * length + length
    for d in range(start_idx, end_idx):
        ws += [((dataset.__getitem__(d).globals[0][0]).item())]
        wd += [((dataset.__getitem__(d).globals[0][1]).item())]
    if show_farm:
        fig, axs =  plt.subplot_mosaic('AAA;BCD;EFG;HIJ;KLM', figsize=(9, 11), dpi=dpi, height_ratios=[1, 0.8, 0.8, 0.8, 0.8])
        # fig, axs = plt.subplot_mosaic('AAA;BFJ;CGK;DHL;EIM', figsize=(9, 11), dpi=dpi)
        coords = dataset.__getitem__(start_idx).pos
        graph = dataset.__getitem__(start_idx)
        ax_list = axs.values()
    else:
        fig, axs = plt.subplots(4, 2, figsize=(8, 12), dpi=dpi)
        ax_list = axs.flatten()

    string_list = ['Power [$MW$]', 'Wind speed [$m.s^{-1}$]', 'Turbulence intensity [$\%$]',
                   'DEL flap [$N.m$]', 'DEL edge [$N.m$]', 'DEL FA [$N.m$]', 'DEL SS [$N.m$]',
                   'DEL torsion [$N.m$]']

    if not isinstance(turb, list):
        turb = [turb]
    assert len(turb) < 4

    color_picker_pred = ['tab:blue', 'darkorange', 'limegreen']
    color_picker_true = ['teal', 'lightcoral', 'forestgreen']
    wt_markers = ['x', '.', '+']
    # color_picker_true = ['#40798C', '#70A9A1', '#9EC1A3', '#CFE0C3', '#1F363D']
    # color_picker_pred = ['#DB7C26', '#D8572A', '#F7B538', '#C32F27', '#780116']

    h1, h2, h3, h4 = [], [], [], []
    farm_legend = []

    for i, wt in enumerate(turb):
        color = color_picker_true[i]
        color2 = color_picker_pred[i]
        marker = wt_markers[i]
        for r, ax in enumerate(ax_list):
            if show_farm:
                if i == 0 and r == 0:
                    ax.scatter(coords[:, 0], coords[:, 1], s=150, c='black', marker="2", linewidth=1, alpha=0.7)
                    node_pos_dict = {}
                    for i in range(graph.num_nodes):
                        node_pos_dict[i] = graph.pos[i, :].tolist()
                    G = torch_geometric.utils.to_networkx(graph, to_undirected=True)
                    nx.draw_networkx_edges(G, node_pos_dict, ax=ax, width=0.5, alpha=0.30)

                    ax.set_aspect('equal', 'datalim')
                    ax.set_xlim(-2.15 * coords[:, 0].max(), 0)
                    ax.set_xlabel('\n')
                    ax.margins(x=0, y=0.05)
                    ax.set_xticks([]), ax.set_yticks([]), ax.set_xticks([], minor=True), ax.set_yticks([], minor=True)

                    # wind rose plot
                    axins = inset_axes(ax, width="45%", height="70%", loc=6, axes_class=WindroseAxes)
                    axins.bar(wd, ws, normed=True, opening=0.8, edgecolor='white', nsector=11)
                    xlabels = ('E', '', 'N', '', 'W', '', 'S', '',)
                    axins.set_xticklabels(xlabels)
                    axins.grid(linewidth=0.5)
                    axins.tick_params(axis='y', which='major', labelsize=7)
                    axins.tick_params(axis='x', which='major', pad=-3, labelsize=9)
                    plt.setp(axins.spines.values(), linewidth=0.1)

                if r == 0:
                    farm_legend.append(
                        ax.plot(coords[wt, 0], coords[wt, 1], marker="2", markersize=16, markeredgewidth=2.5, c=color,
                                linestyle='', label='WT ' + str(wt + 1))[0])
                    continue
                else:
                    r = r - 1

            dict_idx = r%3
            model = list(y_pred_dict.keys())[dict_idx]
            y_pred = y_pred_dict[model]
            vars_idx = vars_idx_to_plot[r//3]

            y1 = np.array([np.array(y[i])[:, vars_idx] for i in range(start_idx, end_idx)])
            if y_pred is not None:
                y2 = np.array([np.array(y_pred[i])[:, vars_idx] for i in range(start_idx, end_idx)])
            if vars_idx == 2:
                y1 *= 100
                if y_pred is not None:
                    y2 *= 100
            if vars_idx == 0:
                y1 /= 1000000
                if y_pred is not None:
                    y2 /= 1000000

            df1 = pd.DataFrame()
            df1.index = pd.date_range("2018-01-01", periods=len(y1[:, wt]), freq="1S")
            df1['ws'] = ws
            df1['wd'] = wd
            df1['true'] = y1[:, wt]
            if y_pred is not None:
                df1['pred'] = y2[:, wt]

            if x_axis.replace(" ", "").casefold() == 'wd':
                top = 360
                x = 'wd'
                x_label = 'Wind direction [$^\circ$]'
            elif x_axis.replace(" ", "").casefold() == 'ws':
                top = 30
                x = 'ws'
                x_label = 'Wind speed [$m.s^{-1}$]'
            else:
                raise ValueError('Please define a valid x_axis (available types: : \'ws\', \'wd\')')

            self = df1
            bins = np.linspace(-1, top, 15)
            y1 = 'true'
            y2 = 'pred'
            x_values = [(a + b) / 2.0 for a, b in zip(bins, bins[1:])]
            center_line = _back_drop(self, x, y1, bins=bins, percentile=50)
            if y_pred is not None:
                center_line2 = _back_drop(self, x, y2, bins=bins, percentile=50)
            shaded_surfaces = [(25, 75), (10, 90)]
            for surface in shaded_surfaces:
                bottom = np.array(_back_drop(self, x, y1, bins, percentile=surface[0]))
                top = np.array(_back_drop(self, x, y1, bins, percentile=surface[1]))
                h1.append(
                    ax.fill_between(x_values, bottom, top, facecolor=color, edgecolor=color, linestyle=':', alpha=0.3))
                h2.append(ax.plot(x_values, center_line, color=color, linestyle='-', marker=marker)[0])

                if y_pred is not None:
                    bottom2 = np.array(_back_drop(self, x, y2, bins, percentile=surface[0]))
                    top2 = np.array(_back_drop(self, x, y2, bins, percentile=surface[1]))
                    h3.append(
                        ax.fill_between(x_values, bottom2, top2, facecolor=color2, edgecolor=color2, linestyle=':',
                                        alpha=0.3))
                    h4.append(
                        ax.plot(x_values, center_line2, color=color2, linestyle='-', marker=marker, markersize=7)[0])

            if y_pred is not None and len(turb) == 1:
                ax.legend([(h1[0], h2[0]), (h3[0], h4[0])], ['True', 'Pred'], loc='best', fontsize=10)
            if r//3==3:
                ax.set_xlabel(x_label)
            else:
                ax.tick_params(axis='x', which='both',labelbottom=False)
            if r%3==0:
                ax.set_ylabel(string_list[vars_idx])
            else:
                ax.tick_params(axis='y', which='both', labelleft=False)
            if r//3==0:
                ax.set_title(model)

    for k, ax in enumerate(ax_list):
        if len(ax_list) == 9 and k == 0:
            ax.legend(farm_legend, ['WT {}'.format(wt) for wt in turb])
            continue
        ax.grid()

    if len(turb) == 1:
        plt.suptitle(f'WT {wt}')
    else:
        legend_tuples = []
        legend_names = []
        for i, wt in enumerate(turb):
            j = int(i * len(h1) / len(turb))
            legend_tuples += [(h1[j], h2[j]), (h3[j], h4[j])]
            legend_names += ['WT {} True'.format(wt), 'WT {} Pred'.format(wt)]
        legend = fig.legend(legend_tuples, legend_names, loc='upper center', fontsize=11, ncol=len(turb),
                            bbox_to_anchor=(0.55, 0.86))
        # legend.get_frame().set_facecolor('none')
        # legend.get_frame().set_edgecolor('none')
    plt.tight_layout(h_pad=1)
    return fig, axs