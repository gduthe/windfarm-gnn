import numpy as np
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
import pandas as pd
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

def back_drop(df, x_prm, y_prm, bins, percentile=10):
    """ Function to obtain from a dataframe the binned results
    
        args:
        df: pd.DataFrame, the dataframe with the data
        x_prm: str, the parameter to bin
        y_prm: str, the parameter to compute the percentile
        bins: list, the bins to use for binning
        percentile: int, the percentile to compute
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

def plot_vs_x(dataset:Dataset, y:list, y_pred, x_axis:str, turb:list, layout_idx:int, num_inflows_per_layout:int, dpi=600, y_pred2=None,  
              ws_filter=None, wd_filter=None, labels=None, shaded_surfaces=[(25, 75), (10, 90)], show_farm=False, plot_graph=True):
    """ Plot the results of ground truth vs predicted values for a given turbine of a given layout from a dataset.

        args:
        dataset: torch_geometric.data.Dataset, the dataset
        y: list, the ground truth values
        y_pred: list, the predicted values
        x_axis: str, the x axis to use for the plot either 'ws' or 'wd', wind direction results in polar plots
        turb: list, the list of turbines to plot for
        layout_idx: int, the layout index to plot (index of the layout in the dataset)
        num_inflows_per_layout: int, number of inflows per layout for this dataset
        dpi: int, the dpi of the plot
        y_pred2: list, the second set of predicted values if available
        ws_filter: tuple, the bounds of the wind speed to filter if needed
        wd_filter: tuple, the bounds of the wind direction to filter if needed
        labels: list, the labels for the plot
        shaded_surfaces: list, the list of tuples with the percentiles to shade
        show_farm: bool, whether to show the farm layout above the plots
        plot_graph: bool, whether to plot the graph of the farm layout
    """
    matplotlib.rcParams['axes.linewidth'] = 1.5
    assert x_axis in ['ws', 'wd']
    if labels is None:
        labels = ['True', 'Pred']
    ws = []
    wd = []
    start_idx = layout_idx*num_inflows_per_layout
    end_idx = layout_idx*num_inflows_per_layout + num_inflows_per_layout
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
        if x_axis == 'wd':
            fig, axs = plt.subplots(4, 2, figsize=(8, 12), dpi=dpi, subplot_kw={'projection': 'polar'})
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
                        G = to_networkx(graph, to_undirected=True)
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
            center_line = back_drop(self, x, y1, bins=bins, percentile=50)
            if y_pred is not None:
                center_line2 = back_drop(self, x, y2, bins=bins, percentile=50)
            if y_pred2 is not None:
                center_line3 = back_drop(self, x, y3, bins=bins, percentile=50)
            
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
                    bottom = np.array(back_drop(self, x, y1, bins, percentile=surface[0]))
                    top = np.array(back_drop(self, x, y1, bins, percentile=surface[1]))
                    if x_axis == 'wd':
                        bottom = np.append(bottom, bottom[0])
                        top = np.append(top, top[0])
                    
                    h1.append(ax.fill_between(x_values, bottom, top, facecolor=color, edgecolor='none',  linestyle=':', alpha=0.3))
                    h2.append(ax.plot(x_values, center_line, color=color, linestyle='-', marker=marker,markersize=2)[0])

                    if y_pred is not None:                    

                        bottom2 = np.array(back_drop(self, x, y2, bins, percentile=surface[0]))
                        top2 = np.array(back_drop(self, x, y2, bins, percentile=surface[1]))
                        if x_axis == 'wd':
                            top2 = np.append(top2, top[0])
                            bottom2 = np.append(bottom2, bottom[0])
                        h3.append(ax.fill_between(x_values, bottom2, top2, facecolor=color2, edgecolor='none',linestyle=':', alpha=0.3))
                        h4.append(ax.plot(x_values, center_line2, color=color2, linestyle='-', marker=marker, markersize=2)[0])

                    if y_pred2 is not None:
                        bottom3 = np.array(back_drop(self, x, y3, bins, percentile=surface[0]))
                        top3 = np.array(back_drop(self, x, y3, bins, percentile=surface[1]))
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

def plot_graph(g: Data, ax=None, highlight:list=None, nx_draw_kwargs:dict=None):
    """ Plots a graph using networkx.
        
        args:
        g: torch_geometric.data.Data, the graph to plot
        ax: matplotlib.axes.Axes, the axis to plot the graph on
        highlight: list, the nodes to highlight
        nx_draw_kwargs: dict, the keyword arguments to pass to the nx.draw function
    """
    
    graph = g
    G = to_networkx(graph, to_undirected=True)
    node_pos_dict = {}
    for i in range(graph.num_nodes):
        node_pos_dict[i] = graph.pos[i, :].tolist()
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    nx.draw(G, pos=node_pos_dict, ax=ax, **nx_draw_kwargs)
    if highlight:
        for h in highlight:
            ax.plot(node_pos_dict[h][0], node_pos_dict[h][1], 'o', color='red')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_edgecolor('black')
    return ax

def plot_farm_qty(data, var_idx_to_plot=0, highlight_turb_idx=None, show_max =False, ax=None, figsize=(10, 8),windrose=True,
                  title=None, label=False, cmap_label=None, cmap='viridis', cmap_range=None, show_cmap=False, show_graph=False):
    """ Plot a quantity on the farm layout.
    
        args:
        data: torch_geometric.data.Data, the data object
        var_idx_to_plot: int, the index of the variable to plot
        highlight_turb_idx: int, the index of a turbine to highlight if needed
        show_max: bool, whether to highlight the turbine with the maximum value
        ax: matplotlib.axes.Axes, the axis to plot the graph on
        figsize: tuple, the size of the figure
        windrose: bool, whether to plot the windrose in a corner of the plot
        title: str, the title of the plot
        label: bool, whether to label the turbines with their index
        cmap_label: str, the label for the colorbar
        cmap: str, the colormap to use
        cmap_range: tuple, the range of the colormap if needed
        show_cmap: bool, whether to show the colorbar
        show_graph: bool, whether to show the graph of the farm layout
    """
    
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
        G = to_networkx(data, to_undirected=True)
        nx.draw_networkx_edges(G, node_pos_dict, ax=ax, width=2, alpha=0.5)
    
    h1 = ax.scatter(coords[:,0], coords[:,1], c=y, s=150, marker='o', linewidth=1.5, edgecolors='black', cmap=cmap)
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1./20)
    if show_cmap:
        pad = axes_size.Fraction(0.6, width)
        cax = divider.append_axes("left", size=width, pad=pad)
        
        # uncomment to get figure 13 of the paper
        # fig = ax.get_figure()
        # cax = fig.add_axes([0.09, 0.56, 0.009, 0.32]) # [left, bottom, width, height]
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
    
    if highlight_turb_idx is not None:
        h_plot = ax.scatter(coords[highlight_turb_idx,0], coords[highlight_turb_idx,1], facecolors='none', s=120, linewidth=3, edgecolors='red')