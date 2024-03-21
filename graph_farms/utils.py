import numpy as np
import torch
from torch_geometric.transforms import FaceToEdge, Polar, KNNGraph, RadiusGraph, Delaunay, Cartesian, LocalCartesian
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
import math
import pandas as pd

def rotate(origin, point, angle):
    """ Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def area(x1, y1, x2, y2, x3, y3):
    """ Calculate the area of a triangle given its vertices.""" 
    a = abs((x1 * (y2 - y3) + (x2 * (y3 - y1)) + (x3 * (y1 - y2))) / 2.0)
    return a


def is_inside(x1, y1, x2, y2, x3, y3, x, y):
    """ Check if a point is inside a triangle. """
    # calculate area of triangle ABC
    A = area(x1, y1, x2, y2, x3, y3)

    # calculate area of triangle PBC
    A1 = area(x, y, x2, y2, x3, y3)

    # calculate area of triangle PAC
    A2 = area(x1, y1, x, y, x3, y3)

    # calculate area of triangle PAB
    A3 = area(x1, y1, x2, y2, x, y)

    # check if sum of A1, A2 and A3 is same as A
    if math.isclose(A, A1 + A2 + A3, abs_tol=1e-1):
        return True
    else:
        return False

def to_graph(points: np.array, connectivity: str, min_dist=None, constant=30, add_edge='polar'):
    '''
    Converts np.array to torch_geometric.data.data.Data

    ...

    Arguments:
    ----------
    points : np.array
        a [num_turbines, 2] array containing the coordinates of the turbines.
    connectivity : str
        string with the accepted values \'delaunay\', \'knn\' and \'radial\'.
    min_dist : (float, optional)
        float number stating the minimal distance between turbines (default None). Required for connectivity = \'radial\'.
    constant : (float, optional)
        float constant to be multiplied by the minimal distance defining a radius (default 30). Required for connectivity = \'radial\'.
    add_edge : (str, optional)
        str specifing if \'polar\', \'cartesian\' or \'local cartesian\' coordinates of the nodes should be added to the edges (default \'polar\').

    Returns
    -------
    torch_geometric.data.data.Data
        a torch_geometric.data.data.Data graph.

    Raises
    ------
    ValueError
        If connectivity string isn't \'delaunay\', \'knn\' or \'radial\'.
        If minimal distance isn't defined for radial connectivity.
    '''

    assert (points.shape[1] == 2)
    t = torch.Tensor(points)
    x = Data(pos=t)
    if connectivity.casefold() == 'delaunay':
        d = Delaunay()
        e = FaceToEdge()
        g = e(d(x))
    elif connectivity.casefold() == 'knn':
        kv = math.ceil(np.sqrt(len(points)))
        knn = KNNGraph(k=kv)
        g = knn(x)
    elif connectivity.casefold() == 'radial':
        if (min_dist is not None):
            radius = min_dist * constant
            r = RadiusGraph(r=radius)
            g = r(x)
        else:
            raise ValueError('Minimal distance between turbines is required.')
    elif connectivity.casefold() == 'fully_connected':
        adj = torch.ones(t.shape[0], t.shape[0])
        g = Data(pos=t, edge_index=dense_to_sparse(adj.fill_diagonal_(0))[0])
    else:
        raise ValueError('Please define the connectivity scheme (available types: : \'delaunay\', \'knn\', \'radial\', , \'fully_connected\')')

    if add_edge == 'polar'.casefold():
        p = Polar(norm=False)
        p(g)
    elif add_edge == 'cartesian'.casefold():
        c = Cartesian(norm=False)
        c(g)
    elif add_edge == 'local cartesian'.casefold():
        lc = LocalCartesian(norm=False)
        lc(g)
    else:
        raise ValueError(
            'Please select a coordinate system that is supported (available types: : \'polar\', \'cartesian\' or \'local cartesian\')')
    return g

def get_mean_values(y:list, y_pred:list, dataset:Dataset, idx:int=0):
    string_list = ['power','ws','ti','del_flap','del_edge','del_fa','del_ss','del_torsion']

    positions = dataset.__getitem__(idx).pos

    df = pd.DataFrame()
    for wt, v in enumerate(positions):
        df_int = pd.DataFrame({'WT':int(wt),'x':[int(v[0])],'y':[int(v[1])]})
        for r, s in enumerate(string_list):
            y1 = np.array([np.array(i)[:, r] for i in y])
            y2 = np.array([np.array(i)[:, r] for i in y_pred])

            if r == 2:
                y1 = np.array([np.array(i)[:, r]*100 for i in y])
                y2 = np.array([np.array(i)[:, r]*100 for i in y_pred])

            df1 = pd.DataFrame()
            df1.index = pd.date_range("2018-01-01", periods=len(y1[:,wt]), freq="1S")
            df1['true'] = y1[:, wt]
            df1['pred'] = y2[:, wt]
            df_mid = pd.DataFrame({'_'.join([s,'true']):[df1.true.values.mean()],
                                 '_'.join([s,'pred']):[df1.pred.values.mean()]})
            df_int = pd.concat([df_int,df_mid],axis=1)
        df = pd.concat([df,df_int],axis=0)
    df.set_index('WT',inplace=True)
    return df