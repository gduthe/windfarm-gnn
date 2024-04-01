from .layout_gen import LayoutGenerator
from .inflow_gen import InflowGenerator
from .pywake_sim import simulate_farm
from .utils import to_graph
import yaml
import numpy as np
import argparse
from box import Box
import os
import pandas as pd
import torch
from joblib import Parallel, delayed
import tempfile
from zipfile import ZipFile

def process_one_layout(layout, inflow_df, layout_id, dset_path, node_scada_features=False, connectivity='delaunay', loads_model='TwoWT'):
    """ Function to process one layout and generate the graphs for the given inflow conditions.
        The function will save the graphs in a zip file in the given dataset path. Is used to parallelize the process.
        
        args:
        layout: dict, the layout to process
        inflow_df: pd.DataFrame, the inflow conditions for the layout
        layout_id: str, the id of the layout
        dset_path: str, the path to save the dataset
        node_scada_features: bool, if the graphs should have node features which 'replicate' SCADA (power, ws, ti)
        connectivity: str, the kind of connectivity to use: either delaunay, knn, radial, fully_connected, all ()
        loads_model: str, the kind of load model to use: either OneWT, TwoWT or both (for comparison)
    """
    assert (loads_model in ['OneWT', 'TwoWT', 'both'])
    assert (connectivity in ['delaunay', 'knn', 'radial', 'fully_connected', 'all'])

    # extract coords of the wind turbines from layout
    wt_coords = layout['coords']

    # extract yaw angles of the wind turbines from layout
    wt_yaws = layout['yaw_angles']

    # extract the operating modes of the wind turbines from layout
    wt_modes = layout['operating_modes']

    # create an info tuple
    info = (str(layout['form']), str(round(layout['min_dist'])), str(len(wt_coords)), str(layout_id))

    # choose load model
    if loads_model == 'both':
        loads_list = ['OneWT', 'TwoWT']
    else:
        loads_list = [loads_model]
        
    # choose connectivity
    if connectivity == 'all':
        connectivity_list = ['delaunay', 'knn', 'radial', 'fully_connected']
    else:
        connectivity_list = [connectivity]

    # create temp folder to store graphs
    with  tempfile.TemporaryDirectory() as tempdir:

        # loop over the loads model (used only if both is selected)
        for loads_method in loads_list:
            # simulate using pywake
            power, loads, wt_yaws, wt_modes = simulate_farm(inflow_df, wt_coords, wt_yaws, wt_modes, loads_method)

            # loop over the connectivity type saving each in a different folder
            for c in connectivity_list:

                # parse the raw points to a graph based on current connectivity
                g = to_graph(wt_coords, connectivity=c, min_dist=layout['min_dist'], add_edge='polar')

                # get the dataset path for this connectivity
                dataset_path = os.path.join(dset_path, loads_method, c)
                os.makedirs(dataset_path, exist_ok=True)

                # open a zip file and store the individual inflows for the layout as separate graphs
                zip_path = os.path.join(dataset_path, '_'.join(info)) + '.zip'
                with ZipFile(zip_path, 'w') as zf:
                    for i in range(0, len(torch.Tensor(power.ws.values))):

                        if node_scada_features:
                            g.x = torch.Tensor(np.array([power.Power.values[:, i], power.WS_eff.values[:, i],
                                                         power.TI_eff.values[:, i], wt_yaws[:, i], wt_modes[:, i]])).T
                            t = torch.Tensor()
                            for s in loads.sensor.values:
                                t = torch.cat((t, torch.Tensor(np.array([loads.sel(sensor=s).DEL.values[:, i]]))))
                        else:
                            g.x = torch.Tensor(np.array([wt_yaws[:, i], wt_modes[:, i]])).T
                            t = torch.Tensor(np.array([power.Power.values[:, i], power.WS_eff.values[:, i],
                                                       power.TI_eff.values[:, i]]))
                            for s in loads.sensor.values:
                                t = torch.cat((t, torch.Tensor(np.array([loads.sel(sensor=s).DEL.values[:, i]]))))

                        g.y = t.T
                        g.globals = torch.Tensor(np.array([power.ws.values[i], power.wd.values[i], power.TI.values[i]]))

                        # if there are no nan values then save graph
                        if not torch.isnan(g.y).any() and not torch.isnan(g.globals).any():
                            graph_file_name = '_'.join(['_'.join(info), 'ws', str(np.round(g.globals[0].item(), 2)),
                                               'wd', str(np.round(g.globals[1].item(), 2))]) + '.pt'

                            graph_temp_path = os.path.join(tempdir, graph_file_name)
                            torch.save(g, graph_temp_path)

                            zf.write(filename=graph_temp_path, arcname=graph_file_name)


def generate_graphs(config_path:str, num_layouts:int, num_inflows:int, dset_path:str, num_threads=1, node_scada_features=False,
                    connectivity='delaunay', loads_model='TwoWT', fixed_wd=None, fixed_ws=None, fixed_ti=None):
    """ Main function to generate a bunch of wind farm graphs in parallel for a number of layouts and inflow conditions.
        The function will save the graphs in zip files in the given dataset path with the following structure, which depends on selected
        loads model and connectivity:
        
        dset_path
            ├── OneWT
            │   ├── delaunay
            │   │   ├── form_min_dist_num_wt_layout_id.zip
            │   │   ├── ...
            │   ├── knn
            │   ├── radial
            │   ├── fully_connected
            ├── TwoWT
                ├── ...

        args:
        config_path: str, the path to the config file with parameters for pywake and inflow conditions
        num_layouts: int, the number of layouts to generate
        num_inflows: int, the number of inflows to generate per layout
        dset_path: str, the path to save the dataset
        num_threads: int, the number of threads to use
        node_scada_features: bool, if the graphs should have node features which 'replicate' SCADA (power, ws, ti)
        connectivity: str, the kind of connectivity to use: either delaunay, knn, radial, fully_connected, all
        loads_model: str, the kind of load model to use: either OneWT, TwoWT or both
        fixed_wd: float, if a fixed wind direction should be used
        fixed_ws: float, if a fixed wind speed should be used
        fixed_ti: float, if a fixed turbulence intensity should be used
    """
    # load the config
    config = Box.from_yaml(filename=config_path, Loader=yaml.FullLoader)

    # first generate the random layouts
    layout_generator = LayoutGenerator(**config.turbine_settings)
    layouts = layout_generator.generate_layouts(num_layouts)

    # then generate the random inflow conditions
    inflow_generator = InflowGenerator(**config)
    inflows = inflow_generator.generate_inflows(num_samples=num_inflows*num_layouts, plot=False)
    inflow_df = pd.DataFrame(data=inflows)
    
    if fixed_wd is not None:
        inflow_df['wd'] = fixed_wd
    if fixed_ws is not None:
        inflow_df['u'] = fixed_ws
    if fixed_ti is not None:
        inflow_df['ti'] = fixed_ti
    

    # loop through the layouts, processing them in parallel
    Parallel(n_jobs=num_threads)(delayed(process_one_layout)
        (layout, inflow_df.iloc[list(range(i*num_inflows, i*num_inflows + num_inflows))], str(i).zfill(len(str(num_layouts))),
         dset_path, node_scada_features, connectivity, loads_model) for i, layout in enumerate(layouts))
    
    print('Finished generating {} graphs ({} layouts x {} inflows)!'.format(num_layouts*num_inflows, num_layouts, num_inflows))


if __name__ == "__main__":
    # set the args of the script
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="path to the config file", type=str, required=False, default='config.yml')
    parser.add_argument('-num_layouts', '-nl', help="number of layouts to generate", type=int, required=False, default=100)
    parser.add_argument('-num_inflows', '-ni', help="number of inflows to generate per layout", type=int, required=False, default=10)
    parser.add_argument('-dset_path', '-d', help="path for the dataset to generate", type=str, required=False, default='./generated_graphs/train_set/')
    parser.add_argument('-node_scada_features', '-f', help="if input node features should be used", action='store_true')
    parser.add_argument('-threads', '-t', help="the number of threads to use", type=int, default=4)
    parser.add_argument('-connectivity', '-co', help="the kind of connectivity to use: either delaunay, knn, radial, fully_connected, all", type=str, default='delaunay')
    parser.add_argument('-loads_model', '-l', help="the kind of load model to use: either OneWT, TwoWT or both", type=str, default='TwoWT')

    args = parser.parse_args()

    generate_graphs(args.config, args.num_layouts, args.num_inflows, args.dset_path, args.threads, args.node_scada_features, args.connectivity, args.loads_model)

