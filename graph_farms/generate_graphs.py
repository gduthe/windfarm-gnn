from layout_generator import LayoutGenerator
from rv_generator import UQRVBCGenerator
from pywake_processor import simulate_farm
from utils import to_graph
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
        layout: tuple, the layout to process
        inflow_df: pd.DataFrame, the inflow conditions for the layout
        layout_id: str, the id of the layout
        dset_path: str, the path to save the dataset
        node_scada_features: bool, if the graphs should have node features which 'replicate' SCADA (power, ws, ti)
        connectivity: str, the kind of connectivity to use: either delaunay, knn, radial, fully_connected, all ()
        loads_model: str, the kind of load model to use: either OneWT, TwoWT or both (for comparison)
    """
    
    assert (loads_model in ['OneWT', 'TwoWT', 'both'])
    assert (connectivity in ['delaunay', 'knn', 'radial', 'fully_connected', 'all'])

    # extract info from the layout: coords of the wind turbines, form, min_dist for radius connectivity
    form = layout[0][0]
    wt_coords = layout[0][1]
    min_dist = layout[1]

    # create an info tuple
    info = (str(form), str(min_dist), str(len(wt_coords)), str(layout_id))

    # choose load model
    if loads_model == 'both':
        loads_list = ['OneWT', 'TwoWT']
    else:
        loads_list = [loads_model]

    # create temp folder to store graphs
    with  tempfile.TemporaryDirectory() as tempdir:

        # loop over the loads model (used only if both is selected)
        for loads_method in loads_list:
            # simulate using pywake
            power, loads = simulate_farm(inflow_df, wt_coords, loads_method)

            # loop over the connectivity type saving each in a different folder
            for c in ['delaunay', 'knn', 'radial', 'fully_connected']:

                # parse the raw points to a graph based on current connectivity
                g = to_graph(wt_coords, connectivity=c, min_dist=min_dist, add_edge='polar')

                # get the dataset path for this connectivity
                if loads_model == 'both':
                    dataset_path = os.path.join(dset_path, loads_method, c)
                else:
                    dataset_path = os.path.join(dset_path, c)
                os.makedirs(dataset_path, exist_ok=True)

                # open a zip file and store the individual inflows for the layout as separate graphs
                zip_path = os.path.join(dataset_path, '_'.join(info)) + '.zip'
                with ZipFile(zip_path, 'w') as zf:
                    for i in range(0, len(torch.Tensor(power.ws.values))):

                        if node_scada_features:
                            g.x = torch.Tensor(np.array([power.Power.values[:, i], power.WS_eff.values[:, i],
                                                         power.TI_eff.values[:, i]])).T
                            t = torch.Tensor()
                            for s in loads.sensor.values:
                                t = torch.cat((t, torch.Tensor(np.array([loads.sel(sensor=s).DEL.values[:, i]]))))
                        else:
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


def generate_graphs(config_path:str, num_layouts:int, num_inflows:int, dset_path:str, num_threads=1, node_features=False,
                    loads_model='TwoWT'):
    """ Main function to generate a bunch of wind farm graphs in parallel for a number of layouts and inflow conditions.
        The function will save the graphs in zip files in the given dataset path 
    # load the config
    config = Box.from_yaml(filename=config_path, Loader=yaml.FullLoader)

    # first generate the random layouts using the LayoutGenerator
    layout_list = [LayoutGenerator(config, name='IEA34').random_generation(min_dist=True, form=True) for i in range(num_layouts)]

    # then generate the random inflow conditions
    bc_generator = UQRVBCGenerator(config)
    inflows = bc_generator.generate_all_bcs(num_samples=num_inflows*num_layouts, plot=False)
    inflow_df = pd.DataFrame(data=inflows)
    
    # uncomment the following code to get a uniform sample of inflow directions from 0 to 360 for all inflow conditions
    # this will multiply the number of inflows by 360 
    # wd_values = wd_values = np.arange(0, 360, 5)
    # replicated_df = pd.DataFrame(np.repeat(inflow_df.values,  len(wd_values), axis=0), columns=inflow_df.columns)
    # replicated_df['wd'] = np.tile(wd_values, len(inflow_df))
    # num_inflows = len(wd_values)* num_inflows
    # inflow_df = replicated_df
    
    # inflow_df.iloc[:, 3:] = inflow_df.iloc[0, 3:] turn on to get same u and ti for all samples

    # loop through the layouts, processing them in parallel
    Parallel(n_jobs=num_threads)(delayed(process_one_layout)
        (layout, inflow_df.iloc[list(range(i*num_inflows, i*num_inflows + num_inflows))], str(i).zfill(len(str(num_layouts))),
         dset_path, node_features, loads_model) for i, layout in enumerate(layout_list))


if __name__ == "__main__":
    # set the args of the script
    parser = argparse.ArgumentParser()
    parser.add_argument('-rv_config', '-rvc', help="path to config file for the BC random variable config file ", type=str, required=False, default='rv_config.yml')
    parser.add_argument('-num_layouts', '-nl', help="number of layouts to generate", type=int, required=False, default=100)
    parser.add_argument('-num_inflows', '-ni', help="number of inflows to generate per layout", type=int, required=False, default=10)
    parser.add_argument('-dset_path', '-d', help="path for the dataset to generate", type=str, required=False, default='./generated_graphs/train_set/')
    parser.add_argument('-node_features', '-f', help="if input node features should be used", action='store_true')
    parser.add_argument('-threads', '-t', help="the number of threads to use", type=int, default=4)
    parser.add_argument('-loads_model', '-l', help="the kind of load model to use: either OneWT, TwoWT or both", type=str, default='TwoWT')

    args = parser.parse_args()

    generate_graphs(args.rv_config, args.num_layouts, args.num_inflows, args.dset_path, args.threads, args.node_features, args.loads_model)

