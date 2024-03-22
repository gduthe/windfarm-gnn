import torch
from zipfile import ZipFile
import io
import os
import itertools
from torch_geometric.data import Dataset
import math

class GraphFarmsDataset(Dataset):
    """ Dataset comprised of PyWake or HAWC2 farmwide simulations that have been parsed into graphs.
        The dataset should be organized as follows:
        root_path
            ├── layout1.zip
            │    ├── graph1.pt
            │    ├── graph2.pt
            │    ...
            ├── layout2.zip
            ...

        Where each zip file contains the different graphs (inflows) for a given layout. 
        The graphs are stored as PyTorch Geometric Data.
        
        
        args:
        root_path: str, the path to the root directory of the dataset
        rel_wd: bool, whether to include the relative wind direction as an edge feature
    """

    def __init__(self, root_path: str, rel_wd=True):
        super().__init__()
        
        self.__root_path = root_path
        
        # get the list of all the zip files and their contents
        zip_list = [os.path.join(path, name) for path, subdirs, files in os.walk(root_path) for name in files]
        
        # create a list of tuples with the zip file path and the contents
        zip_content_list = []
        for zip_path in zip_list:
            with ZipFile(zip_path, 'r') as zf:
                zip_content_list.append(list(zip([zip_path]*len(zf.namelist()),zf.namelist())))
        
        # store the zip file and its contents in a single matrix
        self.zip_matrix = list(itertools.chain(*zip_content_list))
        
        # get the total number of graphs in the dataset
        self.__num_graphs = len(self.zip_matrix)
        
        # initialize the relative wind direction flag and the stats dicts
        self.rel_wd = rel_wd
        self.x_stats, self.y_stats, self.edge_stats, self.glob_stats = None, None, None, None

    @property
    def num_glob_features(self) -> int:
        r"""Returns the number of global features in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        return data.globals.shape[1]

    @property
    def num_glob_output_features(self) -> int:
        r"""Returns the number of global output features in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        return data.globals_y.shape[1]

    @property
    def num_node_output_features(self) -> int:
        r"""Returns the number of node output features in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        return data.y.shape[1]

    def __len__(self):
        return self.__num_graphs

    def __getitem__(self, idx):
        # read the zip file and select the data to load in by index
        with ZipFile(self.zip_matrix[idx][0], 'r') as zf:
            with zf.open(self.zip_matrix[idx][1]) as item:
                stream = io.BytesIO(item.read())
                data = torch.load(stream)

        # add relative wind direction as an edge feature
        if self.rel_wd:
            edge_rel_wd = math.radians(data.globals[1]) - data.edge_attr[:, 1]
            data.edge_attr = torch.cat((data.edge_attr, edge_rel_wd.unsqueeze(1)), dim=1)

        # make sure all features are float
        data.edge_attr = data.edge_attr.float()
        data.pos = data.pos.float()
        data.globals = data.globals.float().unsqueeze(0)

        return data
