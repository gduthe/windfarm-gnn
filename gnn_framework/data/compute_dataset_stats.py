import numpy as np
import torch

def compute_dataset_stats(train_loader, device, avoid_zero_div=False):
    """ Helper function to compute the dataset statistics for the normalization of the data.
        Uses a dataloader for parrallel computation of the statistics.
    """
    with torch.no_grad():
        # iterative variables for the globals
        globals = []
        glob_max = []
        glob_min = []
        glob_sum = torch.zeros(train_loader.dataset.num_glob_features, requires_grad=False)
        glob_sum_squared = torch.zeros(train_loader.dataset.num_glob_features, requires_grad=False)

        # iterative variables for the edge attributes
        ea_min = []
        ea_max = []
        ea_sum = torch.zeros(train_loader.dataset.num_edge_features, requires_grad=False)
        ea_sum_squared = torch.zeros(train_loader.dataset.num_edge_features, requires_grad=False)

        # iterative variables for the y values
        y_min = []
        y_max = []
        y_sum = torch.zeros(train_loader.dataset.num_node_output_features, requires_grad=False)
        y_sum_squared = torch.zeros(train_loader.dataset.num_node_output_features, requires_grad=False)

        if train_loader.dataset.num_node_features != 0:
            # iterative variables for the x values
            x_min = []
            x_max = []
            x_sum = torch.zeros(train_loader.dataset.num_node_features, requires_grad=False)
            x_sum_squared = torch.zeros(train_loader.dataset.num_node_features, requires_grad=False)

        n_nodes = 0
        n_edges = 0
        for data in train_loader:
            # compute batch globals stats
            glob_min.append(data.globals.min(dim=0).values.tolist())
            glob_max.append(data.globals.max(dim=0).values.tolist())
            glob_sum += data.globals.sum(dim=0)
            glob_sum_squared += (data.globals ** 2).sum(dim=0)
            globals += data.globals.tolist()

            # compute batch edge_stats
            ea_min.append(data.edge_attr.min(dim=0).values.tolist())
            ea_max.append(data.edge_attr.max(dim=0).values.tolist())
            ea_sum += data.edge_attr.sum(dim=0)
            ea_sum_squared += (data.edge_attr ** 2).sum(dim=0)
            n_edges += data.edge_attr.shape[0]

            # compute batch y stats
            y_min.append(data.y.min(dim=0).values.tolist())
            y_max.append(data.y.max(dim=0).values.tolist())
            y_sum += data.y.sum(dim=0)
            y_sum_squared += (data.y**2).sum(dim=0)
            n_nodes += data.y.shape[0]

            # compute batch x stats if needed
            if train_loader.dataset.num_node_features != 0:
                x_min.append(data.x.min(dim=0).values.tolist())
                x_max.append(data.x.max(dim=0).values.tolist())
                x_sum += data.x.sum(dim=0)
                x_sum_squared += (data.x ** 2).sum(dim=0)

        # save final global stats
        n_graphs = len(train_loader.dataset)
        glob_stats = {'max': torch.tensor(np.max(glob_max, axis=0), dtype=torch.float, device=device, requires_grad=False),
                      'min': torch.tensor(np.min(glob_min, axis=0), dtype=torch.float, device=device, requires_grad=False),
                      'mean': (glob_sum/n_graphs).to(device),
                      'std': torch.sqrt(glob_sum_squared/n_graphs - (glob_sum/n_graphs)**2).to(device)}

        # save final edge attributes stats
        edge_stats = {'max': torch.tensor(np.max(ea_max, axis=0), dtype=torch.float, device=device, requires_grad=False),
                      'min': torch.tensor(np.min(ea_min, axis=0), dtype=torch.float, device=device, requires_grad=False),
                      'mean': (ea_sum/n_edges).to(device),
                      'std': torch.sqrt(ea_sum_squared/n_edges - (ea_sum/n_edges)** 2).to(device)}

        # save final y stats
        y_stats = {'max': torch.tensor(np.max(y_max, axis=0), dtype=torch.float, device=device, requires_grad=False),
                   'min': torch.tensor(np.min(y_min, axis=0), dtype=torch.float, device=device, requires_grad=False),
                   'mean': (y_sum/n_nodes).to(device),
                   'std': torch.sqrt(y_sum_squared/n_nodes - (y_sum/n_nodes)**2).to(device)}

        # save final x stats
        if train_loader.dataset.num_node_features != 0:
            x_stats = {'max': torch.tensor(np.max(x_max, axis=0), dtype=torch.float, device=device, requires_grad=False),
                       'min': torch.tensor(np.min(x_min, axis=0), dtype=torch.float, device=device, requires_grad=False),
                       'mean': (x_sum/n_nodes).to(device),
                       'std': torch.sqrt(x_sum_squared/n_nodes - (x_sum/n_nodes)** 2).to(device)}
        else:
            x_stats = None
            
        # avoid zero division error if needed
        tol = 1e-6
        if avoid_zero_div:
            for stats in [glob_stats, edge_stats, y_stats, x_stats]:
                if stats is not None:
                    stats['std'][stats['std'] < tol] = 1.0
                    stats['min'][stats['min'].abs() < tol] = 1.0

        return {'x': x_stats, 'y': y_stats, 'edges': edge_stats, 'globals': glob_stats}