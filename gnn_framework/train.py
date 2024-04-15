import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from models import WindFarmGNN
from data import GraphFarmsDataset, compute_dataset_stats
from box import Box
import yaml
import os
import argparse
from tqdm import tqdm
from datetime import datetime

def train(config_path: str):
    """ The main training function of the WindFarmGNN model. """
    
    # use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the config file using box for easy access
    config = Box.from_yaml(filename=config_path, Loader=yaml.FullLoader)

    # initialize the datasets and dataloaders
    train_dataset = GraphFarmsDataset(root_path=config.io_settings.train_dataset_path, rel_wd=config.hyperparameters.rel_wd)
    train_loader = DataLoader(train_dataset, batch_size=config.hyperparameters.batch_size, shuffle=True,
                              exclude_keys=[], num_workers=config.run_settings.num_t_workers, pin_memory=True,
                              persistent_workers=False if config.run_settings.num_t_workers == 0 else True)
    
    # add the main training dataset parameters to the config (will be saved with the model for reference)
    config.hyperparameters.node_feature_dim = train_dataset.num_node_features
    config.hyperparameters.edge_feature_dim = train_dataset.num_edge_features
    config.hyperparameters.glob_feature_dim = train_dataset.num_glob_features
    config.hyperparameters.node_out_dim = train_dataset.num_node_output_features
    
    # if validation is needed, initialize the validation dataset and dataloader
    if config.run_settings.validate:
        validate_dataset = GraphFarmsDataset(root_path=config.io_settings.valid_dataset_path, rel_wd=config.hyperparameters.rel_wd)
        validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=False, exclude_keys=[],
                                num_workers=config.run_settings.num_v_workers, pin_memory=True,
                                 persistent_workers=False if config.run_settings.num_v_workers == 0 else True)

    # create model saving dir and copy config file to run dir
    dt = datetime.now()
    run_name = '_'.join([config.model_settings.processor_settings.mp_type,
                          str(config.model_settings.processor_settings.num_mp_steps), 'layers',
                          str(config.model_settings.processor_settings.dropout), 'dropout',
                          str(config.hyperparameters.start_lr), 'lr',
                          str(config.hyperparameters.epochs), 'epochs',
                          str(config.model_settings.node_latent_dim), 'latent_dim', datetime.strftime(dt, "%m_%d_%H_%M")])
    current_run_dir = os.path.join(config.io_settings.run_dir, run_name)
    os.makedirs(os.path.join(current_run_dir, 'trained_models'), exist_ok=True)
    config.to_yaml(filename=os.path.join(current_run_dir, 'config.yml'))

    if os.path.exists('./comet_settings.yml'):
        with open('./comet_settings.yml', 'r') as f:
            comet_settings = yaml.load(f, Loader=yaml.FullLoader)
            if comet_settings['log_experiment']:
                from comet_ml import Experiment
                log_comet = True
                experiment = Experiment(api_key=comet_settings['api_key'], project_name=comet_settings['comet_project'], workspace=comet_settings['workspace'], auto_metric_logging=False)
                experiment.set_name(run_name)
                experiment.log_parameters(config.hyperparameters.to_dict())
                experiment.log_parameters(config.model_settings.to_dict())

    # initialize the model given the dataset properties and the config
    model = WindFarmGNN(**config.hyperparameters, **config.model_settings)

    # if using a pretrained model, load it here
    if config.io_settings.pretrained_model:
        checkpoint = torch.load(config.io_settings.pretrained_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # compute the dataset statistics for normalization and pass them to the model
    model.trainset_stats  = compute_dataset_stats(train_loader, device)

    # send the models to the gpu if available
    model.to(device)

    # print the number of trainable parameters
    num_t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters: {}'.format(num_t_params))

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=float(config.hyperparameters.start_lr))

    # define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.hyperparameters.epochs)

    # training loop
    print('Starting run {} on {}'.format(run_name, next(model.parameters()).device))

    # initialize tqdm progress bar
    pbar = tqdm(total=config.hyperparameters.epochs)
    pbar.set_description('Training')

    # main training loop
    for epoch in range(config.hyperparameters.epochs):
        train_loss = 0
        model.train()

        # mini-batch loop
        for i_batch, data in enumerate(train_loader):
            # send data to device
            data = data.to(device)

            # reset the gradients back to zero
            optimizer.zero_grad()

            # run forward pass and compute the batch training loss
            data = model(data)
            batch_loss = model.compute_loss(data)
            train_loss += batch_loss.item()

            # perform batched SGD parameter update
            batch_loss.backward()
            optimizer.step()

        # compute the epoch training loss
        train_loss = train_loss / len(train_loader)

        # step the scheduler
        if epoch < config.hyperparameters.lr_decay_stop:
            scheduler.step()

        if log_comet:
            with experiment.train():
                experiment.log_metric('loss', train_loss, step=epoch + 1)

        # save the trained model every n epochs
        if (epoch + 1) % config.io_settings.save_epochs == 0:
            torch.save({'model_state_dict': model.state_dict(),'trainset_stats': model.trainset_stats}
                       ,os.path.join(current_run_dir, 'trained_models', 'e{}.pt'.format(epoch + 1)))

        # compute validation loss if required
        if config.run_settings.validate:
            validation_loss = 0
            model.eval()
            with torch.no_grad():
                for i_batch, data in enumerate(validate_loader):
                    # send data to device
                    data = data.to(device)
                    
                    # compute the batch validation loss
                    data = model(data)
                    validation_loss += model.compute_loss(data).item()

            # get the full dataset validation loss for this epoch
            validation_loss = validation_loss / len(validate_loader)

            if log_comet:
                with experiment.validate():
                    experiment.log_metric('loss', validation_loss, step=epoch + 1)

            # save the model with the best validation loss
            if epoch == 0:
                best_validation_loss = validation_loss
            else:
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    torch.save({'model_state_dict': model.state_dict(),'trainset_stats': model.trainset_stats, 'config': config},
                               os.path.join(current_run_dir, 'trained_models', 'best.pt'))

            pbar.set_postfix({'Train Loss': f'{train_loss:.8f}', 'Validation Loss': f'{validation_loss:.8f}'})
            pbar.update(1)
        
        else:
            # display losses and progress bar
            pbar.set_postfix({'Train Loss': f'{train_loss:.8f}'})
            pbar.update(1)
    
    return current_run_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', '-c', help="path to the yaml config file", type=str,
                        required=False, default='config.yml')
    train(config_path=parser.parse_args().yaml_config)