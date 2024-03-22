import torch
from torch.utils.data import random_split
import torch.optim as optim
from torch_geometric.loader import DataLoader
from models import WindFarmGNN
from minlora import add_lora, get_lora_state_dict, get_lora_params
from dataset import GraphFarmsDataset, compute_dataset_stats
from utils import recursively_merge_dicts
from box import Box
import yaml
import os
import argparse
from tqdm import tqdm
from datetime import datetime

def finetune(ft_config_path: str): 
    """ The main finetuning function of the WindFarmGNN model. This function loads in a pretrained model 
        and finetunes it on a new dataset. 4 finetuning methods are available: vanilla, LoRA, scratch and decoder.
        
        args:
        ft_config_path: str, the path to the finetuning config file    
    """
    ft_config = Box.from_yaml(filename=ft_config_path, Loader=yaml.FullLoader)
    
    # detect nan values in the gradients
    torch.autograd.set_detect_anomaly(True)
       
    # use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load the config file for the pretrained model
    pretrained_model_dir = ft_config.io_settings.pretrained_model_dir
    pt_config = Box.from_yaml(filename=os.path.join(pretrained_model_dir, 'config.yml'), Loader=yaml.FullLoader)
    
    # check the finetuning method
    ft_method = ft_config.model_settings.ft_method
    assert ft_method in ['vanilla', 'LoRA', 'scratch', 'decoder']

    # initialize the datasets and dataloaders
    ft_dataset = GraphFarmsDataset(root_path=ft_config.io_settings.dataset_path, rel_wd=pt_config.hyperparameters.rel_wd)
    train_samples = int(ft_config.hyperparameters.train_ratio*len(ft_dataset))
    data_lens = [train_samples, len(ft_dataset)-train_samples]
    train_dataset, validate_dataset = random_split(ft_dataset, data_lens, generator=torch.Generator().manual_seed(ft_config.run_settings.random_seed))

    # for the scratch method, drop the last batch if it is smaller than 20 (training is unstable with small batches)
    if ft_method == 'scratch' and len(train_dataset) > ft_config.hyperparameters.batch_size and (len(train_dataset)%ft_config.hyperparameters.batch_size)<=20:
        drop_last = True
    else:
        drop_last = False
        
    # initialize the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=ft_config.hyperparameters.batch_size, shuffle=True, exclude_keys=[], 
                              num_workers=ft_config.run_settings.num_t_workers, pin_memory=True, drop_last=drop_last,
                            persistent_workers=False if ft_config.run_settings.num_t_workers == 0 else True)
    validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=False, exclude_keys=[],
                            num_workers=ft_config.run_settings.num_v_workers, pin_memory=True,
                                persistent_workers=False if ft_config.run_settings.num_v_workers == 0 else True)
    
    # create model saving dir and copy config file to run dir
    dt = datetime.now()
    run_name = '_'.join(['FT', ft_method, 'tr_{}'.format(train_samples).zfill(3), 'sd_{}'.format(ft_config.run_settings.random_seed), datetime.strftime(dt, "%m_%d_%H_%M")])
    current_run_dir = os.path.join(ft_config.io_settings.run_dir, run_name)
    os.makedirs(os.path.join(current_run_dir, 'trained_models'), exist_ok=True)
    
    merged_configs = Box(recursively_merge_dicts(pt_config, ft_config))
    merged_configs.to_yaml(filename=os.path.join(current_run_dir, 'config.yml'))

    # initialize the model
    model = WindFarmGNN(**pt_config.hyperparameters, **pt_config.model_settings)
    
    # load the weights of the pretrained model if not retraining from scratch
    checkpoint = torch.load(os.path.join(pretrained_model_dir, 'trained_models/{}.pt'.format(ft_config.io_settings.model_version)), map_location=device)
    pretrained_state = checkpoint['model_state_dict']
    if ft_method != 'scratch':
        model.load_state_dict(pretrained_state)
        
    # recompute the trainset stats if needed and the dataset is large enough
    if ft_config.hyperparameters.recompute_stats and len(train_dataset) > 1:
        model.trainset_stats  = compute_dataset_stats(train_loader, device, avoid_zero_div=True)
    else:
        model.trainset_stats = checkpoint['trainset_stats']
    
    # modify the model if using LoRA
    if ft_method == 'LoRA':
        add_lora(model)
        trainable_params =  [{"params": list(get_lora_params(model))},]
        num_t_params = sum(p.numel() for p in get_lora_params(model))
    elif ft_method == 'vanilla' or ft_method == 'scratch':
        trainable_params = model.parameters()
        num_t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif ft_method == 'decoder':
        for p in model.parameters():
            p.requires_grad = False
        for p in model.decoder.parameters():
            p.requires_grad = True
        trainable_params = model.parameters()
        num_t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        raise NotImplementedError
    
    print('Number of trainable parameters: {}'.format(num_t_params))

    # send the models to the gpu if available
    model.to(device)
    
    # define optimizer
    optimizer = optim.Adam(trainable_params, lr=float(ft_config.hyperparameters.start_lr))

    # define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_config.hyperparameters.epochs)

    # training loop
    print('Starting finetuning run {} on {}'.format(run_name, next(model.parameters()).device))

    pbar = tqdm(total=ft_config.hyperparameters.epochs)
    pbar.set_description('Training')

    for epoch in range(ft_config.hyperparameters.epochs):
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
        if epoch < ft_config.hyperparameters.lr_decay_stop:
            scheduler.step()

        # save the trained model every n epochs
        if (epoch + 1) % ft_config.io_settings.save_epochs == 0:
            save_model(model, current_run_dir, 'e{}.pt'.format(epoch + 1), ft_method, pretrained_state)

        if ft_config.run_settings.validate:
            # compute validation loss
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

            # save the model with the best validation loss
            if epoch == 0:
                best_validation_loss = validation_loss
                save_model(model, current_run_dir, 'best.pt', ft_method, pretrained_state)
            else:
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    save_model(model, current_run_dir, 'best.pt', ft_method, pretrained_state)

            pbar.set_postfix({'Train Loss': f'{train_loss:.8f}', 'Validation Loss': f'{validation_loss:.8f}'})
            pbar.update(1)
            
        else:
            # display losses and progress bar
            pbar.set_postfix({'Train Loss': f'{train_loss:.8f}'})
            pbar.update(1)
            
    return current_run_dir

def save_model(model, run_dir, savename, ft_method, pretrained_state):
    save_dict = {'trainset_stats': model.trainset_stats}
    if ft_method == 'LoRA':
        save_dict['model_state_dict'] = pretrained_state
        save_dict['lora_state_dict'] = get_lora_state_dict(model)
    else:
        save_dict['model_state_dict'] = model.state_dict()
    torch.save(save_dict, os.path.join(run_dir, 'trained_models', savename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_config', '-c', help="path to the finetune config file", type=str, default='ft_config.yml')
    
    finetune(parser.parse_args().ft_config)