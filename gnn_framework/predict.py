import argparse
from box import Box
from models import WindFarmGNN
from finetuning.minlora import add_lora
from data import GraphFarmsDataset
import os
import yaml
from torch_geometric.loader import DataLoader
import torch
import numpy as np

def predict(trained_model_dir:str, test_dataset_path:str, model_version='best', sample_index=None):
    """ Function to predict the output of a trained WindFarmGNN model on a given test dataset.
        The function will load the trained model and the test dataset and return the predictions.
        
        args:
        trained_model_dir: str, path to the trained model directory
        test_dataset_path: str, path to the testing dataset
        model_version: str, the version of the model to use for prediction (default is 'best')
        sample_index: int, the index of the sample to predict, if not passed all test samples are predicted
    """
    
    # load the config file
    config = Box.from_yaml(filename=os.path.join(trained_model_dir, 'config.yml'), Loader=yaml.FullLoader)

    # init dataset and dataloader
    test_dataset = GraphFarmsDataset(root_path=test_dataset_path, rel_wd=config.hyperparameters.rel_wd)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # initialize the model given the dataset properties and the config
    model = WindFarmGNN(**config.hyperparameters, **config.model_settings)

    # use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # recover the saved state and the saved training set statistics
    checkpoint = torch.load(os.path.join(trained_model_dir, 'trained_models/{}.pt'.format(model_version)), map_location=device)
    model.trainset_stats = checkpoint['trainset_stats']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # for lora finetuned models, add the lora layers and load the lora state dict
    if 'lora_state_dict' in checkpoint:
        add_lora(model)
        model.load_state_dict(checkpoint['lora_state_dict'], strict=False)  

    # print the number of trainable parameters
    num_t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters: {}'.format(num_t_params))

    # send the modelto the gpu if available
    model.to(device)
    model.eval()

    # lists to store the results
    y = []
    y_pred = []
    rmse = []
    maxe = []
    mape = []
    lxs = []
    
    with torch.no_grad():
        print('Evaluating model {} ({}) on test set'.format(trained_model_dir, model_version))
        for i_batch, data in enumerate(test_loader):
            
            # for single sample prediction
            if sample_index is not None and i_batch != sample_index:
                continue
            
            data = data.to(device)

            # compute the batch validation loss
            data, latent_x = model(data, denorm=True, return_latent=True)
            lxs += [latent_x.cpu().numpy()]
            
            # gather ground truth and predictions
            y += [data.y.squeeze().cpu().numpy()]
            y_pred += [data.x.squeeze().cpu().numpy()]
            rmse += [torch.sqrt(torch.nn.functional.mse_loss(data.x, data.y, reduction='none')).mean(dim=0).cpu().numpy()]
            l1 = torch.nn.functional.l1_loss(data.x, data.y, reduction='none')
            maxe += [l1.max(dim=0)[0].cpu().numpy()]
            mape +=  [(l1/torch.maximum(data.y.abs(), torch.tensor(np.finfo(np.float64).eps))).mean(dim=0).cpu().numpy()]

    if sample_index is not None:
        return y[0], y_pred[0], rmse[0], maxe[0], mape[0], lxs[0]
    else:
        return y, y_pred, rmse, maxe, mape, lxs


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_dir', '-d', help="path to the trained model directory", type=str, required=True)
    parser.add_argument('--test_dataset_path', '-t', help="path to the testing dataset", type=str, required=True)
    parser.add_argument('--model_version', '-v', help="version of the model to use for prediction", type=str, default='best')
    parser.add_argument('--sample_index', '-s', help="index of the sample to predict, if not passed all test samples are predicted", type=int, default=None)
    
    args = parser.parse_args()
    predict(args.trained_model_dir, args.test_dataset_path, args.model_version, args.sample_index)