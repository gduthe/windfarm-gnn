from models import WindFarmGNN
from data import GraphFarmsDataset
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

error_type = "rel"

assert error_type in ['abs', 'rel', 'none']

def evaluate(trained_model_path: str, test_dataset_path: str):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # assert os.path.exists(model_path), f'Error: File {model_path} does not exist'
    model_data = torch.load(trained_model_path, map_location=device)
    model_dict, train_set_stats, config = model_data['model_state_dict'], model_data['trainset_stats'], model_data['config']

    # init dataset and dataloader
    test_dataset = GraphFarmsDataset(root_path=test_dataset_path, rel_wd=config.hyperparameters.rel_wd)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = WindFarmGNN(**config.hyperparameters, **config.model_settings)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    model.trainset_stats = train_set_stats

    num_t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_t_params:,}')

    # lists to store the results
    x = []
    y = []
    y_pred = []
    rmse = []
    maxe = []
    mape = []
    mae = []
    globs = []

    highest_alpha = 0
    
    with torch.no_grad():
        print('Evaluating model {} on test set'.format(trained_model_path))
        for i_batch, data in enumerate(test_loader):

            if i_batch % 1000 == 0:
                print(i_batch)

            # if i_batch > 100:
            #     break
            
            data = data.to(device)
            if data.globals[0][-1] > highest_alpha:
                highest_alpha = data.globals[0][-1]
            x += [data.x.squeeze().cpu().numpy()]
            globs += [data.globals.squeeze().cpu().numpy()]

            # compute the batch validation loss
            data = model(data, denorm_output=True)
            
            # gather ground truth and predictions
            y += [data.y.squeeze().cpu().numpy()]
            y_pred += [data.x.squeeze().cpu().numpy()]
            rmse += [torch.sqrt(torch.nn.functional.mse_loss(data.x, data.y, reduction='none').mean(dim=0)).cpu().numpy()]
            l1 = torch.nn.functional.l1_loss(data.x, data.y, reduction='none')
            maxe += [l1.max(dim=0)[0].cpu().numpy()]
            mape += [(l1/torch.maximum(data.y.abs(), torch.tensor(np.finfo(np.float64).eps))).mean(dim=0).cpu().numpy()]
            mae += [l1.mean(dim=0).cpu().numpy()]

    print(np.mean(np.vstack(mape), axis=0))
    print(np.median(np.vstack(mape), axis=0))
    print(np.mean(np.vstack(rmse), axis=0))
    print(np.max(np.vstack(maxe), axis=0))
    print(np.mean(np.vstack(mae), axis=0))
    exit()

    yaws = None
    y_preds = None
    y_trues = None
    rel_errors = None
    abs_errors = None
    speeds = None
    global_speeds = None
    t1_speeds = None

    print("HIGHEST ALPHA", highest_alpha)

    max_vals = [0, 0, 0, 0, 0, 0, 0, 0]
    # y and y_pred are lists, where each list item is a n_turbines x 8 numpy array
    # power, ws, ti, del_blfw, del_blew, del_ttyaw, del_tbss, del_tbfa
    # ['DEL_BLFW', 'DEL_BLEW', 'DEL_TTYAW', 'DEL_TBSS', 'DEL_TBFA']
    # n_turbines can vary in length
    for i in range(len(y)):
        if y[i].ndim == 1:
            continue
        y_i = y[i][:, :]
        y_pred_i = y_pred[i][:, :]
        x_i = x[i]

        if i == 0:
            yaws = np.array(x_i[:, 0][:, None])
        else:
            yaws = np.vstack((yaws, x_i[:, 0][:, None]))

        rel_error = np.abs((y_i - y_pred_i) / y_i) * 100
        abs_error = np.abs(y_i - y_pred_i)

        if i == 0:
            rel_errors = rel_error
            abs_errors = abs_error
        else:
            rel_errors = np.vstack((rel_errors, rel_error))
            abs_errors = np.vstack((abs_errors, abs_error))

        if i == 0:
            speeds = np.array(y[i][:, 1][:, None])
        else:
            speeds = np.vstack((speeds, y[i][:, 1][:, None]))

        if i == 0:
            y_preds = np.array(y_pred[i][:, :])
        else:
            y_preds = np.vstack((y_preds, y_pred[i][:, :]))

        if i == 0:
            y_trues = np.array(y[i][:, :])
        else:
            y_trues = np.vstack((y_trues, y[i][:, :]))

        if i == 0:
            global_speeds = np.array(globs[i][0]).reshape((1, 1))
            t1_speeds = np.array(np.mean(y_pred[i][:, 1], axis=0)).reshape((1, 1))
        else:
            global_speeds = np.vstack((global_speeds, np.array(globs[i][0]).reshape((1, 1))))
            t1_speeds = np.vstack((t1_speeds, np.array(np.mean(y_pred[i][:, 1], axis=0)).reshape((1, 1))))

        for j in range(y_i.shape[1]):
            max_val = np.max(y_i[:, j])
            if max_val > max_vals[j]:
                max_vals[j] = max_val

    def plot_var_vs_var(ax, fig, x, y, title, xtitle, ytitle, colors, colortitle, cmap='viridis'):
        ax.set_title(title)
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        sc = ax.scatter(x, y, c=colors, cmap=cmap, s=2)
        ax.set_xlim([np.min(x), np.max(x)])
        ax.set_ylim([0, ax.get_ylim()[1]])
        fig.colorbar(sc, label=colortitle)

    # Absolute predictions vs ws, absolute predictions vs yaw, relative error, r2 score 

    # if error_type == 'abs':
    #     y_title = 'Absolute Error (kNm)'
    # if error_type == 'rel':
    #     y_title = 'Relative Error (%)'
    # if error_type == 'none':
    #     y_title = 'Absolute Prediction (kNm)'

    fig, axs = plt.subplots(7, 4, figsize=(20, 4*7))

    # Can plot power and all DELs vs wind speed no problem. Maybe also TI
    vars_to_plot = [0, 2, 4, 3, 5, 6, 7]
    y_titles = [r'$P (kW)$', r'$TI (-)$', r'$DEL_{bl,ew} (kNm)$', r'$DEL_{bl,fw} (kNm)$', r'$DEL_{tt,yaw} (kNm)$', r'$DEL_{tb,ss} (kNm)$', r'$DEL_{tb,fa} (kNm)$']
    
    x_title = "Wind Speed (m/s)"
    # y = np.vstack(y)
    y = y_preds

    for i, var in enumerate(vars_to_plot):
        plot_var_vs_var(axs[i, 0], fig, speeds, y[:, var], f"Prediction of {y_titles[i]} vs {r'$V_w [m/s]$'}", x_title, y_titles[i], yaws, 'Yaw Angle (deg)')
    
    x_title = "Yaw Angle (deg)"
    x_var_name = r'$\gamma [^\circ]$'

    for i, var in enumerate(vars_to_plot):
        plot_var_vs_var(axs[i, 1], fig, yaws, y[:, var], f"Prediction of {y_titles[i]} vs {x_var_name}", x_title, y_titles[i], speeds, 'Wind Speed (m/s)', cmap='cividis')


    for i, var in enumerate(vars_to_plot):
        residuals = y_trues[:, var] - y[:, var]
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_trues[:, var] - np.mean(y_trues[:, var]))**2)
        r2 = 1 - (ss_res/ss_tot)
        plot_var_vs_var(axs[i, 2], fig, y[:, var], y_trues[:, var], f"{y_titles[i]} - {r'$Y$'} vs {r'$Y_{pred}$'}", y_titles[i], y_titles[i], speeds, 'Wind Speed (m/s)')
        axs[i, 2].set_xlim([np.min(y[:, var]), np.max(y[:, var])])
        axs[i, 2].set_ylim([np.min(y[:, var]), np.max(y[:, var])])
        axs[i, 2].plot([np.min(y[:, var]), np.max(y[:, var])], [np.min(y[:, var]), np.max(y[:, var])], 'r--')
        axs[i, 2].text(0.75, 0.05, f"{r'$R^2$'}: {r2:.3f}", transform=axs[i, 2].transAxes)
        # plot_var_vs_var(axs[i, 2], fig, speeds, rel_errors[:, var], f"Rel. Error of {y_titles[i]} vs {r'$V_w [m/s]$'}", x_title, y_title, yaws, 'Yaw Angle (deg)')

    x_title = "Wind Speed (m/s)"

    for i, var in enumerate(vars_to_plot):
        plot_var_vs_var(axs[i, 3], fig, speeds, abs_errors[:, var], f"Abs. Error of {y_titles[i]} vs {r'$V_w [m/s]$'}", x_title, y_titles[i], yaws, 'Yaw Angle (deg)')

    # plot_var_vs_var(axs[0, 0], fig, x_values, errors[:, 0], "Power", x_title, y_title, color, 'Yaw Angle (deg)')

    # plot_confidence_interval(x_values, errors[:, 1], "WS", x_title, y_title, color)
    # plot_confidence_interval(x_values, errors[:, 2], "TI", x_title, y_title, color)
    # plot_confidence_interval(x_values, errors[:, 4], "Edgewise", x_title, y_title,  color)
    # plot_confidence_interval(x_values, errors[:, 3], "Flapwise", x_title, y_title, color)
    # plot_confidence_interval(x_values, errors[:, 5], "Yaw", x_title, y_title,  color)
    # plot_confidence_interval(x_values, errors[:, 6], "Side-side", x_title, y_title,  color)
    # plot_confidence_interval(x_values, errors[:, 7], "Fore-aft", x_title, y_title,  color)
    
    

    # plt.scatter(global_speeds, t1_speeds)
    # plt.show()
    plt.tight_layout()
    plt.savefig('error_plot.png')


if __name__ == "__main__":

    evaluate('./runs/best.pt', '../graph_farms/generated_graphs/test_set/OneWT/fully_connected')