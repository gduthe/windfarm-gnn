import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from py_wake.utils.model_utils import fix_shape
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtSurrogate
from pathlib import Path
from py_wake.wind_turbines.wind_turbine_functions import FunctionSurrogates

class SurrogateNet(nn.Module):

    def __init__(self):
        super(SurrogateNet, self).__init__()
        
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class Surrogate:

    def __init__(self, model_path):
            
        self.Net = SurrogateNet()
        model_data = torch.load(model_path)
        self.Net.load_state_dict(model_data['state_dict'])

        self.Xmin = model_data['Xmin']
        self.Xmax = model_data['Xmax']
        self.Ymin = model_data['Ymin']
        self.Ymax = model_data['Ymax']

        self.n_inputs = model_data['n_inputs']
        self.input_channel_names = model_data['input_channel_names']
        self.input_channel_units = model_data['input_channel_units']

        self.n_outputs = model_data['n_outputs']
        self.output_channel_names = model_data['output_channel_names']
        self.output_channel_units = model_data['output_channel_units']

        self.wohler_exponent = model_data['wohler_exponent']

    def __call__(self, x):
            
        x = (x - self.Xmin) / (self.Xmax - self.Xmin)
        y = self.Net(torch.tensor(x).float()).detach().numpy()[:, 0]
        y = y * (self.Ymax - self.Ymin) + self.Ymin

        return y
    
class IEA34_130_Base(WindTurbine):
    # load_sensors = ['del_blade_flap', 'del_blade_edge', 'del_tower_bottom_fa', 'del_tower_bottom_ss',
    #                 'del_tower_top_torsion']
    # set_names = ['below_cut_in', 'operating', 'above_cut_out']

    def __init__(self, powerCtFunction, loadFunction):
        WindTurbine.__init__(self, 'IEA 3.4MW', diameter=130, hub_height=110,
                             powerCtFunction=powerCtFunction,
                             loadFunction=loadFunction)
        # self.loadFunction.output_keys = self.load_sensors

class ThreeRegionLoadSurrogate(FunctionSurrogates):

    def __init__(self, model_path, input_parser, sensors, regions):

        this_file_dir = Path(__file__).parent.resolve()

        nets = [[Surrogate(f'{this_file_dir}/{model_path}/{region}/{sensor}.pth') for region in regions] for sensor in sensors]
        input_parser = lambda ws, TI_eff=.1, Alpha=0, yaw=0: [ws, TI_eff, Alpha, yaw]
        output_keys = [fs[0].output_channel_names for fs in nets[0]]

        FunctionSurrogates.__init__(self, nets, input_parser, output_keys)

    def __call__(self, ws, run_only=slice(None), **kwargs):

        ws_flat = ws.ravel()
        x = self.get_input(ws=ws, **kwargs)
        x = np.array([fix_shape(v, ws).ravel() for v in x]).T

        def predict(fs):

            output = np.empty(len(x))

            for fs_, m in zip(fs, [ws_flat < 4, (ws_flat >= 4) & (ws_flat <= 25), ws_flat > 25]):
                if m.sum():
                    output[m] = fs_(x[m])
            return output
        return [predict(fs).reshape(ws.shape) for fs in np.asarray(self.function_surrogate_lst)[run_only]]
    
    @property
    def wohler_exponents(self):
        return [fs[0].wohler_exponent for fs in self.function_surrogate_lst]
    
class OneRegionPowerSurrogate(PowerCtSurrogate):

    def __init__(self, model_path):

        this_file_dir = Path(__file__).parent.resolve()

        power_model_path = f'{this_file_dir}/surrogates/models/mid/Power.pth'
        ct_model_path = f'{this_file_dir}/surrogates/models/mid/CT.pth'

        self.power_surrogate = Surrogate(power_model_path)
        self.ct_surrogate = Surrogate(ct_model_path)

        PowerCtSurrogate.__init__(
            self,
            power_surrogate=self.power_surrogate,
            power_unit='kW',
            ct_surrogate=self.ct_surrogate,
            input_parser=lambda ws, TI_eff=.1, Alpha=0, yaw=0: [ws, TI_eff, Alpha, yaw]
        )

    def __call__(self, ws, run_only, **kwargs):

        ws_shape = ws.shape
        ws = np.atleast_1d(ws)

        oper = (ws >= 4) & (ws <= 25)
        cutout = ws > 25

        if np.sum(oper) > 0:
            kwargs = {k: fix_shape(v, ws)[oper] for k, v in kwargs.items()}
            x = self.get_input(ws=ws[oper], **kwargs)
            x = np.array([fix_shape(v, ws).ravel() for v in x]).T

        if np.sum(oper) > 0:
            if run_only == 0:
                y = self.power_surrogate(x)
                power = np.zeros_like(ws, dtype=y.dtype)
                power[oper] = y
                return y.reshape(ws_shape)
            else:
                y = self.ct_surrogate(x)
                ct = np.full(ws.shape, 0.06, dtype=y.dtype)
                ct[oper] = y
                ct[cutout] = 0
                return y.reshape(ws_shape)
        else:
            if run_only == 0:
                return np.zeros(ws_shape)
            else:
                ct = np.full(ws_shape, 0.06)
                ct[cutout] = 0
                return ct
    
class Custom_IEA34_Surrogate(IEA34_130_Base):

    load_sensors = ['DEL_BLFW', 'DEL_BLEW', 'DEL_TTYAW', 'DEL_TBSS', 'DEL_TBFA']

    def __init__(self):
        loadFunction = ThreeRegionLoadSurrogate('surrogates/models', lambda ws, TI_eff=.1, Alpha=0, yaw=0: [ws, TI_eff, Alpha, yaw], self.load_sensors, ['low', 'mid', 'high'])
        powerCtFunction = OneRegionPowerSurrogate()
        IEA34_130_Base.__init__(self, powerCtFunction=powerCtFunction, loadFunction=loadFunction)