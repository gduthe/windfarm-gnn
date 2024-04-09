import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from py_wake.utils.model_utils import fix_shape
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtSurrogate
from pathlib import Path
from py_wake.wind_turbines.wind_turbine_functions import FunctionSurrogates
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular

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

    def __init__(self, powerCtFunction, loadFunction):
        WindTurbine.__init__(self, 'IEA 3.4MW',
                             diameter=130,
                             hub_height=110,
                             powerCtFunction=powerCtFunction,
                             loadFunction=loadFunction)

class ThreeRegionLoadSurrogate(FunctionSurrogates):

    def __init__(self, model_path, input_parser, sensors, regions):

        this_file_dir = Path(__file__).parent.resolve()

        nets = [[Surrogate(f'{this_file_dir}/{model_path}/{region}/{sensor}_best.pth') for region in regions] for sensor in sensors]
        input_parser = lambda ws, TI_eff=.1, Alpha=0, yaw=0, **kwargs: [ws, TI_eff, Alpha, yaw]
        output_keys = [fs_set[0].output_channel_names[0] for fs_set in nets]

        FunctionSurrogates.__init__(self, nets, input_parser, output_keys)

    def __call__(self, ws, run_only=slice(None), **kwargs):

        ws_flat = ws.ravel()
        x = self.get_input(ws=ws, **kwargs)
        x = np.array([fix_shape(v, ws).ravel() for v in x]).T

        operating_modes = kwargs.get('operating', np.ones_like(ws_flat))
        operating_modes = fix_shape(operating_modes, ws).ravel().T
        print(x)
        print(operating_modes)

        def predict(fs):

            output = np.empty(len(x))

            for fs_, m in zip(fs, [ws_flat < 4, (ws_flat >= 4) & (ws_flat <= 25) & (operating_modes == 1), (ws_flat > 25) | ((operating_modes == 0) & (ws_flat >= 4) & (ws_flat <= 25))]):
                if m.sum():
                    output[m] = fs_(x[m])
            return output
        print([predict(fs).reshape(ws.shape) for fs in np.asarray(self.function_surrogate_lst)[run_only]])
        return [predict(fs).reshape(ws.shape) for fs in np.asarray(self.function_surrogate_lst)[run_only]]
    
    @property
    def wohler_exponents(self):
        return [fs[0].wohler_exponent for fs in self.function_surrogate_lst]
    
class OneRegionPowerSurrogate(PowerCtSurrogate):

    def __init__(self, model_path):

        this_file_dir = Path(__file__).parent.resolve()

        power_model_path = f'{this_file_dir}/{model_path}/mid/Power_best.pth'
        ct_model_path = f'{this_file_dir}/{model_path}/mid/CT_best.pth'

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

        ws = np.atleast_1d(ws)

        oper = (ws >= 4) & (ws <= 25)
        cutout = ws > 25

        if np.sum(oper) > 0:
            kwargs = {k: fix_shape(v, ws)[oper] for k, v in kwargs.items()}
            x = self.get_input(ws=ws[oper], **kwargs)
            x = np.array([fix_shape(v, ws[oper]).ravel() for v in x]).T
            if run_only == 0:
                y = self.power_surrogate(x).reshape(ws[oper].shape)
                power = np.zeros_like(ws, dtype=y.dtype)
                power[oper] = y
                return power
            else:
                y = self.ct_surrogate(x).reshape(ws[oper].shape)
                ct = np.full(ws.shape, 0.06, dtype=y.dtype)
                ct[oper] = y
                ct[cutout] = 0
                return ct
        else:
            if run_only == 0:
                return np.zeros(ws.shape)
            else:
                ct = np.full(ws.shape, 0.06)
                ct[cutout] = 0
                return ct
    
class Custom_IEA34_Surrogate(IEA34_130_Base):

    load_sensors = ['DEL_BLFW', 'DEL_BLEW', 'DEL_TTYAW', 'DEL_TBSS', 'DEL_TBFA']
    load_regions = ['low', 'mid', 'nonop']

    def __init__(self):
        loadFunction = ThreeRegionLoadSurrogate('models', lambda ws, TI_eff=.1, Alpha=0, yaw=0: [ws, TI_eff, Alpha, yaw], self.load_sensors, self.load_regions)
        powerCtFunction = OneRegionPowerSurrogate('models')
        IEA34_130_Base.__init__(self, powerCtFunction=powerCtFunction, loadFunction=loadFunction)

        self.powerCtFunction = PowerCtFunctionList(
            key='operating',
            powerCtFunction_lst=[PowerCtTabular(ws=[0, 100], power=[0, 0], power_unit='w', ct=[0, 0]), # 0=No power and ct
                                 self.powerCtFunction], # 1=Normal operation
            default_value=1)