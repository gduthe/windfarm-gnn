import pandas as pd
import numpy as np
import xarray as xr
from py_wake.deficit_models import NiayifarGaussianDeficit, BastankhahGaussianDeficit
# from py_wake.examples.data.iea34_130rwt._iea34_130rwt import IEA34_130_2WT_Surrogate, IEA34_130_1WT_Surrogate
from py_wake.superposition_models import LinearSum, SquaredSum
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.turbulence_models import CrespoHernandez
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.rotor_avg_models import RotorCenter, GaussianOverlapAvgModel
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular, default_additional_models
from py_wake.site._site import UniformSite
from surrogates.utils import Custom_IEA34_Surrogate
from py_wake.flow_map import HorizontalGrid
import matplotlib.pyplot as plt
# import tensorflow as tf
# import os
# from pathlib import Path

# ignore tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def simulate_farm(inflow_df: pd.DataFrame, positions: np.ndarray, yaw_angles: np.ndarray, operating_modes: np.ndarray, loads_method:str):
    """ Function to simulate the power and loads of a wind farm given the inflow conditions and the
        wind turbine positions using PyWake. The function will return the simulated power and loads 
        for each turbine.
        
        args:
        inflow_df: pd.DataFrame, the inflow conditions for the wind farm
        positions: np.ndarray, the wind turbine positions
        yaw_angles: np.ndarray, the yaw angles for each wind turbine
        operating_modes: np.ndarray, the operating modes for each wind turbine
        loads_method: str, the kind of load model to use: either OneWT or TwoWT
    """
    assert (loads_method in ['OneWT', 'TwoWT'])
    xr.set_options(display_expand_data=False)
    site = UniformSite()

    ws = inflow_df.u
    wd = inflow_df.wd
    ti = inflow_df.ti
    alpha = inflow_df.shearexp
    x = positions[:, 0]
    y = positions[:, 1]

    # Instead of using the 'yaw noise', as in the original model, we use exact yaw angles
    # Since we are simulating a range of inflow conditions, we need to repeat the yaw angles
    # yaw = np.repeat(yaw_angles[np.newaxis, :], len(ws), axis=0) # yaw angles constant for all ws
    yaw = np.vstack([np.random.permutation(yaw_angles) for _ in range(len(ws))]).reshape((len(yaw_angles), len(ws), 1)) # yaw angles shuffled for all ws

    # Since we are simulating a range of inflow conditions, we need to repeat the operating modes
    # operating_modes = np.repeat(operating_modes[np.newaxis, :], len(ws), axis=0) # operating modes constant for all ws
    operating_modes = np.vstack([np.random.permutation(operating_modes) for _ in range(len(ws))]).reshape((len(yaw_angles), len(ws), 1)) # operating modes shuffled for all ws

    # wt = Custom_IEA34_Surrogate() if loads_method == 'OneWT' else IEA34_130_2WT_Surrogate()
    wt = Custom_IEA34_Surrogate()

    if loads_method == 'TwoWT':
        raise NotImplementedError('TwoWT is not supported in this version of the code')

    if loads_method == 'TwoWT':
        # Small patch to manually add power dependency on yaw alignment to the surrogate model
        # (Only needed for the TwoWT class; the custom OneWT class has this inherently implemented in the surrogates)
        wt.powerCtFunction.model_lst = default_additional_models # Default yaw misalignment model and air density model
        for model in wt.powerCtFunction.model_lst:
            wt.powerCtFunction.add_inputs(model.required_inputs, model.optional_inputs) # Add required and optional inputs to the powerCtFunction

    # Adding operating modes to the powerCtFunction, allowing the turbine to be 'off' or 'on'
    wt.powerCtFunction = PowerCtFunctionList(
        key = 'operating',
        powerCtFunction_lst = [
            PowerCtTabular(ws = [0, 100], power = [0, 0], power_unit = 'w', ct = [0, 0]),   # Turbine is 'off'
            wt.powerCtFunction                                                              # Turbine is 'on'
        ],
        default_value = 1 # Default value is the turbine is 'on'
    )

    wf_model = PropagateDownwind(site,
                                 wt,
                                 wake_deficitModel=BastankhahGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel(), use_effective_ws=True),
                                 deflectionModel=JimenezWakeDeflection(),
                                 superpositionModel=SquaredSum(),
                                 turbulenceModel=CrespoHernandez())
    
    # if loads_method == 'OneWT':
    #     # Another small patch to prevent the OneWT load method from throwing a tantrum when 'operating', 'tilt' and 'yaw' are passed
    #     def loads(self, ws, **kwargs):
    #         for key in ['operating', 'tilt', 'yaw']:
    #             kwargs.pop(key, None)
    #         return self.loadFunction(ws, **kwargs)
    #     wf_model.windTurbines.loads = lambda ws, **kwargs: loads(wf_model.windTurbines, ws, **kwargs)

    farm_sim = wf_model(x, y,  # wind turbine positions
                        wd=wd,  # Wind direction time series
                        ws=ws,  # Wind speed time series
                        TI = ti/100,  # Turbulence intensity time series
                        yaw = yaw,  # Yaw angles time series
                        operating = operating_modes,  # Operating modes time series,
                        tilt = 0,
                        time=True,  # time stamps
                        Alpha=alpha)
    
    # Plot flow map
    farm_sim.flow_map(HorizontalGrid(resolution=200), time=0).plot_wake_map()
    plt.show()
    
    farm_sim['duration'] = farm_sim.time.values
    sim_loads = farm_sim.loads(method=loads_method)

    # PyWake's OneWT calculation does not return the series of TI values, so we need to add it manually
    farm_sim['TI'] = ti/100

    return farm_sim, sim_loads, yaw, operating_modes
