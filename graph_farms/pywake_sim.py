import pandas as pd
import numpy as np
import xarray as xr
from py_wake.deficit_models import NiayifarGaussianDeficit, BastankhahGaussianDeficit
from py_wake.examples.data.iea34_130rwt._iea34_130rwt import IEA34_130_2WT_Surrogate, IEA34_130_1WT_Surrogate
from py_wake.superposition_models import LinearSum, SquaredSum
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.turbulence_models import CrespoHernandez
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.rotor_avg_models import RotorCenter, GaussianOverlapAvgModel
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular, default_additional_models
from py_wake.site._site import UniformSite
import tensorflow as tf
import os

# ignore tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
    yaw = np.vstack([np.random.permutation(yaw_angles) for _ in range(len(ws))]) # yaw angles shuffled for all ws

    # Since we are simulating a range of inflow conditions, we need to repeat the operating modes
    # operating_modes = np.repeat(operating_modes[np.newaxis, :], len(ws), axis=0) # operating modes constant for all ws
    operating_modes = np.vstack([np.random.permutation(operating_modes) for _ in range(len(ws))]) # operating modes shuffled for all ws

    wt = IEA34_130_1WT_Surrogate() if loads_method == 'OneWT' else IEA34_130_2WT_Surrogate()

    # Small patch to manually add power dependency on yaw alignment to the surrogate model
    wt.powerCtFunction.model_lst = default_additional_models # Default yaw misalignment model and air density model
    for model in wt.powerCtFunction.model_lst:
        wt.powerCtFunction.add_inputs(model.required_inputs, model.optional_inputs) # Add required and optional inputs to the powerCtFunction

    wt.powerCtFunction = PowerCtFunctionList(
        key = 'operating',
        powerCtFunction_lst = [
            PowerCtTabular(ws = [0, 100], power = [0, 0], power_unit = 'w', ct = [0, 0]), # Turbine is 'off'
            wt.powerCtFunction                                                    # Turbine is 'on'
        ],
        default_value = 1 # Default value is the turbine is 'on'
    )

    wf_model = PropagateDownwind(site,
                                 wt,
                                 wake_deficitModel=BastankhahGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel(), use_effective_ws=True),
                                 deflectionModel=JimenezWakeDeflection(),
                                 superpositionModel=SquaredSum(),
                                 turbulenceModel=CrespoHernandez())

    farm_sim = wf_model(x, y,  # wind turbine positions
                        wd=wd,  # Wind direction time series
                        ws=ws,  # Wind speed time series
                        TI = ti/100,  # Turbulence intensity time series
                        yaw = yaw,  # Yaw angles time series
                        operating = operating_modes,  # Operating modes time series,
                        tilt = 0,
                        time=True,  # time stamps
                        Alpha=alpha)
    
    farm_sim['duration'] = farm_sim.time.values
    sim_loads = farm_sim.loads(method=loads_method)

    return farm_sim, sim_loads, yaw, operating_modes