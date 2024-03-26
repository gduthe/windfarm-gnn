import pandas as pd
import numpy as np
import xarray as xr
from py_wake.deficit_models import NiayifarGaussianDeficit
from py_wake.examples.data.iea34_130rwt._iea34_130rwt import IEA34_130_Base, IEA34_130_2WT_Surrogate, ThreeRegionLoadSurrogates, IEA34_130_PowerCtSurrogate
from py_wake.utils.tensorflow_surrogate_utils import TensorflowSurrogate
from py_wake.examples.data import example_data_path
from py_wake.superposition_models import LinearSum
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.turbulence_models import CrespoHernandez
from py_wake.site._site import UniformSite
import tensorflow as tf
import os
from pathlib import Path

# ignore tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# we redefine here the OneWT surrogate, as we need to override the TI input
# in the original PyWake IEA34_130_1WT_Surrogate implementation TI_eff is set, which does not override the TI input
class IEA34_130_1WT_Surrogate(IEA34_130_Base):

    def __init__(self):
        surrogate_path = Path(example_data_path) / 'iea34_130rwt' / 'one_turbine'
        loadFunction = ThreeRegionLoadSurrogates(
            [[TensorflowSurrogate(surrogate_path / s, n) for n in self.set_names] for s in self.load_sensors],
            input_parser=lambda ws, TI=.1, Alpha=0: [ws, TI, Alpha])
        powerCtFunction = IEA34_130_PowerCtSurrogate(
            surrogate_path,
            input_parser=lambda ws, TI, Alpha=0: [ws, TI, Alpha])
        IEA34_130_Base.__init__(self, powerCtFunction=powerCtFunction, loadFunction=loadFunction)


def simulate_farm(inflow_df: pd.DataFrame, positions: np.ndarray, loads_method:str):
    """ Function to simulate the power and loads of a wind farm given the inflow conditions and the
        wind turbine positions using PyWake. The function will return the simulated power and loads 
        for each turbine.
        
        args:
        inflow_df: pd.DataFrame, the inflow conditions for the wind farm
        positions: np.ndarray, the wind turbine positions
        loads_method: str, the kind of load model to use: either OneWT or TwoWT
    """
    assert (loads_method in ['OneWT', 'TwoWT'])
    xr.set_options(display_expand_data=False)
    site = UniformSite()

    ws = inflow_df.u
    wd = inflow_df.wd
    ti = inflow_df.ti
    yaw = inflow_df.hflowang
    alpha = inflow_df.shearexp
    x = positions[:, 0]
    y = positions[:, 1]
    
    if loads_method == 'OneWT':
        wt = IEA34_130_1WT_Surrogate()
    else:
        wt = IEA34_130_2WT_Surrogate()

    wf_model = PropagateDownwind(site, wt, wake_deficitModel=NiayifarGaussianDeficit(),
                                 superpositionModel=LinearSum(),
                                 turbulenceModel=CrespoHernandez())

    farm_sim = wf_model(x, y,  # wind turbine positions
                            wd=wd,  # Wind direction 'time series'
                            ws=ws,  # Wind speed 'time series'
                            TI=ti/100,  # Turbulence intensity 'time series'
                            yaw=yaw,  # yaw angles 'time series'
                            Alpha=alpha, # shear exponent 'time series'
                            time=True,  # time stamps
                            )
    
    farm_sim['duration'] = farm_sim.time.values
    sim_loads = farm_sim.loads(method=loads_method)
    
    return farm_sim, sim_loads