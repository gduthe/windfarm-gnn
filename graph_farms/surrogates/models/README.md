# Surrogates Description

The surrogates provided in this folder were trained in three operating regions: low wind (below cut-in, ws < 4m/s), normal wind (4m/s >= ws <= 25ms) and high wind (25 < ws). In low-wind conditions, the rotor is set at a pitch angle of 1 degree and is essentially idle. In operating conditions (normal wind), the turbine acts as normal. In high-wind conditions, blades are feathered to 90 degrees pitch and all controllers are turned off. Note that there is an extra folder called 'nonop': in addition to this feathered high-wind condition, this one is also extended with cases where the turbine is turned off for maintenance during normal wind conditions. The nonop models can thus be used whenever the turbine is shut off between 5m/s and 30m/s.

### Training info

The training data was obtained using openFAST v3.5.1. Turbulence files were generated using TurbSim. Parameters for turbsim, being wind speed, shear exponent and turbulence intensity were obtained using SOBOL sampling. Wind speeds were chosen uniformly over the domain min speed and max speed, turbulence intensities were chosen uniformly between a min and max value determined by the chosen wind speed and the formulas provided by the IEC 61400-1 standard. Shear exponents were chosen similarly, uniformly over a domain based on wind speed. The ROSCO2.7.0 controller was used in the simulation, together with the controller config provided in the IEA37 3.4MW github repo. OpenFAST input files were also taken from this repo. 4096 simulations were run for the operating region; an additional 8192 were run over the entire wind speed range of 0-30 m/s, setting the turbine to parked above 25m/s and to 1-degree pitch below 4m/s. Another 4096 simulations were run using a parked rotor between 4m/s and 25m/s. Simulations were 240s to reduce computation time, as experiments showed that the difference between results obtained from 240s and 600s simulations was quite small. Time-series outputs were postprocessed using the openFAST toolbox to determine DEL loads. Yaw angles of the turbines were sampled concurrently with the inflow conditions in the SOBOL sampler.

The neural networks consist of an input layer of four nodes (wind speed, turbulence intensity, shear exponent and yaw angle), three hidden layers with (32, 64, 32) nodes with leaky ReLu activation, and an output node. Each output variable has its own neural network. 3000 training epochs were run for training, saving the model each time the validation loss decreased, thereby obtaining the model with the lowest validation loss over all epochs. Additional data can be found in the metadata of the model files. Below cut-in, all loads were trained with 8192 samples; in the operating region, loads were trained with 12288 samples; above cut-out, loads were trained with 8192 samples. Furthermore, the operating region + parked rotor loads were trained with 4096 samples. CT and Power were trained with only 8192 samples.

### Surrogate validity regions

The 'low' region surrogates are only valid in the region 0m/s - 4m/s, however it is best to just set power and ct to zero in this region. In the operating ('mid') region, everything is valid between 4m/s and 25 m/s, when the turbine is operating (and thus not off). The 'high' region surrogates are only valid in the region 25m/s - 30m/s, but here too it is best to set power and ct to zero. The 'nonop' region is valid from 4m/s to 30m/s, but ONLY when the turbine is off and the rotor is thus parked. Essentially, its behaviour above 25m/s is similar to that of the 'high' region surrogate.

### Accuracy

Below is a table of metrics that were obtained after training and tuning the surrogates. Note that these values are over the entire domain, and thus encapsulate the accuracy when using each of the three region-specific models in their respective regions.

#### V4

| Variable | r2         | RMSPE         | RMSE            | MAE            | MAPE         |
|----------|------------|---------------|-----------------|----------------|--------------|
| power	   | r2 = 0.989 | RMSPE = 0.084 | RMSE =  112.157 | MAE =   57.666 | MAPE = 0.036 |
| ct	   | r2 = 0.996 | RMSPE = 0.062 | RMSE =    0.017 | MAE =    0.011 | MAPE = 0.045 |
| edgewise | r2 = 0.959 | RMSPE = 0.082 | RMSE =   63.179 | MAE =   43.002 | MAPE = 0.023 |
| flapwise | r2 = 0.984 | RMSPE = 0.105 | RMSE =  327.509 | MAE =  229.838 | MAPE = 0.067 |
| yaw	   | r2 = 0.955 | RMSPE = 0.163 | RMSE =  316.677 | MAE =  209.655 | MAPE = 0.103 |
| SS	   | r2 = 0.916 | RMSPE = 0.201 | RMSE = 2326.209 | MAE = 1429.874 | MAPE = 0.140 |
| FA	   | r2 = 0.917 | RMSPE = 0.165 | RMSE = 3542.388 | MAE = 1859.451 | MAPE = 0.108 |