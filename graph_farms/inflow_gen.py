import numpy as np
from scipy.stats import qmc, weibull_min, norm, truncnorm  # Update scipy to latest version
import matplotlib.pyplot as plt
import seaborn as sns
import random

class InflowGenerator:
    """ Inflow generation class, generates inflow conditions for a wind farm.
        Needs to be a class to obtain a consistent sobol sequence across all the parameters.
        
        args:
        config_dict: dict, the settings for the inflow generation
    """
    def __init__(self, config_dict: dict):
        self.inflow_settings = config_dict['inflow_settings']
        self.turbine_settings = config_dict['turbine_settings']
        self.yaw_settings = config_dict['yaw_settings']

        self.sampler = qmc.Sobol(d=5, scramble=True)

    def __gen_wind_velocities(self, sobol_samples: np.array):
        # compute scale param and get shape param
        a = 2 * self.inflow_settings['mean_V'] / np.sqrt(np.pi)
        k_V = self.inflow_settings['k_V']

        # calculate the probabilities corresponding to the bounds
        bounds = np.array([self.turbine_settings['cutin_u'], self.turbine_settings['cutout_u']])
        P_bounds = weibull_min.cdf(bounds, k_V, loc=0, scale=a)

        # A little trick to avoid cyclicity
        random.seed(42)
        random.shuffle(sobol_samples)

        # scale and shift samples to cover P_bounds
        x = P_bounds[0] + sobol_samples * (P_bounds[1] - P_bounds[0])

        # compute inflow velocity
        u = weibull_min.ppf(x, k_V, scale=a)
        
        return u

    def __gen_turbulence(self, u: np.array):
        # generate free stream turbulence and turbulence intensity
        mu_T = self.inflow_settings['Iref'] * (0.75 * u + 3.8)
        sig_T = 1.4 * self.inflow_settings['Iref']
        m = np.log((mu_T ** 2) / np.sqrt(sig_T ** 2 + mu_T ** 2))
        v = np.sqrt(np.log((sig_T / mu_T) ** 2 + 1))
        turb = np.random.lognormal(mean=m, sigma=v)
        ti = 100 * turb / u

        # generate turbulence length scale
        mu_TL = self.inflow_settings['mean_turb_Lscale'] * (0.75 * u + 3.6)
        sig_TL = 1.4 * self.inflow_settings['Iref']
        m = np.log((mu_TL ** 2) / np.sqrt(sig_TL ** 2 + mu_TL ** 2))
        v = np.sqrt(np.log((sig_TL / mu_TL) ** 2 + 1))

        # Turbulence length similar to equation 15 in https://wes.copernicus.org/articles/3/533/2018/
        du_dz = 0.2  # assume constant positive exponent
        tl = self.turbine_settings['height_above_ground'] * np.random.lognormal(mean=m, sigma=v) / 100.00 / du_dz
        
        return turb, ti, tl

    def __gen_yaw_misalignment(self, sobol_samples: np.array, u: np.array, ti: np.array):
        # generate yaw misalignment from yaw error
        mu_yaw = np.log(u) - 3
        sig_yaw = 15. / u
        ti = ti
        bounds = np.array([self.yaw_settings['yawL'], self.yaw_settings['yawH']])

        HFlowAng = np.fromiter(
            (truncnorm.rvs((bounds[0] - mu_yaw) / sig_yaw, (bounds[1] - mu_yaw) / sig_yaw, loc=mu_yaw,
                           scale=sig_yaw)), dtype=np.float64)

        return HFlowAng

    def __gen_wind_direction(self, sobol_samples: np.array):
        bounds = np.array([0, 360])
        wd = np.random.uniform(bounds[0], bounds[1], len(sobol_samples))

        return wd

    def __gen_freestream_shear(self, u: np.array):
        mu_alpha = 0.088 * (np.log(u) - 1)
        sig_alpha = 1.0 / u
        bounds = np.array([self.inflow_settings['min_alpha'], self.inflow_settings['max_alpha']])

        shearExp = np.fromiter(
            (truncnorm.rvs((bounds[0] - mu_alpha) / sig_alpha, (bounds[1] - mu_alpha) / sig_alpha, loc=mu_alpha,
                           scale=sig_alpha)), dtype=np.float64)
        return shearExp

    def __gen_air_properties(self, sobol_samples: np.array):
        qL = -20  # lower bound
        qH = 50  # Upper bound
        bounds = np.array([qL, qH])

        # calculate the probabilities corresponding to the bounds
        mean_temp = self.inflow_settings['mean_air_temp']
        COV_temp = self.inflow_settings['COV_air_temp']
        P_bounds = norm.cdf(bounds, loc=mean_temp, scale=COV_temp * mean_temp)

        # scale and shift samples to cover P_bounds
        x = P_bounds[0] + sobol_samples * (P_bounds[1] - P_bounds[0])

        # variability in air temperature in Kelvin
        airtemp = 273 + norm.ppf(x, mean_temp, COV_temp * mean_temp)

        # mean air density as a function of temperature
        frho = -0.0043 * (airtemp) + 2.465

        # variability in air density
        rho = np.random.normal(loc=frho, scale=frho * 0.01)

        # air viscosity(dynamic) as a function of temperature
        fnu = 5E-08 * airtemp + 4E-06

        # nu
        mu_nu = fnu
        sig_nu = fnu * self.inflow_settings['COV_Dyn_Viscosity']
        m = np.log((mu_nu ** 2) / np.sqrt(sig_nu ** 2 + mu_nu ** 2))
        v = np.sqrt(np.log((sig_nu / mu_nu) ** 2 + 1))
        nu = np.random.lognormal(mean=m, sigma=v)
        
        return airtemp, rho, nu

    def generate_all_bcs(self, num_samples: int, plot=False):
        samples = self.sampler.random_base2(m=int(np.ceil(np.log2(num_samples))))[:num_samples]
        u = self.__gen_wind_velocities(samples[:, 0])
        wd = self.__gen_wind_direction(samples[:, 1])
        turb, ti, tl = self.__gen_turbulence(u)
        shearExp = self.__gen_freestream_shear(u)
        airtemp, rho, nu = self.__gen_air_properties(samples[:, 2])
        hflowang = self.__gen_yaw_misalignment(samples[:, 3], u, ti)
        plt.show()

        output_dict = {'u': u, 'turb': turb, 'ti': ti, 'tl': tl, 'shearexp': shearExp, 'airtemp': airtemp,
                       'rho': rho, 'nu': nu, 'hflowang': hflowang, 'wd': wd}
        if plot:
            self.plot(output_dict)
        
        return output_dict
    
    def plot(self, output_dict: dict):
         # plotting settings
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.linewidth'] = 2
        plt.rc('text', usetex=True)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # plot wind speed
        u = output_dict['u']
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        sns.histplot(u, stat='density')
        sns.kdeplot(u, color='red', clip=[self.turbine_settings['cutin_u'], self.turbine_settings['cutout_u']])
        ax.set_xlabel('Wind Speed [m/s]')
        ax.set_title('Free Wind Speed Distribution')
        ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
        ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
        ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
        ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')

        # plot turbulence
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        sns.scatterplot(x=u, y=output_dict['turb'], ax=axs[0])
        axs[0].set_xlabel('Wind Speed [m/s]')
        axs[0].set_ylabel('Turbulence [m/s]')
        axs[0].set_title('Turbulence vs. Wind Speed')
        sns.scatterplot(x=u, y=output_dict['ti'], ax=axs[1])
        axs[1].set_xlabel('Wind Speed [m/s]')
        axs[1].set_ylabel('Turbulence Intensity []')
        axs[1].set_title('Turbulence Intensity vs. Wind Speed')
        sns.scatterplot(x=u, y=output_dict['tl'], ax=axs[2])
        axs[2].set_xlabel('Wind Speed [m/s]')
        axs[2].set_ylabel('Turbulence Length [% Chord L]')
        axs[2].set_title('Turbulence Length vs. Wind Speed')
        for ax in axs:
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
        fig.tight_layout(pad=3.0)
        
        
        # plot yaw misalignment
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        sns.scatterplot(x=u, y=output_dict['hflowang'], ax=ax[0])
        ax[0].set_xlabel('Wind Speed [m/s]')
        ax[0].set_ylabel('Inflow horizontal skew [$^\circ$]')
        ax[0].set_title('Inflow horizontal skew vs. Wind Speed')
        ax[0].xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
        ax[0].xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
        ax[0].yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
        ax[0].yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
        sns.histplot(output_dict['hflowang'], ax=ax[1], stat='density')
        sns.kdeplot(output_dict['hflowang'], ax=ax[1], color='red')
        ax[1].set_xlabel('Inflow horizontal skew [$^\circ$]')
        ax[1].set_title('Horizontal Skew Distribution')
        fig.tight_layout(pad=3.0)
        
        # plot shear exponent
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        sns.scatterplot(x=u, y=output_dict['shearexp'], ax=ax)
        ax.set_xlabel('Wind Speed [m/s]')
        ax.set_ylabel('Shear Exponent [m/s]')
        ax.set_title('Shear Exponent vs. Wind Speed')
        ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
        ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
        ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
        ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
        
        # plot wind direction
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        sns.histplot(output_dict['wd'], stat='density')
        sns.kdeplot(output_dict['wd'], color='red', clip=[-10, 360])
        ax.set_xlabel('Wind Direction [$^\circ$]')
        ax.set_title('Wind Direction Distribution')
        ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
        ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
        ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
        ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
        
        # plot air properties
        # first figure - air density plots
        airtemp = output_dict['airtemp']
        rho = output_dict['rho']
        nu = output_dict['nu']
        
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        sns.histplot(airtemp, ax=axs[0, 0], stat='density')
        sns.kdeplot(airtemp, ax=axs[0, 0], color='red')
        axs[0, 0].set_xlabel('Temperature [deg. K]')
        axs[0, 0].set_title('Temperature Distribution')
        sns.scatterplot(x=airtemp, y=rho, ax=axs[0, 1])
        axs[0, 1].set_xlabel('Temperature [deg. K]')
        axs[0, 1].set_ylabel('Air Density [kg/m3]')
        axs[0, 1].set_title('Mean Air Density vs. Temperature')
        sns.histplot(rho, ax=axs[1, 0], stat='density')
        sns.kdeplot(rho, ax=axs[1, 0], color='red')
        axs[1, 0].set_xlabel('Air Density [kg/m3]')
        axs[1, 0].set_title('Air Density Distribution')
        sns.scatterplot(x=airtemp, y=rho, ax=axs[1, 1])
        axs[1, 1].set_xlabel('Temperature [deg. K]')
        axs[1, 1].set_ylabel('Air Density [kg/m3]')
        axs[1, 1].set_title('Air Density vs. Temperature')
        for line in axs:
            for ax in line:
                ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
                ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
                ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
                ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
        fig.tight_layout(pad=3.0)

        # second figure - air viscosity plots
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        sns.scatterplot(x=airtemp, y=nu, ax=axs[0])
        axs[0].set_xlabel('Temperature [deg. K]')
        axs[0].set_ylabel('Air Viscosity (dynamic) [kg/m-s]')
        axs[0].set_title('Mean Air Viscosity vs. Temperature')
        sns.scatterplot(x=airtemp, y=nu, ax=axs[1])
        axs[1].set_xlabel('Temperature [deg. K]')
        axs[1].set_ylabel('Air Viscosity (dynamic) [kg/m-s]')
        axs[1].set_title('Air Viscosity vs. Temperature')
        sns.histplot(rho, ax=axs[2], stat='density')
        sns.kdeplot(rho, ax=axs[2], color='red')
        axs[2].set_xlabel('Air Viscosity (dynamic) [kg/m-s]')
        axs[2].set_title('Air Viscosity Distribution')
        fig.tight_layout(pad=3.0)
        for ax in axs:
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
        
        plt.show()
            
if __name__ == "__main__":
    # example usage
    config_dict = {
        'inflow_settings': {'mean_V': 8.5, 'k_V': 2.0, 'Iref': 0.16, 'mean_turb_Lscale': 0.1, 'min_alpha': 0.05,
                            'max_alpha': 0.2, 'mean_air_temp': 15.0, 'COV_air_temp': 0.02, 'COV_Dyn_Viscosity': 0.01},
        'turbine_settings': {'cutin_u': 3.0, 'cutout_u': 25.0, 'height_above_ground': 90.0},
        'yaw_settings': {'yawL': -30.0, 'yawH': 30.0}
    }
    inflow_gen = InflowGenerator(config_dict)
    inflow_gen.generate_all_bcs(num_samples=1000, plot=True)