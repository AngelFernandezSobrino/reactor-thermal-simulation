import math
import numpy as np

import iapws  # type: ignore
import numpy as np
from typing import List

class Water:

    linspace_min: int = 5
    linspace_max: int = 95
    linspace_ammount: int = 91
    linspace_step: float = (linspace_max - linspace_min) / (linspace_ammount - 1)
    
    temperature: float = np.linspace(linspace_min, linspace_max, linspace_ammount) # type: ignore
    rho_lt: List[float] = []
    viscosity_lt: List[float] = []
    k_lookuptable: List[float] = []
    cp_lookuptable: List[float] = []

    pressure = 0.1 # MPa


    @classmethod
    def interpolate(cls, temperature_degC):
        index = math.trunc((temperature_degC - cls.linspace_min)/cls.linspace_step)
        return cls.rho_lt[index] + (cls.rho_lt[index+1] - cls.rho_lt[index]) * (temperature_degC - cls.temperature[index]) / (cls.temperature[index+1] - cls.temperature[index])

    @classmethod
    def init_lookup_tables(cls):
        cls.rho_lt = [iapws.IAPWS95(P=0.1, T = temperature_degC+273.13).rho for temperature_degC in cls.temperature]
        cls.viscosity_lt = [iapws._iapws._Viscosity(iapws.IAPWS95(P=0.1, T = temperature_degC+273.13).rho, temperature_degC + 273.13) for temperature_degC in cls.temperature]
        cls.k_lookuptable = [iapws.IAPWS95(P=0.1, T = temperature_degC+273.13).k for temperature_degC in cls.temperature]
        cls.cp_lookuptable = [iapws.IAPWS95(P=0.1, T = temperature_degC+273.13).cp*1000 for temperature_degC in cls.temperature]

    @classmethod
    def iapws_rho(cls, temperature_degC: float) -> float: # kg/m^3
        return  iapws.IAPWS95(P=cls.pressure, T = temperature_degC+273.13).rho

    @classmethod
    def iapws_viscosity(cls, temperature_degC: float) -> float: # Pa*s
        return iapws._iapws._Viscosity(iapws.IAPWS95(P=cls.pressure, T = temperature_degC+273.13).rho, temperature_degC + 273.13)

    @classmethod
    def iapws_k(cls, temperature_degC: float) -> float: # W/m/K
        return iapws.IAPWS95(P=cls.pressure, T = temperature_degC+273.13).k
    
    @classmethod
    def iapws_cp(cls, temperature_degC: float) -> float: # J/kg/K
        return iapws.IAPWS95(P=cls.pressure, T = temperature_degC+273.13).cp*1000

    @classmethod
    def rho(cls, temperature_degC: float) -> float:
        # return 980.0
        return cls.rho_lt[math.trunc((temperature_degC)/cls.linspace_step)]
    
    @classmethod
    def mu(cls, temperature_degC: float) -> float:
        return cls.viscosity_lt[math.trunc((temperature_degC)/cls.linspace_step)]
    
    @classmethod
    def k(cls, temperature_degC: float) -> float:
        # return 0.63
        return cls.k_lookuptable[math.trunc((temperature_degC - cls.linspace_min)/cls.linspace_step)]

    @classmethod
    def cp(cls, temperature_degC: float) -> float:
        # return 4190.0
        return cls.cp_lookuptable[math.trunc((temperature_degC - cls.linspace_min)/cls.linspace_step)]

Water.init_lookup_tables()

if __name__ == "__main__":

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import copy
    
    figure_default_size = copy.copy(mpl.rcParams['figure.figsize'])
    figure_default_size[0] = 20
    figure_water_data = plt.figure(figsize=figure_default_size)

    axes_density_data, axes_viscosity_data, axes_thermal_conductivity_data, axes_specific_heat_capacity_data = figure_water_data.subplots(1, 4) # type: ignore

    axes_density_data.plot(Water.temperature, Water.rho_lt)
    axes_density_data.set_xlabel('Temperature [degC]')
    axes_density_data.set_ylabel('Density [kg/m^3]')
    axes_density_data.set_title('Density')

    axes_viscosity_data.plot(Water.temperature, Water.viscosity_lt)
    axes_viscosity_data.set_xlabel('Temperature [degC]')
    axes_viscosity_data.set_ylabel('Viscosity [Pa*s]')
    axes_viscosity_data.set_title('Viscosity')

    axes_thermal_conductivity_data.plot(Water.temperature, Water.k_lookuptable)
    axes_thermal_conductivity_data.set_xlabel('Temperature [degC]')
    axes_thermal_conductivity_data.set_ylabel('Thermal conductivity [W/m/K]')
    axes_thermal_conductivity_data.set_title('Thermal conductivity')

    axes_specific_heat_capacity_data.plot(Water.temperature, Water.cp_lookuptable)
    axes_specific_heat_capacity_data.set_xlabel('Temperature [degC]')
    axes_specific_heat_capacity_data.set_ylabel('Specific heat capacity [J/kg/K]')
    axes_specific_heat_capacity_data.set_title('Specific heat capacity')

    plt.show()