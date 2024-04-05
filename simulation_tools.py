from math import pi

import numpy as np

from water import Water

class Steel:
    density: float = 7800.0 # kg/m^3
    specific_heat_capacity: float = 450.0 # J/kg/K

class Reactor:
    class HeatExchanger:
        product_surface: float = 10.0 # m^2

        metal_weight: float = 2400.0 # kg

        height: float = 2.0 # m

        radius: float = 0.5 # m

        diameter: float = radius * 2 # m
        
        air_surface: float = pi * diameter * height # m^2

        water_pipe_length: float = 60.0 # m
        water_pipe_radius: float = 0.005 # m

        metal_air_heat_transfer_coefficient: float = 200 # W/m^2/K
        metal_product_heat_transfer_coefficient: float = 400 # W/m^2/K

class Product:
    specific_heat_capacity: float = 3890.0 # J/kg/K
    mass: float = 100.0 # kg

class SimulationParameters:
    
    time = 60*30 # s

    class Discretization:
        time_step: np.float128 = np.float128(0.02) # s/step

    class Boundaries:
        input_water_temperature = 80 # degC
        air_temperature = 25.0 # degC
        starting_water_temperature = 25.0 # degC
        starting_metal_temperature = 25.0 # degC
        starting_product_temperature = 25.0 # degC

class ModelParameters:
    class Water:
        surface_metal: float = pi * Reactor.HeatExchanger.water_pipe_radius * 2 * Reactor.HeatExchanger.water_pipe_length # m^2
        volume: float = pi * Reactor.HeatExchanger.water_pipe_radius ** 2 * Reactor.HeatExchanger.water_pipe_length # m^3

        @classmethod
        def mass(cls, temperature_degC: float) -> float:
            return Water.rho(temperature_degC) * cls.volume # kg
        
    class Metal:
        surface_product: float = Reactor.HeatExchanger.product_surface
        surface_air: float = Reactor.HeatExchanger.air_surface
        volume: float = Reactor.HeatExchanger.metal_weight / Steel.density
        mass: float = Reactor.HeatExchanger.metal_weight


class SimulationData:
    def __init__(self) -> None:
        self.wm_h: np.float128 = np.float128(0)
        self.ma_h: np.float128 = np.float128(0)
        self.mp_h: np.float128 = np.float128(0)

        self.w_t: np.float128 = np.float128(SimulationParameters.Boundaries.starting_water_temperature) # degC
        self.m_t: np.float128 = np.float128(SimulationParameters.Boundaries.starting_metal_temperature) # degC
        self.p_t: np.float128 = np.float128(SimulationParameters.Boundaries.starting_product_temperature) # degC

        self.pipe_alpha: np.float128 = np.float128(0)
        self.air_alpha: np.float128 = np.float128(0)

    def __str__(self) -> str:
        return f'Water: {self.w_t:4.2f} degC  Metal: {self.m_t:4.2f} degC  Product: {self.p_t:4.2f} degC  Water-Metal: {self.wm_h:12.0f} W  Metal-Product: {self.mp_h:10.0f} W  Metal-Air: {self.ma_h:12.0f} W  Pipe Alpha: {self.pipe_alpha:10.0f} W/m^2/K  Air Alpha: {self.air_alpha:10.2f} W/m^2/K'
