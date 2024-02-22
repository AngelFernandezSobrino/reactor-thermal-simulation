from math import pi

import numpy as np

from water import Water

class Steel:
    density: float = 7800.0 # kg/m^3
    specific_heat_capacity: float = 450.0 # J/kg/K

class Reactor:
    class HeatExchanger:
        product_surface: float = 10.0 # m^2

        metal_weight: float = 230.0 # kg

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

class Parameters:
    
    time = 60*60 # s

    class Discretization:
        nodes: int = 20
        time_step: float = 0.02 # s/step

    class Boundaries:
        input_water_temperature = 70 # degC
        air_temperature = 25.0 # degC
        starting_water_temperature = 25.0 # degC
        starting_metal_temperature = 25.0 # degC
        starting_product_temperature = 25.0 # degC

class NodesParameters:
    class Water:
        surface_metal: float = pi * Reactor.HeatExchanger.water_pipe_radius * 2 * Reactor.HeatExchanger.water_pipe_length / Parameters.Discretization.nodes # m^2
        volume: float = pi * Reactor.HeatExchanger.water_pipe_radius ** 2 * Reactor.HeatExchanger.water_pipe_length / Parameters.Discretization.nodes # m^3

        @classmethod
        def mass(cls, temperature_degC: float) -> float:
            return Water.rho(temperature_degC) * cls.volume # kg
        
    class Metal:
        surface_product: float = Reactor.HeatExchanger.product_surface / Parameters.Discretization.nodes
        surface_air: float = Reactor.HeatExchanger.air_surface / Parameters.Discretization.nodes
        volume: float = Reactor.HeatExchanger.metal_weight / Steel.density / Parameters.Discretization.nodes
        mass: float = Reactor.HeatExchanger.metal_weight / Parameters.Discretization.nodes


class SimulationData:
    def __init__(self) -> None:
        self.nodes_wm_h = np.zeros(Parameters.Discretization.nodes, dtype=np.float64) # W
        self.nodes_ww_h = np.zeros(Parameters.Discretization.nodes, dtype=np.float64) # W
        self.nodes_ma_h = np.zeros(Parameters.Discretization.nodes, dtype=np.float64) # W
        self.nodes_mp_h = np.zeros(Parameters.Discretization.nodes, dtype=np.float64) # W

        self.nodes_w_t  = np.full(Parameters.Discretization.nodes + 1, Parameters.Boundaries.starting_water_temperature, dtype=float) # degC
        self.nodes_m_t  = np.full(Parameters.Discretization.nodes, Parameters.Boundaries.starting_metal_temperature, dtype=float) # degC

        self.p_t = Parameters.Boundaries.starting_product_temperature # degC

        self.nodes_w_t[0] = Parameters.Boundaries.input_water_temperature