
# Thermal simulation of heat exchanger for steady water heating

from dataclasses import dataclass
from math import pi
import time
import multiprocessing.shared_memory
from typing import List, Tuple, cast
import ray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import signal

plt.style.use('dark_background')

from water import Water

from heat_equations import heat_rate_conduction_boundary, next_temperature

class Steel:
    density: float = 7800.0 # kg/m^3
    specific_heat_capacity: float = 450.0 # J/kg/K

class Reactor:
    class HeatExchanger:
        product_surface: float = 10.0 # m^2
        air_surface: float = 11.0 # m^2
        metal_weight: float = 230.0 # kg

        water_pipe_length: float = 1.0 # m
        water_pipe_radius: float = 0.005 # m

        metal_air_heat_transfer_coefficient: float = 200 # W/m^2/K
        metal_product_heat_transfer_coefficient: float = 0.0 # W/m^2/K


class Parameters:
    time = 300# s
    class Discretization:
        nodes: int = 20
        time_step: float = 0.05 # s/step

    class Boundaries:
        input_water_temperature = 70 # degC
        air_temperature = 25.0 # degC
        starting_water_temperature = 25.0 # degC
        starting_metal_temperature = 25.0 # degC
        starting_product_temperature = 25.0 # degC
        water_flow_rate = 0.1 # kg/s

class HeatTransferCoefficients:
    water_flow_speed = Parameters.Boundaries.water_flow_rate / 1000 / (pi * Reactor.HeatExchanger.water_pipe_radius ** 2) # m/s

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

class Product:
    specific_heat_capacity: float = 3890.0 # J/kg/K
    mass: float = 100.0 # kg


class SimulationData:
    nodes_wm_h = np.zeros(Parameters.Discretization.nodes, dtype=np.float64) # W
    nodes_ww_h = np.zeros(Parameters.Discretization.nodes, dtype=np.float64) # W
    nodes_ma_h = np.zeros(Parameters.Discretization.nodes, dtype=np.float64) # W
    nodes_mp_h = np.zeros(Parameters.Discretization.nodes, dtype=np.float64) # W

    nodes_w_t  = np.full(Parameters.Discretization.nodes + 1, Parameters.Boundaries.starting_water_temperature, dtype=float) # degC
    nodes_m_t  = np.full(Parameters.Discretization.nodes, Parameters.Boundaries.starting_metal_temperature, dtype=float) # degC

    p_t = np.full(1, Parameters.Boundaries.starting_product_temperature, dtype=float) # degC
    
    Parameters.Boundaries.water_flow_rate = Parameters.Boundaries.water_flow_rate # kg/s
    Parameters.Boundaries.input_water_temperature = Parameters.Boundaries.input_water_temperature # degC
    nodes_w_t[0] = Parameters.Boundaries.input_water_temperature
    Parameters.Boundaries.air_temperature = Parameters.Boundaries.air_temperature # degC




def step():
    for i in range(0, Parameters.Discretization.nodes):
        SimulationData.nodes_wm_h[i] = heat_rate_conduction_boundary(SimulationData.nodes_w_t[i], SimulationData.nodes_m_t[i], NodesParameters.Water.surface_metal, Reactor.HeatExchanger.water_metal_heat_transfer_coefficient)
        SimulationData.nodes_ww_h[i] = SimulationData.Parameters.Boundaries.water_flow_rate * Water.cp(SimulationData.nodes_w_t[i+1]) * (SimulationData.nodes_w_t[i] - SimulationData.nodes_w_t[i+1]) * Parameters.Discretization.time_step
        SimulationData.nodes_ma_h[i] = heat_rate_conduction_boundary(SimulationData.nodes_m_t[i], SimulationData.Parameters.Boundaries.air_temperature, NodesParameters.Metal.surface_air, Reactor.HeatExchanger.metal_air_heat_transfer_coefficient)
        SimulationData.nodes_mp_h[i] = heat_rate_conduction_boundary(SimulationData.nodes_m_t[i], SimulationData.p_t, NodesParameters.Metal.surface_product, Reactor.HeatExchanger.metal_product_heat_transfer_coefficient)

        SimulationData.p_t = SimulationData.p_t + np.sum(SimulationData.nodes_mp_h) * Parameters.Discretization.time_step / (Product.specific_heat_capacity * Product.mass)
        SimulationData.nodes_m_t[i] = next_temperature(SimulationData.nodes_m_t[i], SimulationData.nodes_wm_h[i] - SimulationData.nodes_ma_h[i] - SimulationData.nodes_mp_h[i], Steel.specific_heat_capacity, Reactor.HeatExchanger.metal_weight)
        SimulationData.nodes_w_t[i+1] = next_temperature(SimulationData.nodes_w_t[i+1], (- SimulationData.nodes_wm_h[i] + SimulationData.nodes_ww_h[i]), Water.cp(SimulationData.nodes_w_t[i+1]), Water.rho(SimulationData.nodes_w_t[i+1]) * NodesParameters.Water.volume)

@ray.remote
def calculate_heat_rates_by_index(node_w_t: float, previous_node_w_t: float, node_m_t: float, p_t: float) -> Tuple[float, float, float, float]:
    return NodesParameters.Water.surface_metal * Reactor.HeatExchanger.water_metal_heat_transfer_coefficient * (node_w_t - node_m_t), \
        Parameters.Boundaries.water_flow_rate * Water.cp(node_w_t) * ( - node_w_t + previous_node_w_t), \
        NodesParameters.Metal.surface_air * Reactor.HeatExchanger.metal_air_heat_transfer_coefficient * (node_m_t - Parameters.Boundaries.air_temperature), \
        NodesParameters.Metal.surface_product * Reactor.HeatExchanger.metal_product_heat_transfer_coefficient * (node_m_t - p_t)

@ray.remote
def calculate_next_temperatures_by_index(node_w_t, node_m_t, node_wm_h, node_ww_h, node_ma_h, node_mp_h, time_step) -> Tuple[float, float]:
    return node_m_t + (node_wm_h - node_ma_h - node_mp_h) * time_step / (Steel.specific_heat_capacity * NodesParameters.Metal.mass), \
        node_w_t + (- node_wm_h + node_ww_h) * time_step / (Water.cp(node_w_t) * NodesParameters.Water.mass(node_w_t))


def calculate_heat_rates_by_index_local(node_w_t: float, previous_node_w_t: float, node_m_t: float, p_t: float) -> Tuple[float, float, float, float]:

    return NodesParameters.Water.surface_metal * Reactor.HeatExchanger.water_metal_heat_transfer_coefficient * (node_w_t - node_m_t), \
        Parameters.Boundaries.water_flow_rate * Water.cp(node_w_t) * ( - node_w_t + previous_node_w_t), \
        NodesParameters.Metal.surface_air * Reactor.HeatExchanger.metal_air_heat_transfer_coefficient * (node_m_t - Parameters.Boundaries.air_temperature), \
        NodesParameters.Metal.surface_product * Reactor.HeatExchanger.metal_product_heat_transfer_coefficient * (node_m_t - p_t)

def calculate_next_temperatures_by_index_local(node_w_t, node_m_t, node_wm_h, node_ww_h, node_ma_h, node_mp_h, time_step) -> Tuple[float, float]:
    return node_m_t + (node_wm_h - node_ma_h - node_mp_h) * time_step / (Steel.specific_heat_capacity * NodesParameters.Metal.mass), \
        node_w_t + (- node_wm_h + node_ww_h) * time_step / (Water.cp(node_w_t) * NodesParameters.Water.mass(node_w_t))


# Configurable variables

node_to_measure = 10

data_time: List[float] = []
data_temperature_product: List[float] = []
data_temperature_metal: List[float] = []
data_temperature_water: List[float] = []
data_heat_wm: List[float] = []
data_heat_ww: List[float] = []
data_heat_ma: List[float] = []
data_heat_mp: List[float] = []


# ray.init()

# Simulation time calculated
print(f'Simulation time: {Parameters.time}s')
print(f'Initial time step: {Parameters.Discretization.time_step}s')
print(f'Initial simulation steps: {int(Parameters.time/Parameters.Discretization.time_step)}')

data_time.append(0)
data_temperature_product.append(SimulationData.p_t[0])
data_temperature_metal.append(SimulationData.nodes_m_t[node_to_measure])
data_temperature_water.append(SimulationData.nodes_w_t[node_to_measure+1])

data_heat_wm.append(SimulationData.nodes_wm_h[node_to_measure])
data_heat_ww.append(SimulationData.nodes_ww_h[node_to_measure])
data_heat_ma.append(SimulationData.nodes_ma_h[node_to_measure])
data_heat_mp.append(SimulationData.nodes_mp_h[node_to_measure])

end_simulation = False

def SIGINT_handler(signum, frame):
    global end_simulation
    end_simulation = True
 
signal.signal(signal.SIGINT, SIGINT_handler)


t = time.time()

simulation_running = True
retries = 0
actual_step: int = 1
actual_time: float = 0

while simulation_running and not end_simulation:


    heat_rates_calls = [] # type: ignore
    heat_rates_results = []
    for j in range(0, Parameters.Discretization.nodes):
        # heat_rates_calls.append(calculate_heat_rates_by_index.remote(SimulationData.nodes_w_t[j+1], SimulationData.nodes_w_t[j], SimulationData.nodes_m_t[j], SimulationData.p_t[0]))
        heat_rates_results.append(calculate_heat_rates_by_index_local(SimulationData.nodes_w_t[j+1], SimulationData.nodes_w_t[j], SimulationData.nodes_m_t[j], SimulationData.p_t[0]))
    
    # heat_results = ray.get(heat_rates_calls)
    heat_results_list: list[list[float]] = list(zip(*heat_rates_results)) # type: ignore

    temperature_calls = [] # type: ignore
    temperature_results = []
    for j in range(0, Parameters.Discretization.nodes):
        # temperature_calls.append(calculate_next_temperatures_by_index.remote(SimulationData.nodes_w_t[j+1], SimulationData.nodes_m_t[j], heat_results_list[0][j], heat_results_list[1][j], heat_results_list[2][j], heat_results_list[3][j], Parameters.Discretization.time_step))
        temperature_results.append(calculate_next_temperatures_by_index_local(SimulationData.nodes_w_t[j+1], SimulationData.nodes_m_t[j], heat_results_list[0][j], heat_results_list[1][j], heat_results_list[2][j], heat_results_list[3][j], Parameters.Discretization.time_step))

    # temperature_results = ray.get(temperature_calls)
    temperature_results_list: list[list[float]] = list(zip(*temperature_results)) # type: ignore

    p_t = SimulationData.p_t[0] + np.sum(heat_results_list[3]) * Parameters.Discretization.time_step / (Product.specific_heat_capacity * Product.mass)

    # if i != 1:
    #     for j in range(15):
    #         LINE_UP = "\033[1A"
    #         LINE_CLEAR = "\x1b[2K"
    #         print(LINE_UP, end=LINE_CLEAR)

    m_delta_t = 0
    m_delta_t = 0

    time_step_reduction = False
    for j in range(0, Parameters.Discretization.nodes):
        m_delta_t = abs(temperature_results_list[0][j] - SimulationData.nodes_m_t[j])
        w_delta_t = abs(temperature_results_list[1][j] - SimulationData.nodes_w_t[j+1])

        if m_delta_t > 5 or w_delta_t > 5 :
            time_step_reduction = True

    if time_step_reduction:
        # print()
        # print('-----------------------------------')
        # print()
        # print(f'Excesive temperature change, time step too big: Water temperature change: {w_delta_t} - Metal temperature change: {m_delta_t}')
        # print('Temperature metal old ' + str([f'{i:3.1f} ' for i in SimulationData.nodes_m_t]))
        # print('Temperature metal     ' + str([f'{i:3.1f} ' for i in temperature_results_list[0]]))
        # print('Temperature water old ' + str([f'{i:3.1f} ' for i in SimulationData.nodes_w_t]))
        # print('Temperature water     ' + str([f'{i:3.1f} ' for i in temperature_results_list[1]]))
        # print('Reducing time step...')
        # print(f'Old time step: {Parameters.Discretization.time_step}')
        Parameters.Discretization.time_step = Parameters.Discretization.time_step / 1.3
        # print(f'New time step: {Parameters.Discretization.time_step}')
        continue
    
    actual_time += Parameters.Discretization.time_step

    # print()
    # print('-----------------------------------')
    # print()
    # print('Heat water-metal   ' + str([f'{i:6.0f} ' for i in heat_results_list[0]]))
    # print('Heat water-water   ' + str([f'{i:6.0f} ' for i in heat_results_list[1]]))
    # print('Heat metal-air     ' + str([f'{i:6.0f} ' for i in heat_results_list[2]]))
    # print('Heat metal-product ' + str([f'{i:6.0f} ' for i in heat_results_list[3]]))
    # print('Temperature metal  ' + str([f'{i:6.1f} ' for i in temperature_results_list[0]]))
    # print('Temperature water  ' + str([f'{i:6.1f} ' for i in temperature_results_list[1]]))
    # print()

    # print(f'Step {actual_step}')
    # print(f'Retries: {retries}')
    # print(f'Actual time: {(actual_time):.2f}s')
    # print(f'Actual time step: {Parameters.Discretization.time_step:.2f}s')
    # print(f'Simulation end time: {Parameters.time}s')

    data_time.append(actual_time)
    # data_heat_wm.append(heat_results[node_to_measure][0])
    data_heat_ww.append(heat_rates_results[node_to_measure][1])
    # data_heat_ma.append(heat_results[node_to_measure][2])
    # data_heat_mp.append(heat_results[node_to_measure][3])

    SimulationData.p_t[0] = p_t
    for j in range(0, Parameters.Discretization.nodes):
        SimulationData.nodes_m_t[j] = temperature_results[j][0]
        SimulationData.nodes_w_t[j+1] = temperature_results[j][1] 

    data_temperature_product.append(SimulationData.p_t[0])
    data_temperature_metal.append(SimulationData.nodes_m_t[node_to_measure])
    data_temperature_water.append(SimulationData.nodes_w_t[node_to_measure+1])

    actual_step += 1
    if actual_time > Parameters.time:
        simulation_running = False

    if actual_step % 10 == 0:
        time_step_increase = True
        for j in range(0, Parameters.Discretization.nodes):
            m_delta_t = abs(temperature_results_list[0][j] - SimulationData.nodes_m_t[j])
            w_delta_t = abs(temperature_results_list[1][j] - SimulationData.nodes_w_t[j+1])

            if m_delta_t > 2 or w_delta_t > 2 :
                time_step_increase = False

        if time_step_increase:
            # print()
            # print('-----------------------------------')
            # print()
            # print(f'Temperature change too small, time step too small: Water temperature change: {w_delta_t} - Metal temperature change: {m_delta_t}')
            # print('Temperature metal old ' + str([f'{i:3.1f} ' for i in SimulationData.nodes_m_t]))
            # print('Temperature metal     ' + str([f'{i:3.1f} ' for i in temperature_results_list[0]]))
            # print('Temperature water old ' + str([f'{i:3.1f} ' for i in SimulationData.nodes_w_t]))
            # print('Temperature water     ' + str([f'{i:3.1f} ' for i in temperature_results_list[1]]))

            # print('Increasing time step...')
            # print(f'Old time step: {Parameters.Discretization.time_step}')
            Parameters.Discretization.time_step = Parameters.Discretization.time_step * 1.3
            # print(f'New time step: {Parameters.Discretization.time_step}')


print()
print('-----------------------------------')
print()

print(f'Elapsed time: {time.time() - t}s')

# print('Product temperature data (degC):')
# print([" {0:0.2f}".format(i) for i in data_temperature_product])

# print('Metal temperature data (degC):')
# print(["{0:0.2f}".format(i) for i in data_temperature_metal])
# print('Water temperature data (degC):')
# print(["{0:0.2f}".format(i) for i in data_temperature_water])
# print('Heat wm data (W):')
# print(["{0:0.2f}".format(i) for i in data_heat_wm])
# print('Heat ww data (W):')
# print(["{0:0.2f}".format(i) for i in data_heat_ww])
# print('Heat ma data (W):')
# print(["{0:0.2f}".format(i) for i in data_heat_ma])
# print('Heat mp data (W):')
# print(["{0:0.2f}".format(i) for i in data_heat_mp])

# for index, time_elapsed in enumerate(data_time):
#     print(f'{time_elapsed:3.2f} {data_temperature_water[index]:3.2f} {data_heat_ww[index]:5.2f}')


figure = plt.figure()
axes = cast(mpl.axes.Axes, figure.subplots(1,1))
product_temperature_line, = axes.plot(data_time, data_temperature_product, label="Product temperature")
metal_temperature_line, = axes.plot(data_time, data_temperature_metal, label="Metal temperature")
water_temperature_line, = axes.plot(data_time, data_temperature_water, label="Water temperature")

axes.set_xlabel("Time")
axes.set_ylabel("Temperature (degC)")
legend = axes.legend()

plt.show()

# ray.shutdown()