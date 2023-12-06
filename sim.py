
# Thermal simulation of heat exchanger for steady water heating

from dataclasses import dataclass
from math import pi
import time
import multiprocessing.shared_memory
import ray
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')

from water import Water

class steel:
    density: float = 7800.0 # kg/m^3
    specific_heat_capacity: float = 450.0 # J/kg/K

class heat_exchanger:
    product_surface: float = 10.0 # m^2
    air_surface: float = 11.0 # m^2
    metal_weight: float = 230.0 # kg

    water_pipe_length: float = 1.0 # m
    water_pipe_radius: float = 0.01 # m

    water_metal_heat_transfer_coefficient: float = 0.0 # W/m^2/K
    metal_air_heat_transfer_coefficient: float = 0.0 # W/m^2/K
    metal_product_heat_transfer_coefficient: float = 0.0 # W/m^2/K

class discretization:
    nodes: int = 20
    time_step: float = 0.005 # s/step

class product:
    specific_heat_capacity: float = 3890.0 # J/kg/K
    mass: float = 100.0 # kg

class water_node:
    surface_metal: float = pi * heat_exchanger.water_pipe_radius * 2 * heat_exchanger.water_pipe_length / discretization.nodes # m^2
    volume: float = pi * heat_exchanger.water_pipe_radius ** 2 * heat_exchanger.water_pipe_length / discretization.nodes # m^3
    def mass(temperature_degC) -> float:
        return Water.rho(temperature_degC) * pi * heat_exchanger.water_pipe_radius**2 * heat_exchanger.water_pipe_length / discretization.nodes

class metal_node:
    surface_product: float = heat_exchanger.product_surface / discretization.nodes
    surface_air: float = heat_exchanger.air_surface / discretization.nodes
    volume: float = heat_exchanger.metal_weight / steel.density / discretization.nodes
    mass: float = heat_exchanger.metal_weight / discretization.nodes


# Conduction heat rate formula
def heat_rate_conduction_boundary(origin_temperature: float, destiny_temperature: float, transfer_surface: float, transfer_coefficient: float) -> float:
    return transfer_surface * transfer_coefficient * (origin_temperature - destiny_temperature)

# Temperature change formula
def next_temperature(previous_temperature: float, heat_rate: float, specific_heat_capacity: float, mass: float) -> float:
    return previous_temperature + heat_rate / (specific_heat_capacity * mass)


input_water_temperature = 70 # degC
air_temperature = 25.0 # degC
starting_water_temperature = 25.0 # degC
starting_metal_temperature = 25.0 # degC
starting_product_temperature = 25.0 # degC
water_flow_rate = 0.1 # kg/s
simulation_time = 20 # s


class Simulador01Data:
    nodes_wm_h = np.zeros(discretization.nodes, dtype=np.float64) # W
    nodes_ww_h = np.zeros(discretization.nodes, dtype=np.float64) # W
    nodes_ma_h = np.zeros(discretization.nodes, dtype=np.float64) # W
    nodes_mp_h = np.zeros(discretization.nodes, dtype=np.float64) # W

    nodes_w_t  = np.full(discretization.nodes + 1, starting_water_temperature, dtype=float) # degC
    nodes_m_t  = np.full(discretization.nodes, starting_metal_temperature, dtype=float) # degC

    p_t = np.full(1, starting_product_temperature, dtype=float) # degC
    
    water_flow_rate = water_flow_rate # kg/s
    input_water_temperature = input_water_temperature # degC
    nodes_w_t[0] = input_water_temperature
    air_temperature = air_temperature # degC

class Simulador01DataShapes:
    nodes_wm_h = Simulador01Data.nodes_wm_h.shape
    nodes_ww_h = Simulador01Data.nodes_ww_h.shape
    nodes_ma_h = Simulador01Data.nodes_ma_h.shape
    nodes_mp_h = Simulador01Data.nodes_mp_h.shape

    nodes_w_t = Simulador01Data.nodes_w_t.shape
    nodes_m_t = Simulador01Data.nodes_m_t.shape

    p_t = Simulador01Data.p_t.shape

class Simulador01DataSizes:
    nodes_wm_h = Simulador01Data.nodes_wm_h.nbytes
    nodes_mp_h = Simulador01Data.nodes_mp_h.nbytes
    nodes_ww_h = Simulador01Data.nodes_ww_h.nbytes
    nodes_ma_h = Simulador01Data.nodes_ma_h.nbytes

    nodes_w_t = Simulador01Data.nodes_w_t.nbytes
    nodes_m_t = Simulador01Data.nodes_m_t.nbytes

    p_t = Simulador01Data.p_t.nbytes


@ray.remote
def heat_wm(i):
    multiprocessing.shared_memory.SharedMemory(name='nodes_wm_h', create=True, size=Simulador01DataSizes.nodes_wm_h)
    
    ref_nodes_w_t = multiprocessing.shared_memory.SharedMemory(name='nodes_w_t', size=Simulador01DataSizes.nodes_w_t)
    nodes_w_t = np.ndarray(Simulador01DataShapes.nodes_w_t, dtype=np.float64, buffer=ref_nodes_w_t.buf)
    ref_nodes_m_t = multiprocessing.shared_memory.SharedMemory(name='nodes_m_t', size=Simulador01DataSizes.nodes_m_t)
    nodes_m_t = np.ndarray(Simulador01DataShapes.nodes_m_t, dtype=np.float64, buffer=ref_nodes_m_t.buf)
    ref_nodes_wm_h = multiprocessing.shared_memory.SharedMemory(name='nodes_wm_h', size=Simulador01DataSizes.nodes_wm_h)
    nodes_wm_h = np.ndarray(Simulador01DataShapes.nodes_m_t, dtype=np.float64, buffer=ref_nodes_wm_h.buf)
    nodes_wm_h[i] = heat_rate_conduction_boundary(nodes_w_t[i], nodes_m_t[i], water_node.surface_metal, heat_exchanger.water_metal_heat_transfer_coefficient)


@ray.remote
def heat_ww(i):
    ref_nodes_w_t = multiprocessing.shared_memory.SharedMemory(name='nodes_w_t', size=Simulador01DataSizes.nodes_w_t)
    nodes_w_t = np.ndarray(Simulador01DataShapes.nodes_w_t, dtype=np.float64, buffer=ref_nodes_w_t.buf)
    
        
            # nodes_ww_h[i] = water_flow_rate * water.cp(nodes_w_t[i+1]) * (self.nodes_w_t[i] - self.nodes_w_t[i+1]) * discretization.time_step
            # self.nodes_ma_h[i] = heat_rate_conduction_boundary(self.nodes_m_t[i], self.air_temperature, metal_node.surface_air, heat_exchanger.metal_air_heat_transfer_coefficient)
            # self.nodes_mp_h[i] = heat_rate_conduction_boundary(self.nodes_m_t[i], self.p_t, metal_node.surface_product, heat_exchanger.metal_product_heat_transfer_coefficient)

            # self.p_t = self.p_t + np.sum(self.nodes_mp_h) * discretization.time_step / (product.specific_heat_capacity * product.mass)
            # self.nodes_m_t[i] = next_temperature(self.nodes_m_t[i], self.nodes_wm_h[i] - self.nodes_ma_h[i] - self.nodes_mp_h[i], steel.specific_heat_capacity, heat_exchanger.metal_weight)
            # self.nodes_w_t[i+1] = next_temperature(self.nodes_w_t[i+1], (- self.nodes_wm_h[i] + self.nodes_ww_h[i]), water.cp(self.nodes_w_t[i+1]), water.rho(self.nodes_w_t[i+1]) * water_node.volume)


class Simulator01 (Simulador01Data):
    def __init__(self):
        
        super().__init__()
        

    def step(self):

        for i in range(0, discretization.nodes):
            self.nodes_wm_h[i] = heat_rate_conduction_boundary(self.nodes_w_t[i], self.nodes_m_t[i], water_node.surface_metal, heat_exchanger.water_metal_heat_transfer_coefficient)
            self.nodes_ww_h[i] = self.water_flow_rate * Water.cp(self.nodes_w_t[i+1]) * (self.nodes_w_t[i] - self.nodes_w_t[i+1]) * discretization.time_step
            self.nodes_ma_h[i] = heat_rate_conduction_boundary(self.nodes_m_t[i], self.air_temperature, metal_node.surface_air, heat_exchanger.metal_air_heat_transfer_coefficient)
            self.nodes_mp_h[i] = heat_rate_conduction_boundary(self.nodes_m_t[i], self.p_t, metal_node.surface_product, heat_exchanger.metal_product_heat_transfer_coefficient)

            self.p_t = self.p_t + np.sum(self.nodes_mp_h) * discretization.time_step / (product.specific_heat_capacity * product.mass)
            self.nodes_m_t[i] = next_temperature(self.nodes_m_t[i], self.nodes_wm_h[i] - self.nodes_ma_h[i] - self.nodes_mp_h[i], steel.specific_heat_capacity, heat_exchanger.metal_weight)
            self.nodes_w_t[i+1] = next_temperature(self.nodes_w_t[i+1], (- self.nodes_wm_h[i] + self.nodes_ww_h[i]), Water.cp(self.nodes_w_t[i+1]), Water.rho(self.nodes_w_t[i+1]) * water_node.volume)


# Conduction heat rate formula
def heat_rate_conduction_boundary(origin_temperature: float, destiny_temperature: float, transfer_surface: float, transfer_coefficient: float) -> float:
    return transfer_surface * transfer_coefficient * (origin_temperature - destiny_temperature)

# Temperature change formula
def next_temperature(previous_temperature: float, heat_rate: float, specific_heat_capacity: float, mass: float) -> float:
    return previous_temperature + heat_rate / (specific_heat_capacity * mass)


@ray.remote
def calculate_heats_by_index(node_w_t: float, next_node_w_t: float, node_m_t: float, p_t: float):
    return water_node.surface_metal * heat_exchanger.water_metal_heat_transfer_coefficient * (node_w_t - node_m_t), \
        water_flow_rate * Water.cp(next_node_w_t) * (node_w_t - next_node_w_t) * discretization.time_step, \
        metal_node.surface_air * heat_exchanger.metal_air_heat_transfer_coefficient * (node_m_t - air_temperature), \
        metal_node.surface_product * heat_exchanger.metal_product_heat_transfer_coefficient * (node_m_t - p_t)

@ray.remote
def calculate_next_temperatures_by_index(node_w_t, node_m_t, node_wm_h, node_ww_h, node_ma_h, node_mp_h):
    return node_m_t + node_wm_h - node_ma_h - node_mp_h / (steel.specific_heat_capacity * metal_node.mass), \
        node_w_t + (- node_wm_h + node_ww_h) / (Water.cp(node_w_t) * water_node.volume * Water.rho(node_w_t))


# Configurable variables

node_to_measure = 10

@dataclass
class data:
    product_temperature = []
    metal_temperature = []
    water_temperature = []
    time = []


simulator = Simulator01()

t1 = time.time()
t2 = time.time()

# Simulation steps calculated
print(f'Simulation steps: {int(simulation_time/discretization.time_step)}')
# Simulation time calculated
print(f'Simulation time: {simulation_time}s')

data.time.append(0 * discretization.time_step)
data.product_temperature.append(simulator.p_t)
data.metal_temperature.append(simulator.nodes_m_t[node_to_measure])
data.water_temperature.append(simulator.nodes_w_t[node_to_measure])

profiling_data_mean_ns = [0,0,0,0,0,0,0]
profiling_data_mean_us = [0,0,0,0,0,0,0]
profiling_data_accumulated_ns = [0,0,0,0,0,0,0]
profiling_data_accumulated_ms = [0,0,0,0,0,0,0]
profiling_data_accumulated_s = [0,0,0,0,0,0,0]
profiling_data_points = [0,0,0,0,0,0,0]

t = time.time()

save_data = {
    'temperature': []
}

for i in range(0, int(simulation_time/discretization.time_step)):
    futures = []
    for i in range(0, discretization.nodes):
        futures.append(calculate_heats_by_index.remote(simulator.nodes_w_t[i], simulator.nodes_w_t[i+1], simulator.nodes_m_t[i], simulator.p_t))
    
    results = ray.get(futures)
    futures = []
    for i in range(0, discretization.nodes):
        futures.append(calculate_next_temperatures_by_index.remote(simulator.nodes_w_t[i+1], simulator.nodes_m_t[i], results[i][0], results[i][1], results[i][2], results[i][3]))

    results = ray.get(futures)

    save_data['temperature'].append(results[10][0])

print(save_data['temperature'])

for i in range(0, len(profiling_data_mean_ns)):

    profiling_data_accumulated_ms[i] = profiling_data_accumulated_ns[i] / 1000000
    profiling_data_accumulated_s[i] = profiling_data_accumulated_ns[i] / 1000000000

    if profiling_data_points[i] > 0:
        profiling_data_mean_ns[i] = profiling_data_accumulated_ns[i] / profiling_data_points[i]
        profiling_data_mean_us[i] = profiling_data_mean_ns[i] / 1000


# Print profiling data for each point

for i in range(0, len(profiling_data_mean_ns)):
    if profiling_data_points[i] > 0:
        print(f"Profiling data for step {i}  -  Accumulated time: {profiling_data_accumulated_ms[i]:.2f} ms  -  Mean: {profiling_data_mean_us[i]:.4f} us")

print()


figure = plt.figure()
axes = figure.subplots(1,1)
product_temperature_line, = axes.plot(data.time, data.product_temperature, label="Product temperature")
metal_temperature_line, = axes.plot(data.time, data.metal_temperature, label="Metal temperature")
water_temperature_line, = axes.plot(data.time, data.water_temperature, label="Water temperature")

axes.set_xlabel("Time")
axes.set_ylabel("Temperature (degC)")
legend = axes.legend()