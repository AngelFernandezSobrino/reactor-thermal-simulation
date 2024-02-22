# Thermal simulation of heat exchanger for steady water heating

import time
import signal
from typing import List, Tuple, cast

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from simulation_tools import (
    NodesParameters,
    Parameters,
    Product,
    Reactor,
    SimulationData,
    Steel,
)

from water import Water
from heat_parameters_convection_internal_pipe_surface import ForcedConvectionInnerPipe
from heat_parameters_natural_convection_vectical_surface import (
    NaturalConvectionVerticalSurface,
)


water_flow_rate = 1  # kg/s

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

actual = SimulationData()
previous = SimulationData()

data_time.append(0)
data_temperature_product.append(actual.p_t)
data_temperature_metal.append(actual.nodes_m_t[node_to_measure])
data_temperature_water.append(actual.nodes_w_t[node_to_measure + 1])

data_heat_wm.append(actual.nodes_wm_h[node_to_measure])
data_heat_ww.append(actual.nodes_ww_h[node_to_measure])
data_heat_ma.append(actual.nodes_ma_h[node_to_measure])
data_heat_mp.append(actual.nodes_mp_h[node_to_measure])


inner_pipe_convective_heat_transfer = ForcedConvectionInnerPipe(
    Reactor.HeatExchanger.water_pipe_radius, Water()
)

natural_convection_vertical_surface = NaturalConvectionVerticalSurface(
    Reactor.HeatExchanger.height, Parameters.Boundaries.air_temperature
)


def update_heat_rates_local():
    for node in range(0, Parameters.Discretization.nodes):
        w_t_node = node + 1
        actual.nodes_wm_h[node] = (
            NodesParameters.Water.surface_metal
            * inner_pipe_convective_heat_transfer.convection_coefficient(
                previous.nodes_w_t[w_t_node], water_flow_rate
            )
            * (previous.nodes_w_t[w_t_node] - previous.nodes_m_t[node])
        )
        actual.nodes_ww_h[node] = (
            water_flow_rate
            * Water.cp(previous.nodes_w_t[w_t_node])
            * (-previous.nodes_w_t[w_t_node] + previous.nodes_w_t[w_t_node - 1])
        )
        actual.nodes_ma_h[node] = (
            NodesParameters.Metal.surface_air
            * natural_convection_vertical_surface.convection_coefficient(
                previous.nodes_m_t[node]
            )
        )
        actual.nodes_mp_h[node] = (
            NodesParameters.Metal.surface_product
            * Reactor.HeatExchanger.metal_product_heat_transfer_coefficient
            * (previous.nodes_m_t[node] - previous.p_t)
        )


def update_temperatures_local():
    for node in range(0, Parameters.Discretization.nodes):
        w_t_node = node + 1
        actual.nodes_m_t[node] = previous.nodes_m_t[node] + (
            actual.nodes_wm_h[node] - actual.nodes_ma_h[node] - actual.nodes_mp_h[node]
        ) * Parameters.Discretization.time_step / (
            Steel.specific_heat_capacity * NodesParameters.Metal.mass
        )
        actual.nodes_w_t[w_t_node] = previous.nodes_w_t[w_t_node] + (
            -actual.nodes_wm_h[node] + actual.nodes_ww_h[node]
        ) * Parameters.Discretization.time_step / (
            Water.cp(previous.nodes_w_t[w_t_node])
            * NodesParameters.Water.mass(previous.nodes_w_t[w_t_node])
        )


# Simulation time calculated
print(f"Simulation time: {Parameters.time}s")
print(f"Initial time step: {Parameters.Discretization.time_step}s")
print(
    f"Initial simulation steps: {int(Parameters.time/Parameters.Discretization.time_step)}"
)


end_simulation = False


def SIGINT_handler(signum, frame):
    global end_simulation
    end_simulation = True


signal.signal(signal.SIGINT, SIGINT_handler)


t = time.time()

simulation_running = True

actual_step: int = 1
actual_time: float = 0

actual_data_index: int = 0
previous_data_index: int = 1

while simulation_running and not end_simulation:
    update_heat_rates_local()
    update_temperatures_local()

    actual.p_t = previous.p_t + np.sum(actual.nodes_mp_h) * Parameters.Discretization.time_step / (Product.specific_heat_capacity * Product.mass)  # type: ignore

    time_step_reduction = False
    for node in range(0, Parameters.Discretization.nodes):
        w_t_node = node + 1
        m_delta_t = abs(actual.nodes_m_t[node] - actual.nodes_m_t[node])
        w_delta_t = abs(actual.nodes_w_t[w_t_node] - actual.nodes_w_t[w_t_node])

        if (
            m_delta_t > 2
            or w_delta_t > 2
            or actual.nodes_w_t[w_t_node] > actual.nodes_m_t[node]
        ):
            if Parameters.Discretization.time_step < 0.001:
                simulation_running = False
                break
            time_step_reduction = True

    if time_step_reduction:
        Parameters.Discretization.time_step = Parameters.Discretization.time_step / 1.3
        print("Reducing time step", Parameters.Discretization.time_step)
        continue


    data_time.append(actual_time)
    data_heat_wm.append(actual.nodes_wm_h[node_to_measure])
    data_heat_ww.append(actual.nodes_wm_h[node_to_measure])
    data_heat_ma.append(actual.nodes_wm_h[node_to_measure])
    data_heat_mp.append(actual.nodes_wm_h[node_to_measure])

    data_temperature_product.append(actual.p_t)
    data_temperature_metal.append(actual.nodes_m_t[node_to_measure])
    data_temperature_water.append(actual.nodes_w_t[node_to_measure + 1])

    actual_time += Parameters.Discretization.time_step
    actual_step += 1

    if actual_time > Parameters.time:
        simulation_running = False

    if actual_step % 10 == 0:
        time_step_increase = True
        for node in range(0, Parameters.Discretization.nodes):
            w_t_node = node + 1
            m_delta_t = abs(actual.nodes_m_t[node] - previous.nodes_m_t[node])
            w_delta_t = abs(actual.nodes_w_t[w_t_node] - previous.nodes_w_t[w_t_node])

            if m_delta_t > 2 or w_delta_t > 2:
                time_step_increase = False

        if time_step_increase:
            Parameters.Discretization.time_step = (
                Parameters.Discretization.time_step * 1.3
            )

    # Swap previous and actual data
    previous, actual = actual, previous

print()
print("-----------------------------------")
print()

print(f"Elapsed time: {time.time() - t}s")


figure = plt.figure()
(
    axes1,
    axes2,
) = cast(List[mpl.axes.Axes], figure.subplots(2, 1, sharex=True))
(product_temperature_line,) = axes1.plot(
    data_time, data_temperature_product, label="Product temperature"
)
(metal_temperature_line,) = axes1.plot(
    data_time, data_temperature_metal, label="Metal temperature"
)
(water_temperature_line,) = axes1.plot(
    data_time, data_temperature_water, label="Water temperature"
)

axes1.set_xlabel("Time")
axes1.set_ylabel("Temperature (degC)")
legend = axes1.legend()

(water_water_heat,) = axes2.plot(data_time, data_heat_ww, label="Water-Water heat")
(water_metal_heat,) = axes2.plot(data_time, data_heat_wm, label="Water-Metal heat")
(metal_air_heat,) = axes2.plot(data_time, data_heat_ma, label="Water-Air heat")
(metal_product_heat,) = axes2.plot(data_time, data_heat_mp, label="Metal-Product heat")

axes2.set_xlabel("Time")
axes2.set_ylabel("Heat rate (W)")
legend = axes2.legend()

plt.show()
