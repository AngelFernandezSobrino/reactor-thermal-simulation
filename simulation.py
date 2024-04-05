# Thermal simulation of heat exchanger for steady water heating

import time
import signal
from typing import List, Tuple, cast

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from simulation_tools import (
    SimulationParameters,
    ModelParameters,
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


water_flow_rate = 0.001  # m3/s

# Configurable variables

data_time: List[np.float128] = []
data_temperature_product: List[np.float128] = []
data_temperature_metal: List[np.float128] = []
data_temperature_water_output: List[np.float128] = []
data_heat_wm: List[np.float128] = []
data_heat_ww: List[np.float128] = []
data_heat_ma: List[np.float128] = []
data_heat_mp: List[np.float128] = []

data_time_step: List[np.float128] = []

actual = SimulationData()
previous = SimulationData()

data_time.append(np.float128(0))
data_temperature_product.append(np.float128(actual.p_t))
data_temperature_metal.append(np.float128(actual.m_t))
data_temperature_water_output.append(np.float128(actual.w_t))

data_heat_wm.append(np.float128(actual.wm_h))
data_heat_ma.append(np.float128(actual.ma_h))
data_heat_mp.append(np.float128(actual.mp_h))

data_time_step.append(np.float128(SimulationParameters.Discretization.time_step))


inner_pipe_convective_heat_transfer = ForcedConvectionInnerPipe(
    Reactor.HeatExchanger.water_pipe_radius, Water()
)

natural_convection_vertical_surface = NaturalConvectionVerticalSurface(
    Reactor.HeatExchanger.height, SimulationParameters.Boundaries.air_temperature
)


def update_heat_rates_local(actual: SimulationData, previous: SimulationData):

    actual.pipe_alpha = inner_pipe_convective_heat_transfer.convection_coefficient(
        float(previous.w_t), water_flow_rate
    )

    actual.air_alpha = natural_convection_vertical_surface.convection_coefficient(
        float(previous.m_t)
    )

    actual.wm_h = (
        ModelParameters.Water.surface_metal
        * actual.pipe_alpha
        * (previous.w_t - previous.m_t)
    )

    actual.ma_h = (
        ModelParameters.Metal.surface_air
        * actual.air_alpha
        * (previous.m_t - SimulationParameters.Boundaries.air_temperature)
    )
    actual.mp_h = (
        ModelParameters.Metal.surface_product
        * Reactor.HeatExchanger.metal_product_heat_transfer_coefficient
        * (previous.m_t - previous.p_t)
    )


def update_temperatures_local(actual: SimulationData, previous: SimulationData):

    actual.m_t = previous.m_t + (
        actual.wm_h - actual.ma_h - actual.mp_h
    ) * SimulationParameters.Discretization.time_step / (
        Steel.specific_heat_capacity * ModelParameters.Metal.mass
    )

    actual.w_t = previous.w_t - (
        actual.wm_h
    ) * SimulationParameters.Discretization.time_step / (
        Water.cp(float(previous.w_t)) * ModelParameters.Water.mass(float(previous.w_t))
    )

    actual.w_t = (
        SimulationParameters.Boundaries.input_water_temperature
        * water_flow_rate
        * SimulationParameters.Discretization.time_step
        + actual.w_t
        * (
            ModelParameters.Water.volume
            - water_flow_rate * SimulationParameters.Discretization.time_step
        )
    ) / ModelParameters.Water.volume



# Simulation time calculated
print(f"Simulation time: {SimulationParameters.time}s")
print(f"Initial time step: {SimulationParameters.Discretization.time_step}s")
print(
    f"Initial simulation steps: {int(SimulationParameters.time/SimulationParameters.Discretization.time_step)}"
)


end_simulation = False


def SIGINT_handler(signum, frame):
    global end_simulation
    end_simulation = True


signal.signal(signal.SIGINT, SIGINT_handler)


t = time.time()

simulation_running = True

actual_step: int = 1
actual_time = np.float128(0)

actual_data_index: int = 0
previous_data_index: int = 1
counter_inestability_reduction = 0

while simulation_running and not end_simulation:

    valid_step = False

    while not valid_step:

        # print_string = f"Time: {actual_time:.2f}  "

        update_heat_rates_local(actual, previous)
        update_temperatures_local(actual, previous)

        actual.p_t = previous.p_t + actual.mp_h * SimulationParameters.Discretization.time_step / (Product.specific_heat_capacity * Product.mass)  # type: ignore

        # print_string += str(actual)

        # if actual.m_t > SimulationParameters.Boundaries.input_water_temperature:
        #     raise ValueError("Metal temperature is higher than water temperature")

        valid_step = True

        # time_step_reduction = False

        # wm_h_delta_t = abs(actual.wm_h) / abs(previous.wm_h + 1)

        # if wm_h_delta_t > 1.015 or wm_h_delta_t < 0.995:
        #     counter_inestability_reduction += 1

        # m_delta_t = abs(actual.m_t - actual.m_t)
        # w_delta_t = abs(actual.w_t - previous.w_t)

        # if (
        #     m_delta_t > 2
        #     or w_delta_t > 2
        #     or actual.w_t > SimulationParameters.Boundaries.input_water_temperature
        #     or counter_inestability_reduction > 4
        # ):
        #     counter_inestability_reduction = 0
        #     if SimulationParameters.Discretization.time_step < 0.0001:
        #         raise ValueError("Time step is too small")
                
        #     time_step_reduction = True

        # if time_step_reduction:
        #     SimulationParameters.Discretization.time_step = (
        #         SimulationParameters.Discretization.time_step / 1.3
        #     )
        #     print_string += (
        #         f" Reducing time step {SimulationParameters.Discretization.time_step}"
        #     )
        #     print(print_string)
        # else:
        #     valid_step = True

    data_time.append(np.float128(actual_time))
    data_heat_wm.append(actual.wm_h)
    data_heat_ma.append(actual.ma_h)
    data_heat_mp.append(actual.mp_h)

    data_temperature_product.append(actual.p_t)
    data_temperature_metal.append(actual.m_t)
    data_temperature_water_output.append(actual.w_t)
    data_time_step.append(SimulationParameters.Discretization.time_step)

    actual_time += np.float128(SimulationParameters.Discretization.time_step)
    actual_step += 1

    if actual_time > SimulationParameters.time:
        # print(f"Simulation time reached: {actual_time} > {SimulationParameters.time}")
        simulation_running = False

    # if actual_step % 10 == 0:
    #     time_step_increase = True

    #     wm_h_delta_t = abs(actual.wm_h) / abs(previous.wm_h)

    #     m_delta_t = abs(actual.m_t - actual.m_t)
    #     w_delta_t = abs(actual.w_t - previous.w_t)

    #     if (
    #         m_delta_t > 2
    #         or w_delta_t > 2
    #         or wm_h_delta_t > 1.005
    #         or wm_h_delta_t < 0.995
    #     ):
    #         time_step_increase = False

    #     if time_step_increase:
    #         print_string += (
    #             f" Increasing time step {SimulationParameters.Discretization.time_step}"
    #         )
    #         SimulationParameters.Discretization.time_step = (
    #             SimulationParameters.Discretization.time_step * 1.05
    #         )

    # print_string += (
    #     f"  Counter inestability reduction: {counter_inestability_reduction}"
    # )

    # Swap previous and actual data
    previous, actual = actual, previous
    # print(print_string)


print()
print("-----------------------------------")
print()

print(f"Elapsed time: {time.time() - t}s")


# Save csv file with columns for data_time, data_temperature_product, data_temperature_metal, data_temperature_water_output, data_heat_wm, data_heat_ma, data_heat_mp,

import csv

with open("simulation_results.csv", mode="w") as file:
    writer = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(
        [
            "Time [s]",
            "Product temperature [degC]",
            "Metal temperature [degC]",
            "Water temperature [degC]",
            "Water-Metal heat rate [W]",
            "Metal-Air heat rate [W]",
            "Metal-Product heat rate [W]",
        ]
    )
    for i in range(len(data_time)):
        writer.writerow(
            [
                data_time[i],
                data_temperature_product[i],
                data_temperature_metal[i],
                data_temperature_water_output[i],
                data_heat_wm[i],
                data_heat_ma[i],
                data_heat_mp[i]
            ]
        )



figure = plt.figure()
(
    axes1,
    axes2,
    axes3,
) = cast(List[mpl.axes.Axes], figure.subplots(3, 1, sharex=True))
(product_temperature_line,) = axes1.plot(
    data_time, data_temperature_product, label="Product temperature"
)
(metal_temperature_line,) = axes1.plot(
    data_time, data_temperature_metal, label="Metal temperature"
)
(water_temperature_line,) = axes1.plot(
    data_time, data_temperature_water_output, label="Water temperature"
)


axes1.set_xlabel("Time")
axes1.set_ylabel("Temperature (degC)")
legend = axes1.legend()

(water_metal_heat_line,) = axes2.plot(data_time, data_heat_wm, label="Water-Metal heat")
(metal_air_heat_line,) = axes2.plot(data_time, data_heat_ma, label="Water-Air heat")
(metal_product_heat_line,) = axes2.plot(data_time, data_heat_mp, label="Metal-Product heat")

axes2.set_xlabel("Time")
axes2.set_ylabel("Heat rate (W)")
legend = axes2.legend()

(time_step_plot,) = axes3.plot(data_time, data_time_step, label="Time step (s)")

axes3.set_xlabel("Time")
axes3.set_ylabel("Time step (s)")
legend = axes3.legend()


plt.show()

