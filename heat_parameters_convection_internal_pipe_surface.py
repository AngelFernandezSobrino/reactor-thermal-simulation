import math
import numpy as np

import numpy as np


class CircularPipe:
    def __init__(self, radius: float) -> None:
        self.radius = radius
        self.area = math.pi * radius**2
        self.perimeter = 2 * math.pi * radius
        self.diameter = 2 * radius


class CircularPipeInnerFlow:

    def Re(self, rho, v, D, mu):
        return rho * v * D / mu

    def Pe(self, cp, rho, v, D, k):
        return cp * rho * v * D / k

    def Pr(self, Re, Pe):
        return Re / Pe

    def Nu(self, Re, Pr):
        return 0.15 * Re**0.33 * Pr**0.43 if Re < 2300 else 0.021 * Re**0.8 * Pr**0.43


class ForcedConvectionInnerPipe:

    def __init__(self, pipe_radius: float, fluid) -> None:
        self.temperature_linspace_min: float = 5
        self.temperature_linspace_max: float = 90
        self.temperature_linspace_ammount: int = 86
        self.temperature_linspace_step: float = (
            self.temperature_linspace_max - self.temperature_linspace_min
        ) / (self.temperature_linspace_ammount - 1)

        self.mass_rate_linspace_min: float = 0.05
        self.mass_rate_linspace_max: float = 2.00
        self.mass_rate_linspace_ammount: int = 100
        self.mass_rate_linspace_step: float = (
            self.mass_rate_linspace_max - self.mass_rate_linspace_min
        ) / (self.mass_rate_linspace_ammount - 1)

        self.temperature_linspace: float = np.linspace(self.temperature_linspace_min, self.temperature_linspace_max, self.temperature_linspace_ammount)  # type: ignore
        self.mass_rate_linspace: float = np.linspace(self.mass_rate_linspace_min, self.mass_rate_linspace_max, self.mass_rate_linspace_ammount)  # type: ignore

        self.convection_coeficient_lookuptable = np.zeros(
            shape=(self.mass_rate_linspace_ammount, self.temperature_linspace_ammount), dtype=np.float128
        )

        self.radius = pipe_radius
        self.pipe = CircularPipe(pipe_radius)
        self.fluid = fluid

        self.flow = CircularPipeInnerFlow()

        self.init_lookup_tables()

    def interpolate(self, temperature_degC, flow_rate_kgs):
        pass

    def init_lookup_tables(self):
        for temperature_index, temperature in enumerate(self.temperature_linspace):
            for mass_rate_index, mass_rate in enumerate(self.mass_rate_linspace):
                # print(temperature, mass_rate)
                self.convection_coeficient_lookuptable[mass_rate_index][
                    temperature_index
                ] = self.calc_convection_coeficient(
                    temperature, mass_rate / self.pipe.area / 1000
                )

    def calc_convection_coeficient(self, temperature, velocity) -> float:
        return self.flow.Nu(
            self.flow.Re(
                self.fluid.rho(temperature),
                velocity,
                self.pipe.diameter,
                self.fluid.mu(temperature),
            ),
            self.flow.Pe(
                self.fluid.cp(temperature),
                self.fluid.rho(temperature),
                velocity,
                self.pipe.diameter,
                self.fluid.k(temperature),
            ),
        )

    def convection_coefficient(self, temperature, mass_rate):
        return self.convection_coeficient_lookuptable[
            math.trunc(
                (mass_rate - self.mass_rate_linspace_min) / self.mass_rate_linspace_step
            )
        ][
            math.trunc(
                (temperature - self.temperature_linspace_min)
                / self.temperature_linspace_step
            )
        ]
