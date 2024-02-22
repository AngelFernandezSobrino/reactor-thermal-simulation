import math
import numpy as np

import CoolProp.CoolProp  # type: ignore


class Wall:
    def __init__(self, base: float, height: float) -> None:
        self.base = base
        self.height = height
        self.area = base * height


class NaturalConvectionVerticalSurface:

    def __init__(self, characteristic_lenght: float, air_temperature: float) -> None:

        self.temperature_linspace_min: float = 25
        self.temperature_linspace_max: float = 90
        self.temperature_linspace_ammount: int = 86
        self.temperature_linspace_step: float = (
            self.temperature_linspace_max - self.temperature_linspace_min
        ) / (self.temperature_linspace_ammount - 1)

        self.temperature_linspace: float = np.linspace(self.temperature_linspace_min, self.temperature_linspace_max, self.temperature_linspace_ammount)  # type: ignore

        self.convection_coeficient_lookuptable = np.zeros(
            shape=(self.temperature_linspace_ammount)
        )

        self.characteristic_lenght = characteristic_lenght
        self.air_temperature = air_temperature + 273.13  # Kelvin

        self.init_lookup_tables()

    def init_lookup_tables(self):
        for temperature_index, temperature in enumerate(self.temperature_linspace):
            self.convection_coeficient_lookuptable[temperature_index] = (
                self.calc_convection_coeficient(temperature)
            )

    def calc_convection_coeficient(self, temperature_degC) -> float:
        film_temperature = (
            temperature_degC + 273.13 + self.air_temperature
        ) / 2  # Kelvin

        beta = CoolProp.CoolProp.PropsSI(
            "isobaric_expansion_coefficient", "T", film_temperature, "P", 101325, "Air"
        )

        Prandtl = CoolProp.CoolProp.PropsSI(
            "Prandtl", "T", film_temperature, "P", 101325, "Air"
        )

        air_dynamic_viscosity = CoolProp.CoolProp.PropsSI(
            "viscosity", "T", film_temperature, "P", 101325, "Air"
        )

        air_density = CoolProp.CoolProp.PropsSI(
            "D", "T", film_temperature, "P", 101325, "Air"
        )

        air_thermal_conductivity = CoolProp.CoolProp.PropsSI(
            "conductivity", "T", film_temperature, "P", 101325, "Air"
        )

        Grasshoff = (
            9.81
            * self.characteristic_lenght**3
            * beta
            * (temperature_degC + 273.13 - self.air_temperature)
            / ((air_dynamic_viscosity / air_density) ** 2)
        )

        Rayleigh = Prandtl * Grasshoff

        Nusselt = 0
        if Rayleigh < 10e9:
            Nusselt = 0.68 + (0.67 * Rayleigh ** (1 / 4)) / (
                1 + (0.492 / Prandtl) ** (9 / 16)
            ) ** (4 / 9)
        else:
            Nusselt = (
                0.825
                + (0.387 * Rayleigh ** (1 / 6))
                / (1 + (0.492 / Prandtl) ** (9 / 16)) ** (8 / 27)
            ) ** 2

        convective_heat_transfer = (
            Nusselt
            * air_thermal_conductivity
            / self.characteristic_lenght
            * (temperature_degC + 273.13 - self.air_temperature)
        )

        radiative_heat_transfer = (
            5.67e-8
            * ((temperature_degC + 273.13) ** 4 - (self.air_temperature) ** 4)
            * 0.85
        )

        return convective_heat_transfer + radiative_heat_transfer

    def convection_coefficient(self, temperature: float):
        return self.convection_coeficient_lookuptable[
            math.trunc(
                (temperature - self.temperature_linspace_min)
                / self.temperature_linspace_step
            )
        ]
