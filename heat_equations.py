
# Conduction heat rate formula
def heat_rate_conduction_boundary(origin_temperature: float, destiny_temperature: float, transfer_surface: float, transfer_coefficient: float) -> float:
    return transfer_surface * transfer_coefficient * (origin_temperature - destiny_temperature)

# Temperature change formula
def next_temperature(previous_temperature: float, heat_rate: float, specific_heat_capacity: float, mass: float) -> float:
    return previous_temperature + heat_rate / (specific_heat_capacity * mass)
