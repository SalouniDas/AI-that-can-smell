import numpy as np


def simulate_sensor_reading(molecule_list, noise_level=0.05):
    """
    Simulates a reading from an 8-channel gas sensor array based on
    crude chemical heuristics of the input molecules.
    """
    # 8 simulated sensor channels
    reading = np.zeros(8)

    # Crude mapping of molecule properties to sensor channels
    properties = {
        "2-furfurylthiol": [0.8, 0.1, 0.9, 0.0, 0.2, 0.1, 0.6, 0.3],  # Sulfur/roasted
        "guaiacol": [0.6, 0.7, 0.2, 0.5, 0.8, 0.1, 0.3, 0.4],  # Phenolic/smoke
        "2-ethylpyrazine": [0.7, 0.2, 0.6, 0.1, 0.4, 0.8, 0.2, 0.5],  # Nutty/roasted
        "vanillin": [0.2, 0.9, 0.1, 0.8, 0.7, 0.3, 0.1, 0.2],  # Sweet/vanilla
        "ethyl vanillin": [0.3, 0.9, 0.2, 0.9, 0.8, 0.2, 0.1, 0.3],  # Sweet/vanilla
        "menthol": [0.1, 0.4, 0.1, 0.2, 0.3, 0.9, 0.8, 0.1],  # Cooling/mint
        "menthone": [0.2, 0.5, 0.1, 0.3, 0.4, 0.8, 0.7, 0.2],  # Mint
        "limonene": [0.9, 0.1, 0.1, 0.1, 0.2, 0.5, 0.4, 0.8],  # Citrus/hydrocarbon
        "citral": [0.8, 0.3, 0.1, 0.4, 0.3, 0.6, 0.5, 0.7],  # Citrus/aldehyde
        "isoamyl acetate": [0.4, 0.8, 0.1, 0.7, 0.5, 0.2, 0.2, 0.4],  # Fruity/banana
        "syringol": [0.5, 0.6, 0.3, 0.6, 0.9, 0.1, 0.2, 0.3],  # Smoke/wood
        "cresol": [0.7, 0.5, 0.4, 0.4, 0.8, 0.2, 0.4, 0.2]  # Phenolic/tar
    }

    # Aggregate signals from all molecules in the sample
    for mol in molecule_list:
        if mol in properties:
            reading += np.array(properties[mol])

    # Normalize the reading to prevent saturation, mimicking sensor limits
    if np.max(reading) > 0:
        reading = reading / np.max(reading)

    # Add simulated environmental noise and humidity fluctuations
    noise = np.random.normal(0, noise_level, 8)
    reading = np.clip(reading + noise, 0, 1)

    return reading