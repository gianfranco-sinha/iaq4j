import time
import random
import math
import requests

ENDPOINT = "http://localhost:8000/predict"
BASE_DELAY = 10  # seconds


def random_gas_resistance():
    """
    BME680 gas resistance behaves logarithmically.
    Generate a realistic value between 5kΩ and 500kΩ.
    """
    log_min = math.log10(5_000)
    log_max = math.log10(500_000)
    return int(10 ** random.uniform(log_min, log_max))


def generate_payload():
    temperature = round(random.uniform(0.0, 28.0), 2)
    rel_humidity = round(random.uniform(45.0, 95.0), 2)

    # Slight correlation: warm air can hold more moisture
    if temperature > 20:
        rel_humidity = min(rel_humidity, random.uniform(50, 80))

    payload = {
        "temperature": temperature,
        "rel_humidity": rel_humidity,
        "pressure": round(random.uniform(985.0, 1035.0), 2),
        "gas_resistance": random_gas_resistance()
    }

    return payload


while True:
    payload = generate_payload()

    try:
        response = requests.post(
            ENDPOINT,
            json=payload,
            timeout=5
        )

        print(f"Sent: {payload}")
        print(f"Response [{response.status_code}]: {response.text}")

    except requests.RequestException as e:
        print(f"Request failed: {e}")

    # Add jitter so multiple clients never align
    time.sleep(BASE_DELAY + random.uniform(-2, 2))
