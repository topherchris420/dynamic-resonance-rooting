import numpy as np

class BenchmarkSystems:
    @staticmethod
    def generate_lorenz_data(duration=30, dt=0.01, initial_state=[1.0, 1.0, 1.0]):
        print("Generating Lorenz data...")
        n_steps = int(duration/dt)
        xyz = np.zeros((n_steps, 3))
        xyz[0] = initial_state
        sigma, rho, beta = 10, 28, 8/3
        for i in range(n_steps-1):
            x, y, z = xyz[i]
            xyz[i+1] = [
                x + sigma * (y - x) * dt,
                y + (x * (rho - z) - y) * dt,
                z + (x * y - beta * z) * dt,
            ]
        t = np.linspace(0, duration, n_steps)
        return t, xyz

    @staticmethod
    def generate_heston_data(duration=252, dt=1/252, initial_state={'s0': 100, 'v0': 0.04}):
        """
        Generates data from the Heston model for stochastic volatility.
        """
        print("Generating Heston model data...")
        n_steps = int(duration * (1/dt))
        s = np.zeros(n_steps)
        v = np.zeros(n_steps)
        s[0] = initial_state['s0']
        v[0] = initial_state['v0']

        kappa, theta, sigma, rho = 2.0, 0.04, 0.2, -0.7

        for i in range(1, n_steps):
            w_s = np.random.normal()
            w_v = rho * w_s + np.sqrt(1 - rho**2) * np.random.normal()

            s[i] = s[i-1] * np.exp((0.05 - 0.5 * v[i-1]) * dt + np.sqrt(v[i-1] * dt) * w_s)
            v[i] = np.maximum(0, v[i-1] + kappa * (theta - v[i-1]) * dt + sigma * np.sqrt(v[i-1] * dt) * w_v)

        t = np.linspace(0, duration, n_steps)
        return t, np.vstack((s, v)).T

    @staticmethod
    def generate_fitzhugh_nagumo_data(duration=500, dt=0.1, initial_state=[0.1, 0.1]):
        """
        Generates data from the FitzHugh-Nagumo model.
        """
        print("Generating FitzHugh-Nagumo data...")
        n_steps = int(duration / dt)
        xy = np.zeros((n_steps, 2))
        xy[0] = initial_state
        a, b, c = 0.7, 0.8, 0.08

        for i in range(n_steps - 1):
            x, y = xy[i]
            xy[i + 1] = [
                x + (x - x**3 / 3 - y) * dt,
                y + c * (x + a - b * y) * dt,
            ]
        t = np.linspace(0, duration, n_steps)
        return t, xy

    @staticmethod
    def generate_rossler_data(duration=30, dt=0.01, initial_state=[1.0, 1.0, 1.0]):
        print("Generating Rössler data...")
        n_steps = int(duration/dt)
        xyz = np.zeros((n_steps, 3))
        xyz[0] = initial_state
        a, b, c = 0.2, 0.2, 5.7
        for i in range(n_steps-1):
            x, y, z = xyz[i]
            xyz[i+1] = [
                x + (-y - z) * dt,
                y + (x + a * y) * dt,
                z + (b + z * (x-c)) * dt,
            ]
        t = np.linspace(0, duration, n_steps)
        return t, xyz
