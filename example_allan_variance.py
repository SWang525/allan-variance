import matplotlib.pyplot as plt
import numpy as np
from allan_variance import allan_variance, params_from_avar

def generate_signal(n, dt, q_white, q_walk, q_ramp, random_state=0):
    rng = np.random.RandomState(random_state)
    white = q_white * rng.randn(n) * dt ** 0.5
    walk = q_walk * dt ** 0.5 * np.cumsum(rng.randn(n))
    ramp = q_ramp * dt * np.arange(n)
    return white + walk * dt + ramp * dt

def main():
    dt = 1e-2
    num_samples = 1000000
    time = np.linspace(0, num_samples * dt, num_samples)
    q_white = 0.1
    q_walk = 0.05
    q_ramp = 0.001
    signal = generate_signal(num_samples, dt, q_white=q_white, q_walk=q_walk, q_ramp=q_ramp)
    plt.plot(time, signal)
    plt.show()

    tau, av = allan_variance(x=signal, dt=dt, min_cluster_size=1, min_cluster_count='auto',
                             n_clusters=100, input_type="increment", verbose=True)
    plt.loglog(tau, av, '.')
    plt.grid(True)
    params, av_pred = params_from_avar(tau, av, output_type='dict')
    print params


    plt.loglog(tau, av_pred)
    plt.show()

if __name__ == "__main__":
    main()