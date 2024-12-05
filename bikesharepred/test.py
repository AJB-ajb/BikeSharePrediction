import trainer
import torch as th
import matplotlib.pyplot as plt

from config import Config
import json
from pathlib import Path

# test third derivativei
def test_third_deriv():
    # note that the computation for float32 already becomes unstable for N = 1000
    xmax = 2 * th.pi
    N = 100
    step_size = xmax / (N-1)
    xs = th.linspace(0, xmax, N)[:-1]
    ys = th.sin(xs)
    true_deriv = -th.cos(xs)

    ys_3rd = trainer.finite_diff_third_derivative(ys, step_size = step_size)
    plot = False
    if plot:
        plt.plot(xs, ys_3rd, label = "3rd derivative")
        plt.plot(xs, true_deriv, label = "True derivative")
        plt.legend()
        plt.show()

    assert ys_3rd.shape == ys.shape
    print("Difference between ys_3rd and -ys:", th.norm(ys_3rd - true_deriv))
    assert th.allclose(ys_3rd, true_deriv, atol = 1e-1)

def test_config_json():
    config = Config.default_config()
    temp_file = Path('temp_config.json')
    config.save_to_json(temp_file)
    
    loaded_config = Config.load_from_json(temp_file)

    temp_file.unlink()
    
    assert config.__dict__ == loaded_config.__dict__, "Loaded config does not match the saved config"
    print("Config JSON save and load test passed ✔")

if __name__ == '__main__':
    test_third_deriv()
    test_config_json()
    print("All tests passed ✔")

