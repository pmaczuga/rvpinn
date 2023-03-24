from __future__ import annotations
import configparser
from src.utils import get_tag_path

class Params:
    """
    This class holds the parameters defined in params.ini

    It's basically a container to make it more "object-oriented"
    and (hopefully) easier to use.
    
    For example having a config file:
    ```
    # params.ini

    [Params]
    beta = 0.2
    epochs = 3000
    ```

    We have::
        >>> params = Params()
        >>> params.beta
        0.2
        >>> params.epochs
        3000

    You can also do this:
        >>> params = Params(beta=0.5)
        >>> params.beta
        0.5
        >>> params.epochs
        3000
    """
    def __init__(self, filename="params.ini", **kwargs):
        config = configparser.ConfigParser()
        config.read(filename)
        self.epochs             = config["Params"].getint("epochs")
        self.layers             = config["Params"].getint("layers")
        self.neurons_per_layer  = config["Params"].getint("neurons_per_layer")
        self.learning_rate      = config["Params"].getfloat("learning_rate")
        self.use_best_pinn      = config["Params"].getboolean("use_best_pinn")
        self.equation           = config["Params"].get("equation")
        self.eps                = config["Params"].getfloat("eps")
        self.Xd                 = config["Params"].getfloat("Xd")
        self.compute_error      = config["Params"].getboolean("compute_error")
        self.n_points_x         = config["Params"].getint("n_points_x")
        self.n_points_error     = config["Params"].getint("n_points_error")
        self.n_test_func        = config["Params"].getint("n_test_func")
        self.atol               = config["Params"].getfloat("atol")
        self.rtol               = config["Params"].getfloat("rtol")
        self.tag                = config["Params"].get("tag")

    def save(self, filename: str):
        params = self.__dict__
        config = configparser.ConfigParser()
        config["Params"] = params
        with open(filename, 'w') as f:
            config.write(f)

    def save_by_tag(self):
        tag = self.tag
        filename = f"{get_tag_path(tag)}/params.ini"
        self.save(filename)

    @classmethod
    def load_by_tag(cls, tag, **kwargs) -> Params:
        filename = f"{get_tag_path(tag)}/params.ini"
        return cls(filename=filename, **kwargs)
