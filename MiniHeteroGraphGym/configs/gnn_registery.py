# The register is only about the container setups
# And we only offer the registery of layer choices and message passing structures
# And this is from GraphGym inside pyG

from typing import Any, Dict, Union
from collections.abc import Callable

layer_dict: Dict[str, Any] = {}
mp_module_dict: Dict[str, Any] = {}
pre_mp_module_dict: Dict[str, Any] = {}

def register_base(mapping: Dict[str, Any], key: str,
                  module: Any = None) -> Union[None, Callable]:
    r"""Base function for registering a module in GraphGym.

    Args:
        mapping (dict): :python:`Python` dictionary to register the module.
            hosting all the registered modules
        key (str): The name of the module.
        module (any, optional): The module. If set to :obj:`None`, will return
            a decorator to register a module.
    """
    if module is not None:
        if key in mapping:
            print(f"Warning: Module with '{key}' already defined. It will be overridden.")
            # raise KeyError(f"Module with '{key}' already defined, and then will be modified/overrided")
        mapping[key] = module
        return

    # Other-wise, use it as a decorator:
    def bounded_register(module):
        register_base(mapping, key, module)
        return module

    return bounded_register


def register_layer(key: str, module:Any = None):
    print()
    return register_base(layer_dict, key, module)

def register_mp_module(key: str, module:Any = None):
    return register_base(mp_module_dict, key, module)

def register_pre_mp_module(key: str, module:Any = None):
    return register_base(pre_mp_module_dict, key, module)