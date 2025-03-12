from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 8
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4
_C.SYSTEM.multi_processing = 20

# gnn choices
_C.network = CN()
_C.network.layers = 5
_C.network.layer_type = "graph"
_C.network.pre_mp = 0
_C.network.post_mp = 1
_C.network.activation = "relu"
_C.network.hidden_neurons = 16
_C.network.batch_norm = True
# gnn choices -- model creator related
_C.network.structure = "basic"
_C.network.use_residual = False
_C.network.output_dim = 2

# Training options
_C.train = CN()
_C.train.epochs = 400
_C.train.learning_rate = 0.001
_C.train.batch_size = 8

# Experiment logging
_C.experiment = CN()
_C.experiment.track = True
_C.experiment.ex_name = "GNN4AGVs_clf"  # "GNN4AGVs_clf" "GNN4AGVs_reg"
_C.experiment.number_of_layers = False
_C.experiment.model_optimisation = True
_C.experiment.aggregator_choice = False

_C.repetition = 3

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

global_cfg = _C