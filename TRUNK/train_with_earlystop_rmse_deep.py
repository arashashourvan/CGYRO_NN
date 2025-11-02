#!/usr/bin/env python3
# Wrapper that reuses the improved trainer but swaps the model to Phi2FluxDeep.
import sys, types

# 1) Import the deep model
from phi2flux_deep import Phi2FluxDeep as Phi2Flux  # alias so the trainer sees 'Phi2Flux'

# 2) Monkey-patch a shim module so the trainer can import build_datasets from your original file
#    (this just re-exports build_datasets from phi2flux_3species)
import phi2flux_3species as _base
shim = types.ModuleType("phi2flux_3species")
shim.Phi2Flux = Phi2Flux                   # deep model
shim.build_datasets = _base.build_datasets # your existing dataset builder
sys.modules["phi2flux_3species"] = shim

# 3) Now run the improved trainer (it will import from the shim above)
import train_with_earlystop_rmse  # noqa: F401  (executes main on import)

