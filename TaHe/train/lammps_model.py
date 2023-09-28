from NN_LIP.lammps.Meam_density import RadialDensity, AngularDensity, MeaMDensityPart
from NN_LIP.lammps.PES import ElementNNModel, PESModel
import numpy as np
import torch
from collections import OrderedDict



model_path = "./params1/ckpt/model_999_0.00465.pt"
static_dict = torch.load(model_path, map_location='cpu')
print(static_dict.keys())
outpath = "."
ele_map = {"Ta": 1, "He": 0}
eatom = np.array([0.00168913, -2.24248703])
cutoff = static_dict["density.angular_filter.cutoff"]
nipsin = static_dict["density.angular_filter.nipsin"][0]
radial_rs = static_dict["density.radial_filter.rs"]
radial_inta = static_dict["density.radial_filter.inta"]
angular_rs = static_dict["density.angular_filter.rs"]
angular_inta = static_dict["density.angular_filter.inta"]
outputneuron = 1
atomtype = ["He", "Ta"]
nl = [80, 64, 64, 32]
actfunc = "Tanh"
filename = "TaHe_{systype}.pt"

radial_filter = RadialDensity(radial_rs, radial_inta, cutoff)
angular_filter = AngularDensity(angular_rs, angular_inta, cutoff, nipsin)
mdescrib = MeaMDensityPart(radial_filter, angular_filter)
nnmod = ElementNNModel(outputneuron, atomtype, nl, actfunc, bias=False)
model = PESModel(mdescrib, nnmod)

new_state_dict = OrderedDict()
for k, v in static_dict.items():
    if "prop_ceff" in k:
        continue
    name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
scripted_pes=torch.jit.script(model)
for params in scripted_pes.parameters():
    params.requires_grad=False
scripted_pes.save(f"{outpath}/{filename.format(systype='float')}")
scripted_pes.to(torch.double)
scripted_pes.save(f"{outpath}/{filename.format(systype='double')}")
print(scripted_pes.get_parameter)