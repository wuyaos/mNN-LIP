import torch
import pickle

model_path = "./params_f/ckpt/best_model.pt"
static_dict = torch.load(model_path, map_location='cpu')

radial_rs = static_dict["density.radial_filter.rs"]
radial_inta = static_dict["density.radial_filter.inta"]
angular_rs = static_dict["density.angular_filter.rs"]
angular_inta = static_dict["density.angular_filter.inta"]


# 保存
with open("./describ_params.pkl", "wb") as f:
    pickle.dump([radial_rs, radial_inta, angular_rs, angular_inta], f)

# 读取
with open("./describ_params.pkl", "rb") as f:
    radial_rs, radial_inta, angular_rs, angular_inta = pickle.load(f)
    print(radial_rs)