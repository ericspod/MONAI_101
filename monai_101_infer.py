import sys
from pathlib import Path

import torch
from monai.networks.nets import densenet121
from monai.transforms import LoadImage, ScaleIntensity, Compose
from monai.networks import eval_mode

input_dir = Path(sys.argv[1])

if not input_dir.is_dir():
    raise IOError("First argument must be a directory path containing input JPEG images.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = densenet121(spatial_dims=2, in_channels=1, out_channels=6).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))

loader = Compose([LoadImage(ensure_channel_first=True), ScaleIntensity()])

class_names = ("AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT")

with eval_mode(model):
    for img in sorted(input_dir.glob("*.jp*g", case_sensitive=False)):
        img_t = loader(img)
        prob = model(img_t[None].to(device))
        pred = class_names[prob.argmax()]
        print(f"Class prediction for {img} is {pred}.")
