import logging
import os
from pathlib import Path
import sys
import tempfile
import torch

from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.engines import SupervisedTrainer
from monai.handlers import StatsHandler
from monai.networks import eval_mode
from monai.networks.nets import densenet121
from monai.transforms import LoadImaged, ScaleIntensityd, Compose

print_config()

directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory

transform = Compose([LoadImaged(keys="image", ensure_channel_first=True), ScaleIntensityd(keys="image")])

dataset = MedNISTDataset(root_dir=root_dir, transform=transform, section="training", download=True, progress=False)

model = densenet121(spatial_dims=2, in_channels=1, out_channels=6).to("cuda:0")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
trainer = SupervisedTrainer(
    device=torch.device("cuda:0"),
    max_epochs=5,
    network=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-5),
    loss_function=torch.nn.CrossEntropyLoss(),
    train_data_loader=DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4),
    train_handlers=StatsHandler(),
)

trainer.run()

torch.save(model.state_dict(), "model.pt")

class_names = sorted(Path(root_dir, "MedNIST").glob("*/"))
testdata = MedNISTDataset(root_dir=root_dir, transform=transform, section="test", progress=False)

with eval_mode(model):
    for item in DataLoader(testdata[:10], batch_size=1, num_workers=0):
        prob = model(item["image"].to("cuda:0"))
        pred = class_names[prob.argmax()]
        gt = item["class_name"][0]
        print(f"Class prediction is {pred}. Ground truth: {gt}")
