import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image

from mona.nn.model2 import Model2
from mona.text import get_lexicon
from mona.nn import predict as predict_net
from mona.datagen.pre_process import pre_process
from mona.config import config

device = "cuda" if torch.cuda.is_available() else "cpu"
lexicon = get_lexicon(config["model_type"])

net = Model2(lexicon.lexicon_size(), in_channels=1).to(device)

name = "model_acc100-epoch197.pt"
net.load_state_dict(torch.load(f"models/{name}", map_location=device, weights_only=True))
net.eval()


def predict_image(image_name):
    im = Image.open(f"data/test/{image_name}")
    im = pre_process(im)
    im.save("test.png")

    tensor = T.ToTensor()(im)
    tensor.unsqueeze_(0)
    tensor = tensor.to(device)

    with torch.no_grad():
        result = predict_net(net, tensor, lexicon)
    return result[0]


names = [
    "test1.png",
    "test2.png",
    "test3.png",
    "test4.png",
    "test5.png",
]

for name in names:
    result = predict_image(name)
    print(f"{name}: {result}")
