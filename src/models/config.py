from torchvision import models

from .reidentification import SiameseBasicCNN, SiameseTransfered
from .utils import resnet_grad_l4_unlock, resnet_grad_linear_unlock

models_list = {
    'siamese_resnet18_linear': SiameseTransfered(
        models.resnet18(pretrained=True),
        freeze_grad_fn=resnet_grad_linear_unlock,
        name='siamese_resnet18_linear'
    ),
    'basic_cnn':  SiameseBasicCNN()
}
