# from training.cnn import cnn13
from training.resnet import resnet10, resnet12,resnet18


class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(target_model, num_classes=2, img_size=64, num_groups=2):
        if target_model == "resnet10":
            model = resnet10(pretrained=False, num_classes=num_classes, num_groups=num_groups, img_size=img_size)
        elif target_model == "resnet12":
            model = resnet12(pretrained=False, num_classes=num_classes, num_groups=num_groups, img_size=img_size)
        elif target_model == "resnet18":
            model = resnet18(pretrained=False, num_classes=num_classes, num_groups=num_groups, img_size=img_size)

        return model
