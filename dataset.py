from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image
class MyCIFAR10(CIFAR10):
    # print(123)
    def __init__(self, root, train, transform, target_transform= None, download=False):
        super(MyCIFAR10, self).__init__(root = root, train=train, transform=transform, download=download)
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, target), index

class MyCIFAR100(CIFAR100):
    # print(123)
    def __init__(self, root, train, transform, target_transform= None, download=False):
        super(MyCIFAR100, self).__init__(root = root, train=train, transform=transform, download=download)
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, target), index