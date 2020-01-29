import glob
from PIL import Image

from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def lastest_arverage_value(values, length=100):
    if len(values) < length:
        length = len(values)
    return sum(values[-length:])/length

class FacadeFolder(Dataset):
    def __init__(self, path_A, path_B, imsize, cropsize, cencrop):
        super(FacadeFolder, self).__init__()
        self.imsize = imsize
        self.cropsize = cropsize
        self.cencrop = cencrop
        self.normalize = _normalize()

        self.lst_A = sorted(glob.glob(path_A+"/*.*"))
        self.lst_B = sorted(glob.glob(path_B+"/*.*"))
        assert  len(self.lst_A) == len(self.lst_B), "Data A and Data B must have same number of images"
        print("%d datas loaded"%(len(self.lst_A)))


    def __len__(self):
        return len(self.lst_A)

    def __getitem__(self, index):
        item_A = Image.open(self.lst_A[index]).convert("RGB")
        item_B = Image.open(self.lst_B[index]).convert("RGB")

        # resize
        if self.imsize:
            item_A = TF.resize(item_A, self.imsize)
            item_B = TF.resize(item_B, self.imsize)

        # crop
        if self.cropsize:
            if self.cencrop:
                item_A = TF.center_crop(item_A)
                item_B = TF.center_crop(item_B)
            else:
                i, j, h, w = transforms.RandomCrop.get_params(item_A, output_size=(self.cropsize, self.cropsize))

                item_A = TF.crop(item_A, i, j, h, w)
                item_B = TF.crop(item_B, i, j, h, w)

        # normalize to tensor
        item_A = self.normalize(TF.to_tensor(item_A))
        item_B = self.normalize(TF.to_tensor(item_B))

        return item_A, item_B

def imsave(tensor, path):
    denormalize = _normalize(denormalize=True)
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(tensor)
    torchvision.utils.save_image(denormalize(tensor).clamp_(0.0, 1.0), path)
    return None

def imshow(tensor):
    denormalize = _normalize(denormalize=True)
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(tensor)
    image = TF.to_pil_image(denormalize(tensor).clamp_(0.0, 1.0))
    return image

def _normalize(denormalize=False):
    MEAN = (0.5, 0.5, 0.5)
    STD = (0.5, 0.5, 0.5)
    if denormalize:
        MEAN = [-m/s for m, s in zip(MEAN, STD)]
        STD = [1/s for s in STD]
    return transforms.Normalize(MEAN, STD)

class BufferN:
    def __init__(self, N):
        super(BufferN, self).__init__()

        self.N = N
        self.buffer = []

    def push(self, x):
        if len(self.buffer) < self.N:
            self.buffer.append(x)
        else:
            self.buffer.pop(0)
            self.buffer.append(x) 

    def get_buffer(self):
        return self.buffer
