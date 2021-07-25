import torch, cv2
import numpy as np


class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self._n = 0
        self._value = 0
    
    def add(self, value):
        self._value += value
        self._n += 1
    
    @property
    def mean(self):
        if self._n:
            return self._value / self._n
        return 0


def img2tensor(image):
    image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    image = image[:,:,::-1].astype(np.float32) / 255.0
    image -= np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
    image /= np.array([0.229, 0.224, 0.225]).reshape((1,1,3))
    return torch.from_numpy(image.transpose(2, 0, 1))
