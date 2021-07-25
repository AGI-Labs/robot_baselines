import baselines.net as net
from baselines.util import Metric


def get_network(name):
    if name == 'VGGSoftmax':
        return net.VGGSoftmax()
    if name == 'PointPredictor':
        return net.PointPredictor()
    raise NotImplementedError


try:
    # all torchvision modules are disabled during inference
    import baselines.datasets as datasets
except:
    pass
