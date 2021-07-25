import baselines.net as net
import baselines.datasets as datasets
from baselines.util import Metric


def get_network(name):
    if name == 'VGGSoftmax':
        return net.VGGSoftmax()
    if name == 'PointPredictor':
        return net.PointPredictor()
    raise NotImplementedError
