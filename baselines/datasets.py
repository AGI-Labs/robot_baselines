import torch, os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


_TRAIN_TRANSFORM = transforms.Compose([transforms.RandomResizedCrop((240, 320), (0.8, 1.0)),
                                    transforms.RandomGrayscale(p=0.05),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.3),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])])
_TEST_TRANSFORM = transforms.Compose([transforms.Resize((240, 320)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])])


class ImageRegression(Dataset):
    def __init__(self, images, targets, transform=None):
        self._images = images.copy()
        self._targets = targets.astype(np.float32).copy()
        self._transform = transform
    
    def __len__(self):
        return int(self._images.shape[0])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        assert 0 <= idx < len(self)
        np.random.seed(None)

        img = Image.fromarray(self._images[idx])
        img = self._transform(img) if self._transform else img
        target = self._targets[idx]
        return img, target


class ImageStateRegression(ImageRegression):
    def __init__(self, images, state, targets, transform=None):
        super().__init__(images, targets, transform)
        self._state = state.copy().astype(np.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, target = super().__getitem__(idx)
        state = self._state[idx]
        return img, state, target


def pretext_dataset(fname, batch_size):
    data = np.load(fname)

    # load dataset
    train_data = DataLoader(ImageRegression(data['train_imgs'], data['train_pos'], _TRAIN_TRANSFORM), 
                            batch_size=batch_size, shuffle=True, num_workers=5)
    test_data = DataLoader(ImageRegression(data['test_imgs'], data['test_pos'], _TEST_TRANSFORM), 
                            batch_size=256)
    return train_data, test_data, data['mean_train_pos']


def state_action_dataset(fname, batch_size, H=30):
    data = np.load(fname)
    def _flat_traj(key):
        d = data[key]
        if 'action' in key:
            B, T, A = d.shape
            batched = []
            for t in range(T - H):
                batched.append(d[:,t:t+H])
            batched = np.concatenate(batched, 0)
        else:
            batched = []
            for t in range(d.shape[1] - H):
                batched.append(d[:,t])
            batched = np.concatenate(batched, 0)
        return batched

    # load dataset
    imgs, states, actions = [_flat_traj(k) for k in 
                                ('train_images', 'train_states', 'train_actions')]
    train_mean, train_std = np.mean(actions, axis=0), np.std(actions, axis=0)
    train_data = DataLoader(ImageStateRegression(imgs, states, actions, _TRAIN_TRANSFORM),
                            batch_size=batch_size, shuffle=True, num_workers=5)
    imgs, states, actions = [_flat_traj(k) for k in 
                                ('test_images', 'test_states', 'test_actions')]
    test_data = DataLoader(ImageStateRegression(imgs, states, actions, _TEST_TRANSFORM), 
                            batch_size=256)
    return train_data, test_data, (train_mean, train_std)


def traj_dataset(fname, batch_size):
    data = np.load(fname)
    
    # load dataset
    imgs, states, actions = data['train_images'][:,0], data['train_states'][:,0], \
                            data['train_actions']
    train_data = DataLoader(ImageStateRegression(imgs, states, actions, _TRAIN_TRANSFORM),
                            batch_size=batch_size, shuffle=True, num_workers=5)
    imgs, states, actions = data['test_images'][:,0], data['test_states'][:,0], \
                            data['test_actions']
    test_data = DataLoader(ImageStateRegression(imgs, states, actions, _TEST_TRANSFORM), 
                            batch_size=256)
    return train_data, test_data
