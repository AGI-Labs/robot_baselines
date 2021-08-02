import torch, os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def _TRAIN_TRANSFORM(h, w): 
    return transforms.Compose([transforms.RandomResizedCrop((h, w), (0.8, 1.0)),
                               transforms.RandomGrayscale(p=0.05),
                               transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.3),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])])
def _TEST_TRANSFORM(h, w):
    return transforms.Compose([transforms.Resize((h, w)),
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


class GoalCondBC(ImageStateRegression):
    def __init__(self, images, goals, states, actions, transform=None):
        super().__init__(images, states, actions, transform)
        self._goal = goals.copy()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, state, actions = super().__getitem__(idx)
        goal = Image.fromarray(self._goal[idx])
        goal = self._transform(goal) if self._transform else goal
        return img, goal, state, actions


def pretext_dataset(fname, batch_size):
    data = np.load(fname)

    # load dataset
    train_imgs = data['train_imgs']
    h, w = train_imgs.shape[1:3]
    train_data = DataLoader(ImageRegression(train_imgs, data['train_pos'], _TRAIN_TRANSFORM(h, w)), 
                            batch_size=batch_size, shuffle=True, num_workers=5)
    test_data = DataLoader(ImageRegression(data['test_imgs'], data['test_pos'], _TEST_TRANSFORM(h, w)), 
                            batch_size=256)
    return train_data, test_data, data['mean_train_pos']


def state_action_dataset(fname, batch_size, H=30):
    data = np.load(fname)
    def _flat_traj(key):
        old_shape = list(data[key].shape)
        shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
        return data[key].reshape(tuple(shape))

    # load dataset
    imgs, states, actions = [_flat_traj(k) for k in 
                                ('train_images', 'train_states', 'train_actions')]
    h, w = train_imgs.shape[1:3]
    train_mean, train_std = np.mean(actions, axis=0), np.std(actions, axis=0)
    train_data = DataLoader(ImageStateRegression(imgs, states, actions, _TRAIN_TRANSFORM(h, w)),
                            batch_size=batch_size, shuffle=True, num_workers=5)
    imgs, states, actions = [_flat_traj(k) for k in 
                                ('test_images', 'test_states', 'test_actions')]
    test_data = DataLoader(ImageStateRegression(imgs, states, actions, _TEST_TRANSFORM(h, w)), 
                            batch_size=256)
    return train_data, test_data, (train_mean, train_std)


def image_goal_dataset(fname, batch_size, H=30):
    data = np.load(fname)
    def _flat_traj(split):
        imgs = data['{}_images'.format(split)]
        states = data['{}_states'.format(split)]
        actions = data['{}_actions'.format(split)]

        B, T, A = actions.shape
        i, g, s, a = [], [], [], []
        for t in range(T - H):
            i.append(imgs[:,t])
            g.append(imgs[:,-1])
            s.append(states[:,t])
            a.append(actions[:,t:t+H])
        i, g, s, a = [np.concatenate(arr, 0) for arr in (i, g, s, a)]
        return i, g, s, a

    # load dataset
    imgs, goals, states, actions = _flat_traj('train')
    h, w = imgs.shape[2:4]
    train_data = DataLoader(GoalCondBC(imgs, goals, states, actions, _TRAIN_TRANSFORM(h, w)),
                            batch_size=batch_size, shuffle=True, num_workers=5)
    imgs, goals, states, actions = _flat_traj('test')
    test_data = DataLoader(GoalCondBC(imgs, goals, states, actions, _TEST_TRANSFORM(h, w)), 
                            batch_size=256)
    return train_data, test_data


def traj_dataset(fname, batch_size):
    data = np.load(fname)
    
    # load dataset
    imgs, states, actions = data['train_images'][:,0], data['train_states'][:,0], \
                            data['train_actions']
    h, w = imgs.shape[1:3]
    train_data = DataLoader(ImageStateRegression(imgs, states, actions, _TRAIN_TRANSFORM(h, w)),
                            batch_size=batch_size, shuffle=True, num_workers=5)
    imgs, states, actions = data['test_images'][:,0], data['test_states'][:,0], \
                            data['test_actions']
    test_data = DataLoader(ImageStateRegression(imgs, states, actions, _TEST_TRANSFORM(h,w)), 
                            batch_size=256)
    return train_data, test_data

