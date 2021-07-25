import argparse
import baselines
import torch, os
import torch.nn as nn


# parse inputs and constants
parser = argparse.ArgumentParser()
parser.add_argument('output_folder')
parser.add_argument('--input_data', default='./data/30hz.npz')
parser.add_argument('--pretrained', default='./model/pretrained.npz')
parser.add_argument('--features', default='VGGSoftmax')
parser.add_argument('--BATCH_SIZE', default=32, type=int)
parser.add_argument('--EPOCHS', default=5, type=int)
parser.add_argument('--LR', default=1e-4, type=float)
args = parser.parse_args()
train, test = baselines.datasets.state_action_dataset(args.input_data, args.BATCH_SIZE)
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
else:
    input('Path already exists! Press enter to continue')


# build network and restore weights
features = baselines.get_network(args.features)
features.load_state_dict(torch.load(args.pretrained))
policy = baselines.net.CNNPolicy(features).cuda()


# build optim
train_metric, test_metric = baselines.Metric(), baselines.Metric()
optim = torch.optim.Adam(policy.parameters(), lr=args.LR)
loss = nn.SmoothL1Loss()
for e in range(args.EPOCHS):
    train_metric.reset(); test_metric.reset()
    policy = policy.train()
    for i, s, acs in train:
        optim.zero_grad()
        acs_hat = policy(i.cuda(), s.cuda())
        train_loss = loss(acs_hat, acs.cuda())
        train_loss.backward()
        optim.step()
        train_metric.add(train_loss.item())
        print('epoch {} \t train {:.6f} \t\t'.format(e, train_metric.mean), end='\r')
    
    policy = policy.eval()
    for i, s, acs in test:
        with torch.no_grad():
            acs_hat = policy(i.cuda(), s.cuda())
            test_metric.add(loss(acs_hat, acs.cuda()).item())
    print('epoch {} \t train {:.6f} \t test {:.6f}'.format(e, train_metric.mean, test_metric.mean))

policy = policy.cpu()
torch.save(features.state_dict(), args.output_folder + '/policy.pt')
