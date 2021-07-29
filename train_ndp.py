import argparse
import baselines
import torch, os
import torch.nn as nn
import matplotlib.pyplot as plt


# parse inputs and constants
parser = argparse.ArgumentParser()
parser.add_argument('output_folder')
parser.add_argument('--input_data', default='./data/30hz.npz')
parser.add_argument('--pretrained', default='./model/pretrained.npz')
parser.add_argument('--features', default='VGGSoftmax')
parser.add_argument('--BATCH_SIZE', default=64, type=int)
parser.add_argument('--EPOCHS', default=5000, type=int)
parser.add_argument('--LR', default=1e-3, type=float)
parser.add_argument('--SAVE_FREQ', default=1000, type=int)
parser.add_argument('--N', default=270, type=int)
parser.add_argument('--T', default=299, type=int)


args = parser.parse_args()
train, test = baselines.datasets.traj_dataset(args.input_data, args.BATCH_SIZE)
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
else:
    input('Path already exists! Press enter to continue')


# build network and restore weights
features = baselines.get_network(args.features)
features.load_state_dict(torch.load(args.pretrained))
policy = baselines.net.DMPNet(features, args.N, args.T).cuda()


# build optim
train_metric, test_metric = baselines.Metric(), baselines.Metric()
optim = torch.optim.Adam(policy.parameters(), lr=args.LR)
loss = nn.MSELoss()
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

    if (e + 1) % args.SAVE_FREQ == 0 or e + 1 == args.EPOCHS:
        acs_hat = acs_hat.cpu().numpy()
        acs = acs.cpu().numpy()
        plt.close()
        fig, axs = plt.subplots(4, 7, figsize=(30, 15))
        for k in range(4):
            for i in range(7):
                axs[k, i].plot(acs_hat[k, :, i], label='global pol test')
                axs[k, i].plot(acs[k, :, i], label='test')
        plt.legend()
        plt.savefig(args.output_folder + '/plots_{}.png'.format(e))
        torch.save(policy.state_dict(), args.output_folder + '/policy_epoch{}.pt'.format(e))
