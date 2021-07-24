import torch
import numpy as np
import glob, os, pickle, argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import Config
from cmodel import cModel, simplex_loss
from utils import OfflineDataset, OnlineDataset

def train_simplex(args):

    cfg = Config()
    device = torch.device(cfg.device)

    # Determine whether we use presimulated data or do it on-the-fly
    if args.offline:
        cfg.prepareTrainData()    
        training_set = OfflineDataset(cfg)
    else:
        training_set = OnlineDataset(cfg)

    training_generator = torch.utils.data.DataLoader(training_set, **cfg.params)

    sample, _ = training_set.__getitem__(0)
    sample = torch.tensor(sample).unsqueeze(0)

    # Build the model and specify the training objective
    model = cModel(sample, cfg.n_models)    
    loss_fn = simplex_loss

    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    elif cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

    model = model.to(device)
    model = model.train()

    global_step = torch.tensor(0.)
    annealing_step = torch.tensor(10. * training_generator.__len__())
    beta = torch.tensor(np.ones((1,cfg.n_models), dtype=np.float32)).to(device)

    for epoch in range(cfg.max_epochs):
        # training loop
        for cnt, (batch, lab) in enumerate(training_generator):
            # get a batch of data, transfer over to the device  
            batch, lab = batch.to(device), lab.to(device)

            # reset grads
            optimizer.zero_grad()
            
            # forward pass through the model
            alpha_hat = model(batch)
            alpha = alpha_hat + 1.

            #lp = F.log_softmax(alpha_hat, dim=-1)
            #loss = F.nll_loss(lp, torch.argmax(lab,axis=1).long())

            #import ipdb; ipdb.set_trace()
            loss = loss_fn(p=lab, alpha=alpha, global_step=global_step, annealing_step=annealing_step, K=cfg.n_models, beta=beta)

            # backward pass 
            loss.backward()
            optimizer.step()

            # logging train stats
            print('Epoch:{} | Batch: {}/{} | loss: {}'.format(epoch, cnt, training_generator.__len__(), loss.detach().item()))
            global_step += 1.
    
    # saving model
    torch.save(model.state_dict(), 'models/{}.pth'.format(cfg.cmodel_name))


def eval_simplex(args):
    cfg = Config()
    device = torch.device(cfg.device)

    model = cModel(test_input=None, n_classes=cfg.n_models)    
    model.load_state_dict(torch.load('models/{}.pth'.format(cfg.cmodel_name)))

    model = model.eval()
    import ipdb; ipdb.set_trace()

    dset = 'weibull'
    inference_data_path = '/media/data_cifs/projects/prj_approx-bayes/projectABC/data/{}/parameter_recovery_data_binned_1_nbins_512_n_4096'.format(dset)
    data_file = glob.glob(os.path.join(inference_data_path, '*.pickle'))[0]

    X = pickle.load(open(data_file,'rb'))

    test_data = torch.tensor(X[1][0])
    out = model(test_data.float())

    alpha = out + 1.
    probs = alpha / torch.sum(alpha, axis=1, keepdim=True)
    U = cfg.n_models /  torch.sum(alpha, axis=1, keepdim=True)

    U = U.detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot()

    for k in range(cfg.n_models):
        p1 = torch.where(torch.argmax(probs, axis=1) == k)
        print(p1[0].shape)
        # demonstration
        ax.hist(U[p1], bins=50, alpha=0.5, label=cfg.model_names[k])

    ax.set_xlabel('Uncertainity')
    ax.set_ylabel('Count')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='frobble')
    parser.add_argument('--train_simplex', action='store_true',
                     help='train the classification model')
    parser.add_argument('--eval_simplex', action='store_true',
                     help='run inference on the classification model')
    parser.add_argument('--offline', action='store_true',
                     help='Use presimulated data to train the classification model')

    args = parser.parse_args()

    if args.train_simplex:
        train_simplex(args)
    elif args.eval_simplex:
        eval_simplex(args)
    else:
        raise NotImplementedError
