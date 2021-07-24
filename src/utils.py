import torch
import numpy as np
import ssms

# This data structure is used when we try to train models from
# pre simulated data. i.e., when training mode is 'offline'
class OfflineDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_datapoints = cfg.train_data.shape[0]

        self.train_data = self.cfg.train_data.astype(np.float32)
        self.train_labels = self.cfg.train_labels.astype(np.float32)

    def __len__(self):
        return self.n_datapoints

    def __getitem__(self, index):
        return np.expand_dims(self.train_data[index], 0), self.train_labels[index]

class OnlineDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        # just an arbitrary number here to keep the data generation going
        self.n_datapoints = 1000000 
        self.simulators = {}

        # initialize the simulators for each of these models
        for model in self.cfg.model_names:
            my_model_config = ssms.config.model_config[model]
            self.simulators.update({model: {
                                        'low': my_model_config['param_bounds'][0],
                                        'high': my_model_config['param_bounds'][1]}})

    def __len__(self):
        return self.n_datapoints

    def __getitem__(self, index):
        # flip a coin to pick a model
        model_idx = np.random.choice(self.cfg.n_models)
        model_name = self.cfg.model_names[model_idx]

        theta = np.float32(np.random.uniform(low = self.simulators[model_name]['low'], 
                                             high = self.simulators[model_name]['high']))


        sim_data = ssms.basic_simulators.simulator(
                                theta = theta,
                                model = model_name,
                                bin_dim = 512,
                                bin_pointwise = True
                            )

        labs = np.zeros((1,self.cfg.n_models))
        labs[model_idx] = 1

        import ipdb; ipdb.set_trace()

        return sim_data['data'][0], labs
