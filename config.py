import os, pickle, glob, tqdm
import numpy as np
import bz2
import _pickle as cPickle

class Config:
    def __init__(self):
        self.cmodel_name = 'pilot_multi_model'
        self.n_models = 5
        self.model_names = ['ddm', 'angle', 'ornstein', 'weibull', 'levy']        
        self.data_path = '/media/data_cifs/projects/prj_approx-bayes/projectABC/data'

        # will be populated on prepareTrainData() call
        # if training mode is set to 'offline'
        self.train_data, self.train_labels = [], []

        # device settings
        self.device = 'cuda:0'

        # Training hyperparameters
        self.optimizer = 'Adam' # 'SGD'
        self.params = {
                    'batch_size': 8192,
                    'shuffle': True,
                    'num_workers': 1
                    }
        self.max_epochs = 250

    def prepareTrainData(self):
        cache_file = os.path.join('cache', self.cmodel_name+'.p')
        # check the cache first to see if meta data exists
        if os.path.exists(cache_file + '.pbz2'):
            print('Loading from cache...')
            #X = pickle.load(open(cache_file, 'rb'))

            data = bz2.BZ2File(cache_file + '.pbz2', 'rb')
            X = cPickle.load(data)

            self.train_data = X['train_data']
            self.train_labels = X['train_labels']
            return

        for m in range(self.n_models):
            file_path = os.path.join(
                            self.data_path,
                            self.model_names[m],
                            'training_data_binned_1_nbins_512_n_100000/*'
                        )
            files = glob.glob(file_path)
            for f in tqdm.tqdm(files[:10]):
                #import ipdb; ipdb.set_trace()
                X = pickle.load(open(f, 'rb'))
                data = X[1].astype(np.float32)
                
                self.train_data.append(data)

                # create a one-hot encoding for the label
                labs = np.zeros((X[1].shape[0], self.n_models), dtype=np.float32)
                labs[:, m] = 1
                self.train_labels.append(labs)

        self.train_data = np.vstack(self.train_data)
        self.train_labels = np.vstack(self.train_labels)

        print('Writing cache entry...')
        X = {}
        X.update({'train_data': self.train_data, 'train_labels': self.train_labels})
        #pickle.dump(X, open(cache_file, 'wb'))

        with bz2.BZ2File(cache_file + '.pbz2', 'w') as f: 
            cPickle.dump(X, f)

        return
