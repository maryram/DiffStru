
import pickle
from utils import rmse
from pathlib import Path

import numpy, math, itertools # pytest,
import numpy as np
from Init import common_neighbor_directed,common_cascade,common_neighbor
from DiffStru import bnmf_gibbs
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from distributions import algebra
from performance_metrics import print_metrics, f_score 
from distributions.algebra import sigmoid
from sklearn.preprocessing import normalize
from scipy.sparse import csgraph


import utils

class Model():

    def __init__(self, args, zero_time=0.1):
        self.model_path = Path(args.dataset_path) / 'models' / str(args.dim) / 'model.pkl'
        self.args = args
        self.args.dataset_path = Path(args.dataset_path)
        self.model = None
        self.output_path = self.args.dataset_path / 'output' / str(args.dim)
        self.ZERO_TIME = zero_time

        dataset_path = args.dataset_path
        self.true_c = np.genfromtxt(dataset_path / 'Groundtruth_C.txt', delimiter=',')
        self.observed_c = np.genfromtxt(dataset_path / 'Observed_C.txt', delimiter=',')
        self.observed_c[self.true_c == self.ZERO_TIME] = self.ZERO_TIME

        second_node_indice = utils.get_second_node_indice(self.true_c)
        self.observed_c[second_node_indice] = self.true_c[second_node_indice]

        self.true_g = np.genfromtxt(dataset_path / 'Groundtruth_G.txt', delimiter=',')
        self.observed_g = np.genfromtxt(dataset_path / 'Observed_G.txt', delimiter=',')

        self.N = self.observed_c.shape[0]
        self.M = self.observed_c.shape[1]
        T = np.max(np.max(self.observed_c))
        self.T = T

        self.C_index = (self.observed_c>0) * (self.observed_c <=T)
        self.G_index=self.observed_g>0
        self.C_index_ground=(self.observed_c==0) * (self.true_c>0)
        self.G_index_ground=(self.observed_g==0) * (self.true_g>0)


    def train(self, params):
        args = self.args
    
        # create directories
        if not (args.dataset_path / 'models').exists():
            (args.dataset_path / 'models').mkdir()
        if not (args.dataset_path / 'models' / str(self.args.dim)).exists():
            (args.dataset_path / 'models' / str(self.args.dim)).mkdir()
        if self.model_path.exists():
            print('Model is trained')
            return


        dataset_path = args.dataset_path
        D = args.dim

        true_c = self.true_c
        observed_c = self.observed_c
        true_g = self.true_g
        observed_g = self.observed_g

        N, M, T = self.N, self.M, self.T

        C_index = self.C_index
        G_index = self.G_index
        C_index_ground = self.C_index_ground
        G_index_ground = self.G_index_ground

        # hyperparameters

        beta1=0.1
        beta2=0.2
        alpha1=0.2
        alpha2=0.3
        sigmaC=1
        sigmaR=1

        index = observed_c == self.ZERO_TIME

        # #CT kernel
      
        # covX = np.linalg.pinv(csgraph.laplacian(common_neighbor(self.observed_g), normed=False))
       
        # #Diffusionkernel
    
        # covU=np.copy(covX)

        # #diag kernel
        # covY = numpy.eye(M, M)

        covX= np.linalg.pinv(csgraph.laplacian(self.observed_g,normed=False))
        covU = numpy.eye(N, N)
        covY = numpy.eye(M, M)

        hyperparameters = {'beta1': beta1, 'beta2': beta2, 'alpha1': alpha1, 'alpha2': alpha2,
                'sigmaC':sigmaC,'sigmaR':sigmaR,'covX':covX,'covY':covY,'covU':covU}
        name=''
        BNMF = bnmf_gibbs(observed_g,observed_c,G_index,C_index,D,T,name,hyperparameters)
        init_UXY = 'random'
        BNMF.train(init_UXY=init_UXY, params=params)

        # save model
        pickle.dump(BNMF, self.model_path.open('wb'))

    def test(self, burn_in, thinning, e_threshold):
        model = pickle.load(self.model_path.open("rb"))
        T = model.T
        (exp_X, exp_Y, exp_U, exp_E) = model.approx_expectation(burn_in, thinning)
        C_pred, G_pred = self.predict_C_G(burn_in, thinning)
        G_pred[exp_E <= e_threshold] = 0

        """ 
        gt: ground truth graph
        ob: observed graph
        probs: probabilities of the predicted graph
        """

        best_g = print_metrics(
            gt=self.true_g,
            ob=self.observed_g,
            probs=G_pred,
            out_path=str(self.args.dataset_path / "models" / str(self.args.dim)),
        )

        # metrics for cascade prediction
        activity = utils.jaccard_cascade(self.observed_c)
        tmp = (self.C_index.T.dot(activity)).T
        sparse_C = C_pred.copy()
        sparse_C[tmp < tmp.mean()] = 0

        print(
            "Number of cascade: train, test",
            np.sum(self.C_index),
            np.sum(self.C_index_ground),
        )
        print(
            "Test rmse:",
            rmse(self.true_c[self.C_index_ground], C_pred[self.C_index_ground]),
        )
        print("Train rmse:", rmse(self.observed_c[self.C_index], C_pred[self.C_index]))
        print(
            "Non Observed rmse:",
            rmse(self.observed_c[~self.C_index], sparse_C[~self.C_index]),
        )

        params = {
            "expX": exp_X,
            "expU": exp_U,
            "expY": exp_Y,
            "G": G_pred,
            "C": C_pred,
            "G_out": best_g,
            "G_out": G_pred,
            "C_out": sparse_C,
        }

        with open("g_pred.txt", "w") as f:
            np.savetxt(f, G_pred,fmt='%1.3f')

        with open("c_pred.txt", "w") as f:
            np.savetxt(f, C_pred,fmt='%1.3f')

        self.save_output(params)


    def predict_C_G(self, burn_in, thinning):
        # load model
        model = pickle.load(self.model_path.open('rb'))
        self.model = model
        T = model.T
        (exp_X,exp_Y,exp_U,exp_E) = model.approx_expectation(burn_in,thinning)

        # predict G and C
        C_pred = numpy.dot(exp_X.T, exp_Y)
        G_pred = sigmoid(numpy.dot(exp_X.T, exp_U))
      
 
        C_pred[C_pred<0]=0
        C_pred[(C_pred < self.ZERO_TIME) * (C_pred > 0)] = self.ZERO_TIME
        C_pred[C_pred > T + self.ZERO_TIME] = 0
       
        '''
        activity = utils.jaccard_cascade(self.observed_c)
        tmp = self.C_index.T.dot(activity).T
        C_pred[tmp < tmp.mean()] = 0
        '''

        return C_pred, G_pred
        
    def save_output(self, params):
        """
        saves `X, Y, U, E, precited cascades, predicted graph` in a directory named `output`.
        """


        path = self.output_path

        if not path.exists():
            path.mkdir(parents=True)

        exp_X = params['expX']
        exp_U = params['expU']
        exp_Y = params['expY']
        G = params['G']
        C = params['C']
        G_out = params['G_out']
        C_out = params['C_out']

        np.savetxt(path / 'X', exp_X, delimiter=",", fmt='%1.8f')
        np.savetxt(path / 'U', exp_U, delimiter=",", fmt='%1.8f')
        np.savetxt(path / 'Y', exp_Y, delimiter=",", fmt='%1.8f')
        np.savetxt(path / 'G', G, delimiter=",", fmt='%1.8f')
        np.savetxt(path / 'C', C, delimiter=",", fmt='%1.8f')
        np.savetxt(path / 'G_out', G_out, delimiter=",", fmt='%1.8f')
        np.savetxt(path / 'C_out', C_out, delimiter=",", fmt='%1.8f')


    def print_cascades(self, burn_in, thinning):
        predicted_c, _ = self.predict_C_G(burn_in, thinning)

        true_c = self.true_c.T
        predicted_c = predicted_c.T
        observed_c = self.observed_c.T

        # Get a row of cascade matrix and return the sequence of nodes
        def get_sequence(row):
            sorted_row = np.sort(row)[::-1]
            return [np.where(row == x)[0][0] for x in sorted_row[sorted_row > 0]]

        for cascade_index in range(true_c.shape[0]):
            print(f"{cascade_index+1}.")
            print(f"predicted: {predicted_c[cascade_index, :]}")
            print(f"true: {true_c[cascade_index, :]}")
            print(f"observed: {observed_c[cascade_index, :]}")
            print(f"Ground Truth: {get_sequence(true_c[cascade_index, :])[::-1]}")
            print(f"Observed: {get_sequence(observed_c[cascade_index, :])[::-1]}")
            print(f"Predicted: {get_sequence(predicted_c[cascade_index, :])[::-1]}")
            exit()
            print()
