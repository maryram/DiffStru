import pickle
from utils import rmse
from pathlib import Path
import numpy as np
from DiffStru import bnmf_gibbs


from performance_metrics import print_metrics
from distributions.algebra import sigmoid
import utils


class Model:
    def __init__(self, args):

        # path for saving model as pickle file
        self.model_path = (
            Path(args.dataset_path) / "models" / str(args.dim) / "model.pkl"
        )

        self.args = args
        self.args.dataset_path = Path(args.dataset_path)
        self.model = None

        # path for saving parameters like X, Y, ...
        self.output_path = self.args.dataset_path / "output" / str(args.dim)

        dataset_path = args.dataset_path

        self.true_c = np.genfromtxt(dataset_path / 'Groundtruth_C.txt', delimiter=',')
        self.observed_c = np.genfromtxt(dataset_path / 'Observed_C.txt', delimiter=',')
        self.ZERO_TIME = utils.get_zero_time(self.true_c)
        self.observed_c[self.true_c == self.ZERO_TIME] = self.ZERO_TIME

        second_indice = utils.get_second_node_indice(self.true_c)
        self.observed_c[second_indice] = self.true_c[second_indice]

        self.true_g = np.genfromtxt(dataset_path / 'Groundtruth_G.txt', delimiter=',')
        self.observed_g = np.genfromtxt(dataset_path / 'Observed_G.txt', delimiter=',')

        self.N = self.observed_c.shape[0]
        self.M = self.observed_c.shape[1]
        T = np.max(np.max(self.observed_c))
        self.T = T

        # Mask
        self.C_index = (self.observed_c > 0) * (self.observed_c <= T)
        self.G_index = self.observed_g > 0
        self.C_index_ground = (self.observed_c == 0) * (self.true_c > 0)
        self.G_index_ground = (self.observed_g == 0) * (self.true_g > 0)

    def train(self, params, hyperparameters=None):
        args = self.args

        # create directories
        if not (args.dataset_path / "models").exists():
            (args.dataset_path / "models").mkdir()
        if not (args.dataset_path / "models" / str(self.args.dim)).exists():
            (args.dataset_path / "models" / str(self.args.dim)).mkdir()
        if self.model_path.exists():
            print("Model is trained")
            return
        if not self.model_path.exists():
            (args.dataset_path / "params").write_text(str(params))

        D = args.dim

        observed_c = self.observed_c
        observed_g = self.observed_g

        N, M, T = self.N, self.M, self.T

        C_index = self.C_index
        G_index = self.G_index

        covX = np.eye(N, N)
        covY = np.eye(M, M)
        covU = np.copy(covX)

        hyperparameters["covX"] = covX
        hyperparameters["covY"] = covY
        hyperparameters["covU"] = covU

        BNMF = bnmf_gibbs(
            G=observed_g,
            C=observed_c,
            E=G_index,
            P=C_index,
            D=D,
            T=T,
            hyperparameters=hyperparameters,
        )

        init_UXY = "random"
        BNMF.train(init_UXY=init_UXY, params=params)

        # save model
        pickle.dump(BNMF, self.model_path.open("wb"))

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
        model = pickle.load(self.model_path.open("rb"))
        self.model = model
        T = model.T
        (exp_X, exp_Y, exp_U, exp_E) = model.approx_expectation(burn_in, thinning)

        # predict G and C
        C_pred = np.dot(exp_X.T, exp_Y)
        G_pred = sigmoid(np.dot(exp_X.T, exp_U))
        C_pred[C_pred < 0] = 0

        C_pred[(C_pred < self.ZERO_TIME) * (C_pred > 0)] = self.ZERO_TIME
        C_pred[C_pred > T + self.ZERO_TIME] = 0

        """
        activity = utils.jaccard_cascade(self.observed_c)
        tmp = self.C_index.T.dot(activity).T
        C_pred[tmp < tmp.mean()] = 0
        """

        return C_pred, G_pred

    def save_output(self, params):
        """
        saves `X, Y, U, E, precited cascades, predicted graph` in a directory named `output`.
        """

        # model = self.model
        # (exp_X,exp_Y,exp_U,exp_E,_,_,_,_,_) = model.approx_expectation(burn_in,thinning)
        # C, G = self.predict_C_G(burn_in, thinning)

        path = self.output_path

        if not path.exists():
            path.mkdir(parents=True)

        exp_X = params["expX"]
        exp_U = params["expU"]
        exp_Y = params["expY"]
        G = params["G"]
        C = params["C"]
        G_out = params["G_out"]
        C_out = params["C_out"]

        np.savetxt(path / "X", exp_X, delimiter=",", fmt="%1.8f")
        np.savetxt(path / "U", exp_U, delimiter=",", fmt="%1.8f")
        np.savetxt(path / "Y", exp_Y, delimiter=",", fmt="%1.8f")
        np.savetxt(path / "G", G, delimiter=",", fmt="%1.8f")
        np.savetxt(path / "C", C, delimiter=",", fmt="%1.8f")
        np.savetxt(path / "G_out", G_out, delimiter=",", fmt="%1.8f")
        np.savetxt(path / "C_out", C_out, delimiter=",", fmt="%1.8f")
