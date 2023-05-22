
from distributions.normal import normal_draw, multivariate_draw, normalpdf,multivariate_draw_v1
from distributions.algebra import sigmoid, vec, unvec, is_pos_def, isPSD, nearPD
from distributions.bernoulli import bernoulli_draw
from distributions.beta import beta_draw
from distributions.polyagamma import polyagamma_draw

import numpy, itertools, math, time
import numpy as np
from tqdm import tqdm
import utils
import time
import ray

from multiprocessing import Process
import multiprocessing

import scipy.sparse as sps
from scipy.sparse.linalg import inv

ALL_METRICS = ['MSE_Cascade', 'R^2_Cascade', 'Rp_Cascade', 'MSE_Graph', 'R^2_Graph', 'Rp_Graph']
ALL_QUALITY = ['loglikelihood', 'BIC', 'AIC', 'MSE', 'ELBO']
OPTIONS_INIT_UXY = ['zero', 'random', 'normal', 'beta', 'bernoulli']


class bnmf_gibbs:
    def __init__(self, G, C, E, P, D, T, name, hyperparameters):
        ''' Set up the class and do some checks on the values passed. '''
        self.G = numpy.array(G, dtype=float)
        self.C = numpy.array(C, dtype=float)
        self.E = numpy.array(E, dtype=float)
        self.P = numpy.array(P, dtype=float)
        self.D = D
        self.T = T
        self.name = name

        assert len(self.G.shape) == 2, "Input matrix G is not a two-dimensional array, " \
                                       "but instead %s-dimensional." % len(self.G.shape)
        assert self.G.shape == self.E.shape, "Input matrix G is not of the same size as " \
                                             "the indicator matrix E: %s and %s respectively." % (
                                             self.G.shape, self.E.shape)

        assert len(self.C.shape) == 2, "Input matrix C is not a two-dimensional array, " \
                                       "but instead %s-dimensional." % len(self.C.shape)
        assert self.C.shape == self.P.shape, "Input matrix C is not of the same size as " \
                                             "the indicator matrix P: %s and %s respectively." % (
                                             self.C.shape, self.P.shape)

        (self.N, self.M) = self.C.shape
        self.G_index = numpy.copy(self.E)
        self.C_index = numpy.copy(self.P)
        self.size_Omega = self.E.sum()
        self.size_Gamma = self.P.sum()
        self.I_N = numpy.eye(self.N, self.N)
  

        self.alpha1, self.alpha2 = float(hyperparameters['alpha1']), float(hyperparameters['alpha2'])
        self.sigmaC, self.sigmaR = float(hyperparameters['sigmaC']), float(
            hyperparameters['sigmaR'])  
        self.covX, self.covY, self.covU = numpy.array(hyperparameters['covX']), numpy.array(
            hyperparameters['covY']), numpy.array(hyperparameters['covU'])
        # Make lambdaU/V into a numpy array if they are a float
        self.covX_inv = np.linalg.pinv(self.covX)
        self.covY_inv = np.linalg.pinv(self.covY)
        self.covU_inv = np.linalg.pinv(self.covU)


    def check_empty_rows_columns(R, M):
        ''' Raise an exception if an entire row or column is empty. '''
        sums_columns = M.sum(axis=0)
        sums_rows = M.sum(axis=1)

        # Assert none of the rows or columns are entirely unknown values
        for i, c in enumerate(sums_rows):
            assert c != 0, "Fully unobserved row in" + R + ", row %s." % i
        for j, c in enumerate(sums_columns):
            assert c != 0, "Fully unobserved column in" + R + ", column %s." % j

    def train(self, init_UXY, params):
        ''' Initialise and run the sampler. '''
        self.initialise(init_UXY=init_UXY)
        self.run(params)

    def initial_positive_normal(self, shape):
        matrix = np.random.normal(size=shape)
        matrix[matrix < 0] = 0
        return matrix

    def initialise(self, init_UXY='zero'):
        ''' Initialise U, V, tau, and lambda (if ARD). '''
        assert init_UXY in OPTIONS_INIT_UXY, "Unknown initialisation option: %s. Should be in %s." % (
        init_UXY, OPTIONS_INIT_UXY)

        if init_UXY == 'zero':
            self.X = numpy.zeros((self.D, self.N))
            self.U = numpy.zeros((self.D, self.N))
            self.Y = numpy.zeros((self.D, self.M))
            self.muE = self.alpha1 / self.alpha2
            self.Lambda = numpy.zeros((self.N, self.N))
            self.R = numpy.zeros((self.N, self.N))

        if init_UXY == 'random':
            self.X = self.initial_positive_normal(shape=(self.D, self.N))
            self.U = self.initial_positive_normal(shape=(self.D, self.N))
            self.Y = self.initial_positive_normal(shape=(self.D, self.M))
            self.muE = self.alpha1 / self.alpha2
            self.Lambda = self.initial_positive_normal(shape=(self.N, self.N))
            self.R = self.initial_positive_normal(shape=(self.N, self.N))

        # Initialise X,Y,U,R,Lambda
        for i in (range(self.D)):
           

            self.X[i, :] = multivariate_draw_v1(numpy.zeros(self.N), self.covX, 1) if init_UXY == 'random' else 0.1
            self.Y[i, :] = multivariate_draw_v1(numpy.zeros(self.M), self.covY, 2) if init_UXY == 'random' else 0.1
            self.U[i, :] = multivariate_draw_v1(numpy.zeros(self.N), self.covU, 3) if init_UXY == 'random' else 0.1

        for j, k in itertools.product(range(self.N), range(self.N)):
            if j == k:
                self.R[j, k] = -1000
            else:
                self.R[j, k] = 100 if self.G[j, k] == 1 else -4

    def run(self, params):
        ray.init()
        iterations = int(params['iterations'])
        initial_burn_in = int(params['initial_burn_in'])
        compute_perf = bool(params['compute_perf'])

        ''' Run the Gibbs sampler. '''
        self.all_U = numpy.zeros((iterations + 1 - initial_burn_in, self.D, self.N))
        self.all_Y = numpy.zeros((iterations + 1 - initial_burn_in, self.D, self.M))
        self.all_X = numpy.zeros((iterations + 1 - initial_burn_in, self.D, self.N))
        self.all_E = numpy.zeros((iterations + 1 - initial_burn_in, self.N, self.N))
      
        self.all_times = []  # to plot performance against time
        self.all_performances = {}  # for plotting convergence of metrics
        self.all_loglikelihood = []
        for metric in ALL_METRICS:
            self.all_performances[metric] = []
     
        self.all_E[0], self.all_X[0], self.all_Y[0], self.all_U[0]= \
            numpy.copy(self.E), numpy.copy(self.X), numpy.copy(self.Y), numpy.copy(self.U)

        self.sample_P()
        for it in tqdm(range(iterations)):
            # for it in (range(iterations)):
            time_start = time.time()
            self.R = np.copy(self.sample_R(self.R, self.N, self.E, self.G, self.sigmaR, self.X, self.U, self.Lambda))
            self.E = np.copy(self.sample_E(self.E, self.N, self.R, self.muE, self.G))
            new_X = self.sample_X.remote(self.Y, self.N, self.P, self.sigmaC, self.U, self.D, self.covX_inv, self.I_N,
                                         self.sigmaR, self.C, self.R, self.X,self)
            new_U = self.sample_U.remote(self.sigmaR, self.I_N, self.X, self.N, self.D, self.U, self.R, self.covU_inv,self)
            new_Y = self.sample_Y.remote(self.M, self.X, self.P, self.sigmaC, self.D, self.C, self.covY_inv, self.Y,
                                         self)
            new_Lambda = self.sample_Lambda.remote(self.Lambda, self.N, self.E, self.R)
            new_muE = self.sample_muE.remote(self.E, self.alpha1, self.alpha2, self.N)

            self.U = np.copy(ray.get(new_U))
            self.X = np.copy(ray.get(new_X))
            self.Y = np.copy(ray.get(new_Y))
            self.Lambda = np.copy(ray.get(new_Lambda))
            self.muE = np.copy(ray.get(new_muE))

            del new_U
            del new_X
            del new_Y
            del new_Lambda
            del new_muE

            # Store draws
            if it > initial_burn_in:
             
                self.all_E[it + 1 - initial_burn_in], self.all_X[it + 1 - initial_burn_in], self.all_Y[
                    it + 1 - initial_burn_in], self.all_U[it + 1 - initial_burn_in], = numpy.copy(self.E),numpy.copy(
                    self.X), numpy.copy(self.Y), numpy.copy(self.U)
                # Store and #print performances

                if compute_perf:
                    perf = self.predict_while_running()
                    self.all_loglikelihood.append(self.log_likelihood(self.X, self.Y, self.U, self.E, self.P))
                    for metric in ALL_METRICS:
                        self.all_performances[metric].append(perf[metric])
                    # Store time taken for iteration
                    time_iteration = time.time()
                    tt = time_iteration - time_start
                    self.all_times.append(tt)

    # @ray.remote
    def sample_E(self, E, N, R, muE, G):
        new_E = numpy.copy(E)
        for i, j in itertools.product(range(N), range(N)):
            if G[i, j] == 1:
                new_E[i, j] = 1
            elif i == j:
                new_E[i, j] = 0
            else:
                parameter = (muE - muE * sigmoid(R[i, j])) / \
                            (1 - muE * sigmoid(R[i, j]))
                new_E[i, j] = bernoulli_draw(parameter)
        return new_E

    @ray.remote
    def get_y123(M, X, P, sigmaC):
        y1 = utils.kron_identity2arr(M, X)
        y123 = y1 * (vec(P).T * (1. / sigmaC))
        y123 = sps.csc_matrix(y123)
        return y123

    @ray.remote
    def get_y4(M, X):
        y4 = utils.kron_identity2arr(M, X.T)
        y4 = sps.csc_matrix(y4)
        return y4

    @ray.remote
    def get_y5(D, covY_inv):
        y5 = utils.kron_identity2arr(D, covY_inv)
        return y5

    @ray.remote
    def sample_Y(M, X, P, sigmaC, D, C, covY_inv, Y, self):

        ty4 = self.get_y4.remote(M, X)
        ty123 = self.get_y123.remote(M, X, P, sigmaC)
        ty5 = self.get_y5.remote(D, covY_inv)

        y123 = ray.get(ty123)
        y4 = ray.get(ty4)
        y5 = ray.get(ty5)

        del ty123
        del ty4
        del ty5

        y1234 = y123.dot(y4)
        y1234 = y1234.toarray()
        temp = y1234 + y5
        cov = np.linalg.pinv(temp) #utils.inverse_block_matrix(temp, D)
        # mean = cov.dot(y123.dot(vec(np.multiply(P, C))))
        mean = cov.dot(y123.dot(vec(C)))
        new_Y = unvec(multivariate_draw(mean.reshape((-1)), cov, 4), Y.shape)
        return new_Y

    @ray.remote
    def get_u1u2(sigmaR, N, X):
        u1 = np.dot(X, X.T)
        u2 = ((1. / sigmaR) * utils.kron_identity2arr(N, u1))
        return u2

    @ray.remote
    def get_u3(D, covU_inv):
        u3 = utils.kron_identity2arr(D, covU_inv)
        return u3

    @ray.remote
    def get_ukron(sigmaR, N, X, R):
        kro = (1. / sigmaR) * utils.kron_identity2arr(N, X)
        krodot = np.dot(kro, vec(R))
        return krodot

    @ray.remote
    def sample_U(sigmaR, I_N, X, N, D, U, R, covU_inv, self):
        tu2 = self.get_u1u2.remote(sigmaR, N, X)
        tu3 = self.get_u3.remote(D, covU_inv)
        tkro = self.get_ukron.remote(sigmaR, N, X, R)

        u2 = ray.get(tu2)
        u3 = ray.get(tu3)
        ukro = ray.get(tkro)

        del tu2
        del tu3
        del tkro

        temp=u2+u3
        cov = np.linalg.pinv(temp)#utils.inverse_block_matrix(temp, D)
        mean = np.dot(cov, ukro)
        new_U = (unvec(multivariate_draw(mean.reshape((-1)), cov, 5), U.shape))
        return new_U

    def sample_P(self):
        for i, r in itertools.product(range(self.N), range(self.M)):

            if self.C[i, r] > 0.00 and self.C[i, r] <= self.T:
                self.P[i, r] = 1

    @ray.remote
    def sample_muE(E, alpha1, alpha2, N):
        E_total = np.sum(E)
        param1 = alpha1 + E_total
        param2 = alpha2 + (N ** 2) - E_total
        new_muE = beta_draw(param1, param2)
        return new_muE



    @ray.remote
    def sample_Lambda(Lambda, N, E, R):
        new_Lambda = np.copy(Lambda)
        for i, j in itertools.product(range(N), range(N)):
            if i == j:
                new_Lambda[i, j] = 0.00
            elif (E[i, j] == 1):
                pg = polyagamma_draw(1 * np.ones(1), R[i, j] * np.ones(1))
                new_Lambda[i, j] = pg
        return new_Lambda


    @ray.remote
    def get_X1234(Y, N, P, sigmaC):
        X1 = utils.kron_arr2identity(Y, N)
        X4 = utils.kron_arr2identity(Y.T, N)
        X4 = sps.csc_matrix(X4)
        X123 = X1 * (vec(P).T * (1. / sigmaC))
        X123 = sps.csc_matrix(X123)
        X1234 = np.dot(X123,X4)
        return X1234, X123

    @ray.remote
    def get_X5(U, N, sigmaR):
        X51 = utils.kron_arr2identity(U, N)
        X51 = sps.csc_matrix(X51)
        X512 = X51 * (1. / sigmaR)
        X53 = utils.kron_arr2identity(U.T, N)
        X53 = sps.csc_matrix(X53)
        X5 = np.dot(X512, X53)
        return X5, X512

    @ray.remote
    def get_X6(D, covX_inv):
        X6 = utils.kron_identity2arr(D, covX_inv)
        X6 = sps.csc_matrix(X6)
        return X6


    @ray.remote
    def sample_X(Y, N, P, sigmaC, U, D, covX_inv, I_N, sigmaR, C, R, X,self):
        t1 = self.get_X1234.remote(Y, N, P, sigmaC)
        t2 = self.get_X5.remote(U, N, sigmaR)
        t3 = self.get_X6.remote(D, covX_inv)

        X1234, X123 = ray.get(t1)
        X5, X512 = ray.get(t2)
        X6 = ray.get(t3)

        del t1
        del  t2
        del t3

        tmp = X1234 + X5 + X6
        cov = inv(tmp).toarray()
        # mean = cov.dot((X123.dot(vec(np.multiply(P, C)))) + (X512.dot(vec(R))))
        mean = cov.dot((X123.dot(vec(C))) + (X512.dot(vec(R))))
        new_X = unvec(multivariate_draw(mean.reshape((-1)), cov), X.T.shape).T
        return new_X

    # @ray.remote
    def sample_R(self, R, N, E, G, sigmaR, X, U, Lambda):
        new_R = np.copy(R)
        for i, j in itertools.product(range(N), range(N)):
            if i == j:
                new_R[i, j] = 0.00
            else:
                mean = ((E[i, j] * (G[i, j] - 0.5) * sigmaR) + (X[:, i].T.dot(U[:, j]))) \
                       / ((E[i, j] * Lambda[i, j] * sigmaR) + 1)
                new_R[i, j] = normal_draw(mean, 1)
        return new_R

    def approx_expectation(self, burn_in, thinning):
        ''' Return our expectation of E,P,R,X,Y,U,Lambda,muP,muE.'''

        print(burn_in, thinning)
        indices = range(burn_in, len(self.all_X), thinning)
        exp_E = numpy.array([self.all_E[i] for i in indices]).sum(axis=0) / float(len(indices))
        #exp_P = numpy.array([self.all_P[i] for i in indices]).sum(axis=0) / float(len(indices))
        #exp_R = numpy.array([self.all_R[i] for i in indices]).sum(axis=0) / float(len(indices))
        exp_X = numpy.array([self.all_X[i] for i in indices]).sum(axis=0) / float(len(indices))
        exp_Y = numpy.array([self.all_Y[i] for i in indices]).sum(axis=0) / float(len(indices))
        exp_U = numpy.array([self.all_U[i] for i in indices]).sum(axis=0) / float(len(indices))
        #exp_Lambda = numpy.array([self.all_Lambda[i] for i in indices]).sum(axis=0) / float(len(indices))
        #exp_muP = sum([self.all_muP[i] for i in indices]) / float(len(indices))
        #exp_muE = sum([self.all_muE[i] for i in indices]) / float(len(indices))
        return (exp_X, exp_Y, exp_U, exp_E) #, exp_P, exp_Lambda, exp_muP, exp_muE, exp_R)

    def predict(self, C_index, G_index, burn_in, thinning):
        ''' Compute the expectation of U and V, and use it to predict missing values. '''
        (exp_X, exp_Y, exp_U, exp_E) = self.approx_expectation(burn_in, thinning)

        C_pred = numpy.dot(exp_X.T, exp_Y)
        G_pred = sigmoid(numpy.dot(exp_X.T, exp_U))

        C_pred[C_pred < 0] = 0
        C_pred[C_pred > self.T] = 0

        MSE_C = self.compute_MSE(C_index, self.C, C_pred, self.P, 'Cascade')
        R2_C = self.compute_R2(C_index, self.C, C_pred)
        Rp_C = self.compute_Rp(C_index, self.C, C_pred)

        MSE_G = self.compute_MSE(G_index, self.G, G_pred, exp_E, 'Graph')
        R2_G = 0  # self.compute_R2(G_index, self.G, G_pred)
        Rp_G = 0  # self.compute_Rp(G_index, self.G, G_pred)

        return {'MSE_Cascade': MSE_C, 'R^2_Cascade': R2_C, 'Rp_Cascade': Rp_C, 'MSE_Graph': MSE_G, 'R^2_Graph': R2_G,
                'Rp_Graph': Rp_G}

    def predictTest(self, C_ground, C_index, G_ground, G_index, burn_in, thinning):
        ''' Compute the expectation of U and V, and use it to predict missing values. '''
        (exp_X, exp_Y, exp_U, exp_E) = self.approx_expectation(burn_in, thinning)

        C_pred = numpy.dot(exp_X.T, exp_Y)
        C_pred[C_pred < 0] = 0
        C_pred[C_pred > self.T] = 0

        G_pred = sigmoid(numpy.dot(exp_X.T, exp_U))
        G_pred[exp_E == 0.0] = 0.0
        G_pred[G_pred > 0.5] = 1.0
        G_pred[G_pred <= 0.5] = 0.0

        MSE_C = self.compute_MSE(C_index, C_ground, C_pred, self.P, 'Cascade')
        R2_C = self.compute_R2(C_index, C_ground, C_pred)
        Rp_C = self.compute_Rp(C_index, C_ground, C_pred)

        MSE_G = self.compute_MSE(G_index, G_ground, G_pred, exp_E, 'Graph')
        R2_G = 0  # self.compute_R2(G_index, self.G, G_pred)
        Rp_G = 0  # self.compute_Rp(G_index, self.G, G_pred)

        return {'MSE_Cascade': MSE_C, 'R^2_Cascade': R2_C, 'Rp_Cascade': Rp_C, 'MSE_Graph': MSE_G, 'R^2_Graph': R2_G,
                'Rp_Graph': Rp_G}

    def train_test_mse(self):
        C_pred = numpy.dot(self.X.T, self.Y)
        G_pred = sigmoid(numpy.dot(self.X.T, self.U))
        C_pred[C_pred < 0] = 0
        C_pred[C_pred > self.T] = 0
        MSE_C_trian = self.compute_MSE(self.C_index, self.C, C_pred, None, 'Cascade')
        MSE_C_test = self.compute_MSE(self.C_index_ground, self.C, C_pred, None, 'Cascade')

        return MSE_C_trian, MSE_C_test

    def predict_while_running(self):
        ''' Predict the training error while running. '''
        C_pred = numpy.dot(self.X.T, self.Y)
        G_pred = sigmoid(numpy.dot(self.X.T, self.U))
        C_pred[C_pred < 0] = 0
        C_pred[C_pred > self.T] = 0
        MSE_C = self.compute_MSE(self.C_index, self.C, C_pred, self.P, 'Cascade')
        R2_C = self.compute_R2(self.C, self.C, C_pred)
        Rp_C = self.compute_Rp(self.C, self.C, C_pred)

        MSE_G = self.compute_MSE(self.G_index, self.G, G_pred, self.E, 'Graph')
        R2_G = 0  # self.compute_R2(self.G, self.G, G_pred)
        Rp_G = 0  # self.compute_Rp(self.G, self.G, G_pred)

        return {'MSE_Cascade': MSE_C, 'R^2_Cascade': R2_C, 'Rp_Cascade': Rp_C, 'MSE_Graph': MSE_G, 'R^2_Graph': R2_G,
                'Rp_Graph': Rp_G}

    ''' Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation) '''

    def compute_MSE(self, M, R, R_pred, learnindex, type):
        ''' Return the MSE of predictions in R_pred, expected values in R, for the entries in M. '''
        if type == 'Cascade':
            return (M * (R - R_pred) ** 2).sum() / float(M.sum())
        if type == 'Graph':
            R_pred[learnindex == 0.0] = 0.0
            R_pred[R_pred > 0.5] = 1.0
            # R_pred [R_pred <=0.5] = 0.0
            return (M * (R - R_pred) ** 2).sum() / float(M.sum())

    def compute_R2(self, M, R, R_pred):
        ''' Return the R^2 of predictions in R_pred, expected values in R, for the entries in M. '''
        mean = (M * R).sum() / float(M.sum())
        SS_total = float((M * (R - mean) ** 2).sum())
        SS_res = float((M * (R - R_pred) ** 2).sum())
        return 1. - SS_res / SS_total if SS_total != 0. else numpy.inf

    def compute_Rp(self, M, R, R_pred):
        ''' Return the Rp of predictions in R_pred, expected values in R, for the entries in M. '''
        mean_real = (M * R).sum() / float(M.sum())
        mean_pred = (M * R_pred).sum() / float(M.sum())
        covariance = (M * (R - mean_real) * (R_pred - mean_pred)).sum()
        variance_real = (M * (R - mean_real) ** 2).sum()
        variance_pred = (M * (R_pred - mean_pred) ** 2).sum()
        return covariance / float(math.sqrt(variance_real) * math.sqrt(variance_pred))

    def quality(self, metric, burn_in, thinning):
        ''' Return the model quality, either as log likelihood, BIC, AIC, MSE, or ELBO. '''
        assert metric in ALL_QUALITY, 'Unrecognised metric for model quality: %s.' % metric
        (exp_X, exp_Y, exp_U, exp_E) = self.approx_expectation(burn_in, thinning)
        log_likelihood = self.log_likelihood(exp_X, exp_Y, exp_U, exp_E, self.P)

        if metric == 'loglikelihood':
            return log_likelihood
        elif metric == 'BIC':
            # -2*loglikelihood + (no. free parameters * log(no data points))
            return - 2 * log_likelihood + self.number_parameters() * math.log(self.size_Omega) * math.log(
                self.size_Gamma)
        elif metric == 'AIC':
            # -2*loglikelihood + 2*no. free parameters
            return - 2 * log_likelihood + 2 * self.number_parameters()
        elif metric == 'MSE':
            C_pred = numpy.dot(exp_X.T, exp_U)
            P_pred = sigmoid(numpy.dot(exp_X.T, exp_Y))
            return (self.compute_MSE(self.C_index, self.C, C_pred, self.P, 'Cascade') + self.compute_MSE(self.G_index,
                                                                                                        self.G, exp_E,
                                                                                                        G_pred,
                                                                                                        'Graph'))
        elif metric == 'ELBO':
            return 0.

    def log_likelihood(self, exp_X, exp_Y, exp_U, exp_E, P):
        ''' Return the likelihood of the data given the trained model's parameters. '''
        ''' HERE!'''
        C_pre = numpy.dot(exp_X.T, exp_Y)
        C_pre[P == 0] = 0.00
        exp_logsigmaC = math.log(self.sigmaC)
        C = self.size_Gamma / 2. * (exp_logsigmaC - math.log(2 * math.pi)) - exp_logsigmaC / 2. * (
                    self.C_index * (self.C - C_pre) ** 2).sum()
        G = 0.00
        R = sigmoid(numpy.dot(exp_X.T, exp_U))
        R[exp_E == 0] = 0.00
        R[R >= 0.5] = 1.00

        for i, j in itertools.product(range(self.N), range(self.N)):
            G += (-self.G[i, j] * math.log(1 + math.exp(-1 * R[i, j]))) + (
                        (1 - self.G[i, j]) * (-R[i, j] - math.log(1 + math.exp(-1 * R[i, j]))))
        return C, G
  

    def number_parameters(self):
        ''' Return the number of free variables in the model. '''
        return (self.N * self.N * 2) + (self.D * self.N * 2) + (self.D * self.M) + (self.M * self.N) + 2

