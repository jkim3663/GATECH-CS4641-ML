import numpy as np
from tqdm import tqdm
from kmeans import KMeans


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = False # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """
        # It is possible that logit(i,j) is very large, making exp() to explode.
        # To make sure it is numerically stable, you need to subtract the max from each 
        # ROW of logits.
        a = np.exp(logit - np.max(logit, 1)[:,np.newaxis])
        b = np.sum(a, axis=1, keepdims=True)
        return a / b

    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        # s_i = log(\sum_{j=1}^{d} exp(logit(i,j)))
        # add maximum for each row of logit
        N = logit.shape[0]
        # maximum logits as row
        maxlogit = np.max(logit, 1)
        
        # find exponential value like softmax
        expVal = np.exp(logit - maxlogit[:,np.newaxis])
        
        # find the sum
        tot = np.sum(expVal, 1, keepdims=True)
        
        # add value of log while maintaining (5,1)
        logVal = np.log(tot) + np.reshape(maxlogit, (N,1))
        
        return logVal

    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        # sigma^2 is variance for the ith feature, which is the diagonal element of the 
        # covariance matrix.
        N, D = points.shape
        diag = np.diagonal(sigma_i)
        diff = points - mu_i

        frontmul = 1 / (np.sqrt(2 * np.pi * diag))
        expVal = (-1 * (diff ** 2)) / (2 * diag)
        expVal = np.exp(expVal)
        
        prod = np.prod(frontmul * expVal, axis=1)
        return prod

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """

        raise NotImplementedError

    def _init_components(self, **kwargs):  # [5pts]
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case
        """
        np.random.seed(5) #Do not remove this line!
#         self.points = X
#         self.max_iters = max_iters

#         self.N = self.points.shape[0]  # number of observations
#         self.D = self.points.shape[1]  # number of features
#         self.K = K  # number of components/clusters
        pi = (1 / self.K) * np.ones(self.K)

        idx = np.random.choice(len(self.points), replace=True, size=self.K)
        mu = self.points[idx,:]
        
        sigma = []
        for i in range(self.K):
            sigma.append(np.eye(self.D, k=0))
        
        return (pi, mu, np.array(sigma))

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        # === graduate implementation
        #if full_matrix is True:
            #...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        # loglikelihood = log(pi) + log(normalpdf)
        normal = []
        for i in range(self.K):
            normal.append(self.normalPDF(self.points, mu[i,:], sigma[i,:]))
        normal = np.stack(normal, 1)
        ans = np.log(normal + 1e-32) + np.log(pi + 1e-32)
        
        return ans

    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...

        return self.softmax(self._ll_joint(pi,mu,sigma,full_matrix))

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        pi = np.mean(gamma, axis=0)
        mu = np.zeros((self.K, self.D))
        sigma = np.zeros((self.K, self.D, self.D))
        
        # mu.shape = (3,3) KxD
        # sigma.shape = (3,3,3) KxDxD
        
#         for i in range(self.K):
#             mu[i] = np.average(self.points, axis=0, weights = gamma[:, i] / np.sum(gamma[:, i]))
#             diff = self.points - mu[i]
#             # use sqrt() / sum() for gamma so that when diff * diff.T, there it gives regular gamma multiplied
#             # shape (15,)
#             tempGamma = (np.sqrt(gamma[:,i]) / np.sum(gamma[:,i]))
#             diff *= np.nan_to_num(tempGamma)[:,np.newaxis]
#             sigma[i] = np.dot(diff.T, diff)
        
        for i in range(self.K):
            mu[i] = np.average(self.points, axis=0, weights = gamma[:, i] / np.sum(gamma[:, i]))
            diff = self.points - mu[i]
            sum_k = np.sum(gamma[:,i], axis=0)
            diag = np.diagonal((gamma[:,i]).T * diff.T @ diff / sum_k)
            np.fill_diagonal(sigma[i], diag)
        
        return (pi, mu, sigma)

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)