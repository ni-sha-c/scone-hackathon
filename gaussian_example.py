import numpy as np
FINAL_2PI = 2*np.pi
def normal(x, mu_i, sigma_i, w = 1):
        """
        Args:
            x: d x n numpy array
            mu_i: (d,) numpy array, the center for the ith gaussian.
            sigma_i: dxd numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (n,) numpy array, the probability density value of N data for the ith gaussian
        """
        det = np.linalg.det(sigma_i)
        # if det == 0:
        #     new_sig = sigma_i+SIGMA_CONST
        #     det = np.linalg.det(new_sig)
        #     inv_sig = np.linalg.inv(new_sig)
        # else:
        #     inv_sig = np.linalg.inv(sigma_i)
        inv_sig = np.linalg.inv(sigma_i)
        diff = (x - mu_i)
        exp = diff @ inv_sig
        exp = np.sum(exp.T*diff.T, axis = 0) 
        normal_pdf = w * np.exp(-0.5*exp)/np.sqrt(FINAL_2PI**(len(mu_i))*det)
        return normal_pdf


class Gaussiannd:
    def __init__(self, w_lst = None, mu_lst = None, sigma_lst = None):
        self.M = len(w_lst)
        self.w_lst = np.array(w_lst)      # w_lst : (m,) a list of the weights for each gaussian
        self.mu_lst = np.array(mu_lst)    # mu_lst : (m, d) the mean for the gaussian distribution
        self.sigma_lst = np.array(sigma_lst)      # sigma_lst : (m, d, d) the variance for the gaussian distribution 
        if (self.M != len(mu_lst) and self.M != len(sigma_lst)):
            raise ValueError("All the arrays must be of same size.") 

    def rnorm(self, x, i):
        '''
        Args:
            x : (n, d) a list of the points for which we need the density
            i : index of the required normal distribution
        Return : 
            pdf : (n,) the weighted density at x of the given gaussian component
        '''
        w = self.w_lst[i]
        mu = self.mu_lst[i]
        sigma = self.sigma_lst[i]
        return normal(x, mu, sigma, w)

    def prob(self, x):
        '''
        Args:
            x : (n, d) a list of the points for which we need the density
        Return : 
            pdf : (n, ) the density at x
        '''
        pdf = np.sum(np.array([self.rnorm(x, i) for i in range(self.M)]), axis = 0)
        return pdf

    
    def log_prob(self, x):
        '''
        Args:
            x : (1, n) a list of the points for which we need the density
        Return : 
            log_pdf : (1, n) the log density at x
        '''
        log_pdf = np.log(self.prob(x))
        return log_pdf
    
    def neglogprob(self, x):
        return -self.log_prob(x)
    
    def logprob_grad(self, x):
        '''
        Args:
            x : (n, d) a list of the points for which we need the density
        Return : 
            log_pdf_grad : (n, d) the gradient of log density at x 
        '''
        arr = np.array([self.rnorm(x, i) for i in range(self.M)])
        pdf = np.sum(arr, axis = 0)
        log_pdf_grad = []
        for i in range(self.M):
            log_pdf_grad.append(-arr[i].reshape(-1,1)*((x-self.mu_lst[i]) @ (np.linalg.inv(self.sigma_lst[i])).T))
        log_pdf_grad = np.sum(log_pdf_grad, axis = 0)/pdf.reshape(-1,1)
        return log_pdf_grad
    
    def neglogprobgrad(self, x):
        return -self.logprob_grad(x)
    

"""sigma_lst = np.empty((2, 2, 2))
sigma_lst[0] = np.eye(2)
sigma_lst[1] = np.eye(2)
c = Gaussiannd([1, 0], [[0,0], [0,0]], sigma_lst)
print(c.logprob_grad(np.array([[5,2], [3,2]])))"""

