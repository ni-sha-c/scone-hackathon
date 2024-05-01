import torch
FINAL_2PI = 2 * torch.tensor(torch.pi)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def normal(x, mu_i, sigma_i, w=1):
    det = torch.det(sigma_i)
    inv_sig = torch.inverse(sigma_i)
    diff = (x - mu_i)
    exp = torch.matmul(diff, inv_sig)
    exp = torch.sum(exp.T * diff.T, axis=0)
    normal_pdf = w * torch.exp(-0.5 * exp) / torch.sqrt(FINAL_2PI ** mu_i.size(0) * det)
    return normal_pdf


class Gaussiannd:
    def __init__(self, w_lst=None, mu_lst=None, sigma_lst=None):
        self.M = len(w_lst)
        if torch.is_tensor(w_lst):
            self.w_lst = w_lst
        else:
            self.w_lst = torch.tensor(w_lst)
        if torch.is_tensor(mu_lst):
            self.mu_lst = mu_lst
        else:
            self.mu_lst = torch.tensor(mu_lst)
        if torch.is_tensor(sigma_lst):
            self.sigma_lst = sigma_lst
        else:
            self.sigma_lst = torch.tensor(sigma_lst)
        self.dim = self.mu_lst.shape[1]
        if (self.M != len(mu_lst) and self.M != len(sigma_lst)):
            raise ValueError("All the arrays must be of the same size.")

    def rnorm(self, x, i):
        w = self.w_lst[i]
        mu = self.mu_lst[i]
        sigma = self.sigma_lst[i]
        return normal(x, mu, sigma, w)

    def prob(self, x):
        pdf = torch.sum(torch.stack([self.rnorm(x, i) for i in range(self.M)], dim=0), axis=0)
        return pdf

    def log_prob(self, x):
        log_pdf = torch.log(self.prob(x))
        return log_pdf

    def neglogprob(self, x):
        return -self.log_prob(x)

    def logprob_grad(self, x):
        arr = torch.stack([self.rnorm(x, i) for i in range(self.M)], dim=0)
        pdf = torch.sum(arr, axis=0)
        log_pdf_grad = torch.empty(self.M, self.dim ,self.dim)
        for i in range(self.M):
            log_pdf_grad[i] = -arr[i].reshape(-1, 1) * ((x - self.mu_lst[i]) @ torch.inverse(self.sigma_lst[i]).T)
        log_pdf_grad = torch.sum(log_pdf_grad, axis=0) / pdf.reshape(-1, 1)
        return log_pdf_grad

    def neglogprobgrad(self, x):
        return -self.logprob_grad(x)


# Example usage:
# sigma_lst = torch.empty((2, 2, 2))
# sigma_lst[0] = torch.eye(2)
# sigma_lst[1] = torch.eye(2)
# c = Gaussiannd([1, 0], [[0, 0], [0, 0]], sigma_lst)
# print(c.logprob_grad(torch.tensor([[5, 2], [3, 2]], dtype=torch.float32)))