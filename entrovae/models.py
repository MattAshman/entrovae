import torch
import torch.nn as nn
import numpy as np

from .utils import gaussian_diagonal_kl, gaussian_diagonal_ll

__all__ = ['VAE', 'EntroVAE']


class VAE(nn.Module):
    def __init__(self, encoder, loglikelihood, z_dim):
        super().__init__()

        self.encoder = encoder
        self.loglikelihood = loglikelihood
        self.z_dim = z_dim

    def pz(self, x):
        pz_mu = torch.zeros(x.shape[0], self.z_dim)
        pz_sigma = torch.ones(x.shape[0], self.z_dim)

        return pz_mu, pz_sigma

    def qz(self, x):
        qz_mu, qz_sigma = self.encoder(x)

        return qz_mu, qz_sigma

    def elbo(self, x, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        pz_mu, pz_sigma = self.pz(x)
        qz_mu, qz_sigma = self.qz(x)

        kl = gaussian_diagonal_kl(qz_mu, qz_sigma, pz_mu, pz_sigma).sum()

        # z_samples is shape (num_samples, batch, z_dim).
        z_samples = qz_mu + qz_sigma * torch.randn(num_samples, *qz_mu.shape)

        log_px_z = 0
        for z in z_samples:
            log_px_z += self.loglikelihood(z, x).sum()

        log_px_z /= num_samples
        elbo = (log_px_z - kl) / x.shape[0]

        return elbo


class EntroVAE(VAE):
    def __init__(self, encoder, loglikelihood, z_dim, init_scale=1.):
        super().__init__(encoder, loglikelihood, z_dim)

        self.logscale = nn.Parameter(torch.ones(z_dim) * np.log(init_scale))

    def qz(self, x, h):
        qz_mu = self.encoder(x)[0]
        qz_sigma = h.unsqueeze(1).matmul(self.logscale.exp().unsqueeze(0))

        return qz_mu, qz_sigma

    def elbo(self, x, h, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        pz_mu, pz_sigma = self.pz(x)
        qz_mu, qz_sigma = self.qz(x, h)

        kl = gaussian_diagonal_kl(qz_mu, qz_sigma, pz_mu, pz_sigma).sum()

        # z_samples is shape (num_samples, batch, z_dim).
        z_samples = qz_mu + qz_sigma * torch.randn(num_samples, *qz_mu.shape)

        log_px_z = 0
        for z in z_samples:
            log_px_z += self.loglikelihood(z, x).sum()

        log_px_z /= num_samples
        elbo = (log_px_z - kl) / x.shape[0]

        return elbo


class GMMVAE(nn.Module):

    def __init__(self, encoder, loglikelihood, z_dim, k):
        super().__init__()

        self.encoder = encoder
        self.loglikelihood = loglikelihood
        self.z_dim = z_dim

        # Initialise GMM parameters.
        self.pz_y_mu = nn.Parameter(torch.zeros((k, z_dim)),
                                    requires_grad=True)
        self.pz_y_logsigma = nn.Parameter(torch.zeros((k, z_dim)),
                                          requires_grad=True)

    def qz(self, x):
        qz_mu, qz_sigma = self.encoder(x)

        return qz_mu, qz_sigma

    def py_z(self, z, pi):
        # Compute the marginal likelihood, p(z) = \sum_k p(z|y)p(y).
        pzy = torch.zeros_like(pi)
        for k in range(pi.shape[-1]):
            pzy[k] = gaussian_diagonal_ll(z, self.pz_y_mu[k, :],
                                          self.pz_y_logsigma[k, :].exp() ** 2)
            pzy[k] += pi[:, k].log()

        pz = torch.logsumexp(pzy, dim=1)

        # Compute the posterior p(y|z) = p(z, y) / p(z)
        py_z = pzy.exp() / pz

        return py_z

    def elbo(self, x, pi, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        qz_mu, qz_sigma = self.qz(x)

        # z_samples is shape (num_samples, batch, z_dim).
        z_samples = qz_mu + qz_sigma + torch.randn(num_samples, *qz_mu.shape)

        log_px_z = 0
        kl_py_z_py = 0
        kl_qz_x_pz_y = 0
        for z in z_samples:
            log_px_z += self.loglikelihood(z, x).sum()

            py_z = self.py_z(z, pi)
            kl_py_z_py += py_z * (py_z / pi).log()

            for k in range(pi.shape[-1]):
                kl_qz_x_pz_y += py_z[:, k] * gaussian_diagonal_kl(
                    qz_mu, qz_sigma ** 2, self.pz_y_mu[k, :],
                    self.pz_y_logsigma[k, :].exp() ** 2)

        log_px_z /= num_samples
        kl_py_z_py /= num_samples
        kl_qz_x_pz_y /= num_samples
        elbo = (log_px_z - kl_py_z_py - kl_qz_x_pz_y) / x.shape[0]

        return elbo
