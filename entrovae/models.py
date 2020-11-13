import torch
import torch.nn as nn
import numpy as np

from .utils import gaussian_diagonal_kl

__all__ = ['VAE']


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
    def __init__(self, encoder, loglikelihood, latent_dim, init_scale=1.):
        super().__init__(encoder, loglikelihood, latent_dim)

        self.log_scale = nn.Parameter(torch.ones(latent_dim) *
                                      np.log(init_scale))

    def qz(self, x, h):
        qz_mu = self.encoder(x)[0]

        qz_sigma = h.unsqueeze(1).matmul(self.log_scale.exp().unsqueeze(0))

        # Reshape.
        qz_mu = qz_mu.transpose(0, 1)
        qz_sigma = qz_sigma.transpose(0, 1)

        return qz_mu, qz_sigma

    def elbo(self, x, h, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        pz_mu, pz_sigma = self.pz(x)
        qz_mu, qz_sigma = self.qz(x, h)

        kl = gaussian_diagonal_kl(qz_mu, qz_sigma, pz_mu, pz_sigma)

        # z_samples is shape (num_samples, batch, z_dim).
        z_samples = qz_mu + qz_sigma * torch.randn(num_samples, *qz_mu.shape)

        log_px_z = 0
        for z in z_samples:
            log_px_z += self.loglikelihood(z, x)

        log_px_z /= num_samples
        elbo = (log_px_z - kl) / x.shape[0]

        return elbo


