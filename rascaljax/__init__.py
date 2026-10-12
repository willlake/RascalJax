"""RascalJax: importance-sampled covariance matrices of 2-point correlation functions (RascalC in JAX)."""
from .binning import Binning, mu_bin_legendre_factors
from .catalog import Catalog, Sample, format_positions
from .driver import CorrelationInput, RawCovariance, run_cov
from .postprocess import (CovarianceTerms, compare_covariances, jackknife_data_covariance, post_process, project_joint,
                          sample_covariance)
from .xi_table import XiTable

__version__ = "0.1.0"
__all__ = ["Binning", "mu_bin_legendre_factors", "Catalog", "Sample", "format_positions", "CorrelationInput", "RawCovariance",
           "run_cov", "CovarianceTerms", "compare_covariances", "jackknife_data_covariance", "post_process", "project_joint",
           "sample_covariance", "XiTable"]
