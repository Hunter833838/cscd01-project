"""Testing for Gaussian process regression """

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Modified by: Pete Green <p.l.green@liverpool.ac.uk>
# License: BSD 3 clause

import sys
import numpy as np
import warnings

from scipy.optimize import approx_fprime

import pytest

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process.kernels import DotProduct, ExpSineSquared
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.exceptions import ConvergenceWarning

from sklearn.utils._testing \
    import (assert_array_less,
            assert_almost_equal, assert_raise_message,
            assert_array_almost_equal, assert_array_equal,
            assert_allclose, assert_warns_message)


def f(x):
    return x * np.sin(x)


X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
X2 = np.atleast_2d([2., 4., 5.5, 6.5, 7.5]).T
y = f(X).ravel()

fixed_kernel = RBF(length_scale=1.0, length_scale_bounds="fixed")
kernels = [RBF(length_scale=1.0), fixed_kernel,
           RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
           C(1.0, (1e-2, 1e2)) *
           RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
           C(1.0, (1e-2, 1e2)) *
           RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) +
           C(1e-5, (1e-5, 1e2)),
           C(0.1, (1e-2, 1e2)) *
           RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) +
           C(1e-5, (1e-5, 1e2))]
non_fixed_kernels = [kernel for kernel in kernels
                     if kernel != fixed_kernel]


@pytest.mark.parametrize('kernel', kernels)
def test_gpr_interpolation(kernel):
    if sys.maxsize <= 2 ** 32 and sys.version_info[:2] == (3, 6):
        pytest.xfail("This test may fail on 32bit Py3.6")

    # Test the interpolating property for different kernels.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    y_pred, y_cov = gpr.predict(X, return_cov=True)

    assert_almost_equal(y_pred, y)
    assert_almost_equal(np.diag(y_cov), 0.)


def test_gpr_interpolation_structured():
    # Test the interpolating property for different kernels.
    kernel = MiniSeqKernel(baseline_similarity_bounds='fixed')
    X = ['A', 'B', 'C']
    y = np.array([1, 2, 3])
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    y_pred, y_cov = gpr.predict(X, return_cov=True)

    assert_almost_equal(kernel(X, eval_gradient=True)[1].ravel(),
                        (1 - np.eye(len(X))).ravel())
    assert_almost_equal(y_pred, y)
    assert_almost_equal(np.diag(y_cov), 0.)


@pytest.mark.parametrize('kernel', non_fixed_kernels)
def test_lml_improving(kernel):
    if sys.maxsize <= 2 ** 32 and sys.version_info[:2] == (3, 6):
        pytest.xfail("This test may fail on 32bit Py3.6")

    # Test that hyperparameter-tuning improves log-marginal likelihood.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    assert (gpr.log_marginal_likelihood(gpr.kernel_.theta) >
            gpr.log_marginal_likelihood(kernel.theta))


@pytest.mark.parametrize('kernel', kernels)
def test_lml_precomputed(kernel):
    # Test that lml of optimized kernel is stored correctly.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    assert (gpr.log_marginal_likelihood(gpr.kernel_.theta) ==
            gpr.log_marginal_likelihood())


@pytest.mark.parametrize('kernel', kernels)
def test_lml_without_cloning_kernel(kernel):
    # Test that lml of optimized kernel is stored correctly.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    input_theta = np.ones(gpr.kernel_.theta.shape, dtype=np.float64)

    gpr.log_marginal_likelihood(input_theta, clone_kernel=False)
    assert_almost_equal(gpr.kernel_.theta, input_theta, 7)


@pytest.mark.parametrize('kernel', non_fixed_kernels)
def test_converged_to_local_maximum(kernel):
    # Test that we are in local maximum after hyperparameter-optimization.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

    lml, lml_gradient = \
        gpr.log_marginal_likelihood(gpr.kernel_.theta, True)

    assert np.all((np.abs(lml_gradient) < 1e-4) |
                  (gpr.kernel_.theta == gpr.kernel_.bounds[:, 0]) |
                  (gpr.kernel_.theta == gpr.kernel_.bounds[:, 1]))


@pytest.mark.parametrize('kernel', non_fixed_kernels)
def test_solution_inside_bounds(kernel):
    # Test that hyperparameter-optimization remains in bounds#
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

    bounds = gpr.kernel_.bounds
    max_ = np.finfo(gpr.kernel_.theta.dtype).max
    tiny = 1e-10
    bounds[~np.isfinite(bounds[:, 1]), 1] = max_

    assert_array_less(bounds[:, 0], gpr.kernel_.theta + tiny)
    assert_array_less(gpr.kernel_.theta, bounds[:, 1] + tiny)


@pytest.mark.parametrize('kernel', kernels)
def test_lml_gradient(kernel):
    # Compare analytic and numeric gradient of log marginal likelihood.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

    lml, lml_gradient = gpr.log_marginal_likelihood(kernel.theta, True)
    lml_gradient_approx = \
        approx_fprime(kernel.theta,
                      lambda theta: gpr.log_marginal_likelihood(theta,
                                                                False),
                      1e-10)

    assert_almost_equal(lml_gradient, lml_gradient_approx, 3)


@pytest.mark.parametrize('kernel', kernels)
def test_prior(kernel):
    # Test that GP prior has mean 0 and identical variances.
    gpr = GaussianProcessRegressor(kernel=kernel)

    y_mean, y_cov = gpr.predict(X, return_cov=True)

    assert_almost_equal(y_mean, 0, 5)
    if len(gpr.kernel.theta) > 1:
        # XXX: quite hacky, works only for current kernels
        assert_almost_equal(np.diag(y_cov), np.exp(kernel.theta[0]), 5)
    else:
        assert_almost_equal(np.diag(y_cov), 1, 5)


@pytest.mark.parametrize('kernel', kernels)
def test_sample_statistics(kernel):
    # Test that statistics of samples drawn from GP are correct.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

    y_mean, y_cov = gpr.predict(X2, return_cov=True)

    samples = gpr.sample_y(X2, 300000)

    # More digits accuracy would require many more samples
    assert_almost_equal(y_mean, np.mean(samples, 1), 1)
    assert_almost_equal(np.diag(y_cov) / np.diag(y_cov).max(),
                        np.var(samples, 1) / np.diag(y_cov).max(), 1)


def test_no_optimizer():
    # Test that kernel parameters are unmodified when optimizer is None.
    kernel = RBF(1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(X, y)
    assert np.exp(gpr.kernel_.theta) == 1.0


@pytest.mark.parametrize('kernel', kernels)
def test_predict_cov_vs_std(kernel):
    if sys.maxsize <= 2 ** 32 and sys.version_info[:2] == (3, 6):
        pytest.xfail("This test may fail on 32bit Py3.6")

    # Test that predicted std.-dev. is consistent with cov's diagonal.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    y_mean, y_cov = gpr.predict(X2, return_cov=True)
    y_mean, y_std = gpr.predict(X2, return_std=True)
    assert_almost_equal(np.sqrt(np.diag(y_cov)), y_std)


def test_anisotropic_kernel():
    # Test that GPR can identify meaningful anisotropic length-scales.
    # We learn a function which varies in one dimension ten-times slower
    # than in the other. The corresponding length-scales should differ by at
    # least a factor 5
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (50, 2))
    y = X[:, 0] + 0.1 * X[:, 1]

    kernel = RBF([1.0, 1.0])
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    assert (np.exp(gpr.kernel_.theta[1]) >
            np.exp(gpr.kernel_.theta[0]) * 5)


def test_random_starts():
    # Test that an increasing number of random-starts of GP fitting only
    # increases the log marginal likelihood of the chosen theta.
    n_samples, n_features = 25, 2
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features) * 2 - 1
    y = np.sin(X).sum(axis=1) + np.sin(3 * X).sum(axis=1) \
        + rng.normal(scale=0.1, size=n_samples)

    kernel = C(1.0, (1e-2, 1e2)) \
        * RBF(length_scale=[1.0] * n_features,
              length_scale_bounds=[(1e-4, 1e+2)] * n_features) \
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-5, 1e1))
    last_lml = -np.inf
    for n_restarts_optimizer in range(5):
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=n_restarts_optimizer,
            random_state=0,).fit(X, y)
        lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        assert lml > last_lml - np.finfo(np.float32).eps
        last_lml = lml


@pytest.mark.parametrize('kernel', kernels)
def test_y_normalization(kernel):
    """
    Test normalization of the target values in GP

    Fitting non-normalizing GP on normalized y and fitting normalizing GP
    on unnormalized y should yield identical results. Note that, here,
    'normalized y' refers to y that has been made zero mean and unit
    variance.

    """

    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std

    # Fit non-normalizing GP on normalized y
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X, y_norm)

    # Fit normalizing GP on unnormalized y
    gpr_norm = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr_norm.fit(X, y)

    # Compare predicted mean, std-devs and covariances
    y_pred, y_pred_std = gpr.predict(X2, return_std=True)
    y_pred = y_pred * y_std + y_mean
    y_pred_std = y_pred_std * y_std
    y_pred_norm, y_pred_std_norm = gpr_norm.predict(X2, return_std=True)

    assert_almost_equal(y_pred, y_pred_norm)
    assert_almost_equal(y_pred_std, y_pred_std_norm)

    _, y_cov = gpr.predict(X2, return_cov=True)
    y_cov = y_cov * y_std**2
    _, y_cov_norm = gpr_norm.predict(X2, return_cov=True)

    assert_almost_equal(y_cov, y_cov_norm)

@pytest.mark.parametrize('kernel', kernels)
def test_no_standard_deviation(kernel):
    """
    Here we test that when normalize_y=True, and when the standard
    deviation(s) of the dataset is zero, that we do not continue
    with the normalization, and that the user should be notified.
    """
    # here, we test when our dataset consists of only 1 sample. Data
    # is taken from testing/make_friedman2 regression problem.
    X = np.array([[5.48813504e+01, 1.29401721e+03, 6.02763376e-01, 6.44883183e+00]])
    y = np.array([781.91445769])
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    assert_raise_message(ValueError,
                        "error normalizing data; standard deviation "
                        "along one or more columns is equal to 0.",
                        gpr.fit, X, y)
    
    # here, we test that when our dataset has more than 1 sample, but
    # the samples have standard deviation of zero.
    X = np.array([[5.48813504e+01, 1.29401721e+03, 6.02763376e-01, 6.44883183e+00],
        [5.48813504e+01, 1.29401721e+03, 6.02763376e-01, 6.44883183e+00],
        [5.48813504e+01, 1.29401721e+03, 6.02763376e-01, 6.44883183e+00]])
    y = np.array([781.91445769,781.91445769,781.91445769])
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    assert_raise_message(ValueError,
                        "error normalizing data; standard deviation "
                        "along one or more columns is equal to 0.",
                        gpr.fit, X, y)

def test_large_variance_y():
    """
    Here we test that, when noramlize_y=True, our GP can produce a
    sensible fit to training data whose variance is significantly
    larger than unity. This test was made in response to issue #15612.

    GP predictions are verified against predictions that were made
    using GPy which, here, is treated as the 'gold standard'. Note that we
    only investigate the RBF kernel here, as that is what was used in the
    GPy implementation.

    The following code can be used to recreate the GPy data:

    --------------------------------------------------------------------------
    import GPy

    kernel_gpy = GPy.kern.RBF(input_dim=1, lengthscale=1.)
    gpy = GPy.models.GPRegression(X, np.vstack(y_large), kernel_gpy)
    gpy.optimize()
    y_pred_gpy, y_var_gpy = gpy.predict(X2)
    y_pred_std_gpy = np.sqrt(y_var_gpy)
    --------------------------------------------------------------------------
    """

    # Here we utilise a larger variance version of the training data
    y_large = 10 * y

    # Standard GP with normalize_y=True
    RBF_params = {'length_scale': 1.0}
    kernel = RBF(**RBF_params)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(X, y_large)
    y_pred, y_pred_std = gpr.predict(X2, return_std=True)

    # 'Gold standard' mean predictions from GPy
    y_pred_gpy = np.array([15.16918303,
                           -27.98707845,
                           -39.31636019,
                           14.52605515,
                           69.18503589])

    # 'Gold standard' std predictions from GPy
    y_pred_std_gpy = np.array([7.78860962,
                               3.83179178,
                               0.63149951,
                               0.52745188,
                               0.86170042])

    # Based on numerical experiments, it's reasonable to expect our
    # GP's mean predictions to get within 7% of predictions of those
    # made by GPy.
    assert_allclose(y_pred, y_pred_gpy, rtol=0.07, atol=0)

    # Based on numerical experiments, it's reasonable to expect our
    # GP's std predictions to get within 15% of predictions of those
    # made by GPy.
    assert_allclose(y_pred_std, y_pred_std_gpy, rtol=0.15, atol=0)


def test_y_multioutput():
    # Test that GPR can deal with multi-dimensional target values
    y_2d = np.vstack((y, y * 2)).T

    # Test for fixed kernel that first dimension of 2d GP equals the output
    # of 1d GP and that second dimension is twice as large
    kernel = RBF(length_scale=1.0)

    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                   normalize_y=False)
    gpr.fit(X, y)

    gpr_2d = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                      normalize_y=False)
    gpr_2d.fit(X, y_2d)

    y_pred_1d, y_std_1d = gpr.predict(X2, return_std=True)
    y_pred_2d, y_std_2d = gpr_2d.predict(X2, return_std=True)
    _, y_cov_1d = gpr.predict(X2, return_cov=True)
    _, y_cov_2d = gpr_2d.predict(X2, return_cov=True)

    assert_almost_equal(y_pred_1d, y_pred_2d[:, 0])
    assert_almost_equal(y_pred_1d, y_pred_2d[:, 1] / 2)

    # Standard deviation and covariance do not depend on output
    assert_almost_equal(y_std_1d, y_std_2d)
    assert_almost_equal(y_cov_1d, y_cov_2d)

    y_sample_1d = gpr.sample_y(X2, n_samples=10)
    y_sample_2d = gpr_2d.sample_y(X2, n_samples=10)
    assert_almost_equal(y_sample_1d, y_sample_2d[:, 0])

    # Test hyperparameter optimization
    for kernel in kernels:
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gpr.fit(X, y)

        gpr_2d = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gpr_2d.fit(X, np.vstack((y, y)).T)

        assert_almost_equal(gpr.kernel_.theta, gpr_2d.kernel_.theta, 4)


@pytest.mark.parametrize('kernel', non_fixed_kernels)
def test_custom_optimizer(kernel):
    # Test that GPR can use externally defined optimizers.
    # Define a dummy optimizer that simply tests 50 random hyperparameters
    def optimizer(obj_func, initial_theta, bounds):
        rng = np.random.RandomState(0)
        theta_opt, func_min = \
            initial_theta, obj_func(initial_theta, eval_gradient=False)
        for _ in range(50):
            theta = np.atleast_1d(rng.uniform(np.maximum(-2, bounds[:, 0]),
                                              np.minimum(1, bounds[:, 1])))
            f = obj_func(theta, eval_gradient=False)
            if f < func_min:
                theta_opt, func_min = theta, f
        return theta_opt, func_min

    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer)
    gpr.fit(X, y)
    # Checks that optimizer improved marginal likelihood
    assert (gpr.log_marginal_likelihood(gpr.kernel_.theta) >
            gpr.log_marginal_likelihood(gpr.kernel.theta))


def test_gpr_correct_error_message():
    X = np.arange(12).reshape(6, -1)
    y = np.ones(6)
    kernel = DotProduct()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    assert_raise_message(np.linalg.LinAlgError,
                         "The kernel, %s, is not returning a "
                         "positive definite matrix. Try gradually increasing "
                         "the 'alpha' parameter of your "
                         "GaussianProcessRegressor estimator."
                         % kernel, gpr.fit, X, y)


@pytest.mark.parametrize('kernel', kernels)
def test_duplicate_input(kernel):
    # Test GPR can handle two different output-values for the same input.
    gpr_equal_inputs = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
    gpr_similar_inputs = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)

    X_ = np.vstack((X, X[0]))
    y_ = np.hstack((y, y[0] + 1))
    gpr_equal_inputs.fit(X_, y_)

    X_ = np.vstack((X, X[0] + 1e-15))
    y_ = np.hstack((y, y[0] + 1))
    gpr_similar_inputs.fit(X_, y_)

    X_test = np.linspace(0, 10, 100)[:, None]
    y_pred_equal, y_std_equal = \
        gpr_equal_inputs.predict(X_test, return_std=True)
    y_pred_similar, y_std_similar = \
        gpr_similar_inputs.predict(X_test, return_std=True)

    assert_almost_equal(y_pred_equal, y_pred_similar)
    assert_almost_equal(y_std_equal, y_std_similar)


def test_no_fit_default_predict():
    # Test that GPR predictions without fit does not break by default.
    default_kernel = (C(1.0, constant_value_bounds="fixed") *
                      RBF(1.0, length_scale_bounds="fixed"))
    gpr1 = GaussianProcessRegressor()
    _, y_std1 = gpr1.predict(X, return_std=True)
    _, y_cov1 = gpr1.predict(X, return_cov=True)

    gpr2 = GaussianProcessRegressor(kernel=default_kernel)
    _, y_std2 = gpr2.predict(X, return_std=True)
    _, y_cov2 = gpr2.predict(X, return_cov=True)

    assert_array_almost_equal(y_std1, y_std2)
    assert_array_almost_equal(y_cov1, y_cov2)


@pytest.mark.parametrize('kernel', kernels)
def test_K_inv_reset(kernel):
    y2 = f(X2).ravel()

    # Test that self._K_inv is reset after a new fit
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    assert hasattr(gpr, '_K_inv')
    assert gpr._K_inv is None
    gpr.predict(X, return_std=True)
    assert gpr._K_inv is not None
    gpr.fit(X2, y2)
    assert gpr._K_inv is None
    gpr.predict(X2, return_std=True)
    gpr2 = GaussianProcessRegressor(kernel=kernel).fit(X2, y2)
    gpr2.predict(X2, return_std=True)
    # the value of K_inv should be independent of the first fit
    assert_array_equal(gpr._K_inv, gpr2._K_inv)


def test_warning_bounds():
    kernel = RBF(length_scale_bounds=[1e-5, 1e-3])
    gpr = GaussianProcessRegressor(kernel=kernel)
    assert_warns_message(ConvergenceWarning, "The optimal value found for "
                                             "dimension 0 of parameter "
                                             "length_scale is close to "
                                             "the specified upper bound "
                                             "0.001. Increasing the bound "
                                             "and calling fit again may "
                                             "find a better value.",
                         gpr.fit, X, y)

    kernel_sum = (WhiteKernel(noise_level_bounds=[1e-5, 1e-3]) +
                  RBF(length_scale_bounds=[1e3, 1e5]))
    gpr_sum = GaussianProcessRegressor(kernel=kernel_sum)
    with pytest.warns(None) as record:
        with warnings.catch_warnings():
            # scipy 1.3.0 uses tostring which is deprecated in numpy
            warnings.filterwarnings("ignore", "tostring", DeprecationWarning)
            gpr_sum.fit(X, y)

    assert len(record) == 2
    assert record[0].message.args[0] == ("The optimal value found for "
                                         "dimension 0 of parameter "
                                         "k1__noise_level is close to the "
                                         "specified upper bound 0.001. "
                                         "Increasing the bound and calling "
                                         "fit again may find a better value.")

    assert record[1].message.args[0] == ("The optimal value found for "
                                         "dimension 0 of parameter "
                                         "k2__length_scale is close to the "
                                         "specified lower bound 1000.0. "
                                         "Decreasing the bound and calling "
                                         "fit again may find a better value.")

    X_tile = np.tile(X, 2)
    kernel_dims = RBF(length_scale=[1., 2.],
                      length_scale_bounds=[1e1, 1e2])
    gpr_dims = GaussianProcessRegressor(kernel=kernel_dims)

    with pytest.warns(None) as record:
        with warnings.catch_warnings():
            # scipy 1.3.0 uses tostring which is deprecated in numpy
            warnings.filterwarnings("ignore", "tostring", DeprecationWarning)
            gpr_dims.fit(X_tile, y)

    assert len(record) == 2
    assert record[0].message.args[0] == ("The optimal value found for "
                                         "dimension 0 of parameter "
                                         "length_scale is close to the "
                                         "specified lower bound 10.0. "
                                         "Decreasing the bound and calling "
                                         "fit again may find a better value.")

    assert record[1].message.args[0] == ("The optimal value found for "
                                         "dimension 1 of parameter "
                                         "length_scale is close to the "
                                         "specified lower bound 10.0. "
                                         "Decreasing the bound and calling "
                                         "fit again may find a better value.")


def test_bound_check_fixed_hyperparameter():
    # Regression test for issue #17943
    # Check that having a hyperparameter with fixed bounds doesn't cause an
    # error
    k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
    k2 = ExpSineSquared(length_scale=1.0, periodicity=1.0,
                        periodicity_bounds="fixed")  # seasonal component
    kernel = k1 + k2
    GaussianProcessRegressor(kernel=kernel).fit(X, y)
