from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
X, y = make_friedman2(n_samples=5, noise=0, random_state=0)
print(X, y)
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel,normalize_y=True, random_state=0).fit(X, y)
print(gpr.score(X, y))
print(gpr.predict(X[:2,:], return_std=True))
