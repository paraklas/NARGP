import GPy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as ml
import matplotlib.patches as mpatches
import seaborn as sns

import time

np.random.seed(11)

def ismember_rows(a, b):
    '''Equivalent of 'ismember' from Matlab
    a.shape = (nRows_a, nCol)
    b.shape = (nRows_b, nCol)
    return the idx where b[idx] == a
    '''
    return np.nonzero(np.all(b == a[:,np.newaxis], axis=2))[1]

# dir = "/Users/Paris/Desktop/Research/MixedCondunction/data/"
# x1 = np.loadtxt(dir + "X_train_L1.txt")
# x2 = np.loadtxt(dir + "X_train_L2.txt")
# y1 = np.loadtxt(dir + "Y_train_L1.txt")
# y2 = np.loadtxt(dir + "Y_train_L2.txt")

# ''' Create training set '''
# N1 = 500
# N2 = 280

# # randomly select high fidelity points
# perm = np.random.permutation(x2.shape[0])
# X2 = x2[perm[0:N2], :]
# Y2 = y2[perm[0:N2]][:,None]
# Xtest = x2[perm[N2:], :]
# Exact = y2[perm[N2:]][:,None]

# # Create nested low-fidelity training set
# ib = ismember_rows(X2, x1)
# nn = ib.shape[0]
# X1 = x1[ib,:]
# Y1 = y1[ib][:,None]
# ia = np.arange(x1.shape[0])
# idx = np.setdiff1d(ia, ib)
# X1 = np.vstack((X1, x1[idx[0:N1-nn], :]))
# Y1 = np.vstack((Y1, y1[idx[0:N1-nn]][:,None]))

dir = "/Users/Paris/Desktop/Research/MixedCondunction/data/tmp/"
X1 = np.loadtxt(dir + "X_train_L1.txt")
X2 = np.loadtxt(dir + "X_train_L2.txt")
Xtest = np.loadtxt(dir + "X_test.txt")

Y1 = np.loadtxt(dir + "Y_train_L1.txt")[:,None]
Y2 = np.loadtxt(dir + "Y_train_L2.txt")[:,None]
Exact = np.loadtxt(dir + "Y_test.txt")[:,None]

N1 = X1.shape[0]
N2 = X2.shape[0]

plot = 0
save = 1

dim = 3
lb = np.array([0.0, 0.0, 0.0])
ub = np.array([100.0, 1.0, 180.0])


nn = 40
x1 = np.linspace(lb[0], ub[0], nn)
x2 = np.linspace(lb[1], ub[1], nn)
X, Y = np.meshgrid(x1, x2)

active_dimensions = np.arange(0,dim)

start = time.time()

''' Train level 1 '''
k1 = GPy.kern.RBF(dim, ARD = True)
m1 = GPy.models.GPRegression(X=X1, Y=Y1, kernel=k1)

m1[".*Gaussian_noise"] = m1.Y.var()*0.01
m1[".*Gaussian_noise"].fix()

m1.optimize(max_iters = 500)

m1[".*Gaussian_noise"].unfix()
m1[".*Gaussian_noise"].constrain_positive()

m1.optimize_restarts(30, optimizer = "bfgs",  max_iters = 1000)

mu1, v1 = m1.predict(X2)

''' Train level 2 '''
XX = np.hstack((X2, mu1))

k2 = GPy.kern.RBF(1, active_dims = [dim])*GPy.kern.RBF(dim, active_dims = active_dimensions, ARD = True) \
    + GPy.kern.RBF(dim, active_dims = active_dimensions, ARD = True)

m2 = GPy.models.GPRegression(X=XX, Y=Y2, kernel=k2)

m2[".*Gaussian_noise"] = m2.Y.var()*0.01
m2[".*Gaussian_noise"].fix()

m2.optimize(max_iters = 500)

m2[".*Gaussian_noise"].unfix()
m2[".*Gaussian_noise"].constrain_positive()

m2.optimize_restarts(30, optimizer = "bfgs",  max_iters = 1000)

end = time.time()
print "Training AR-GP done in %f seconds" % (end - start)

# Predict at test points
nsamples = 1000
ntest = Xtest.shape[0]
mu0, C0 = m1.predict(Xtest, full_cov=True)
Z = np.random.multivariate_normal(mu0.flatten(),C0,nsamples)
tmp_m = np.zeros((nsamples,ntest))
tmp_v = np.zeros((nsamples,ntest))

# push samples through f_2
for i in range(0,nsamples):
    mu, v = m2.predict(np.hstack((Xtest, Z[i,:][:,None])))
    tmp_m[i,:] = mu.flatten()
    tmp_v[i,:] = v.flatten()

# get mean and variance at X3
mu2 = np.mean(tmp_m, axis = 0)
v2 = np.mean(tmp_v, axis = 0) + np.var(tmp_m, axis = 0)
mu2 = mu2[:,None]
v2 = np.abs(v2[:,None])



start = time.time()

''' Standard GP '''
k4 = GPy.kern.RBF(dim, ARD = True)
m4 = GPy.models.GPRegression(X=X2, Y=Y2, kernel=k4)

m4[".*Gaussian_noise"] = m4.Y.var()*0.01
m4[".*Gaussian_noise"].fix()

m4.optimize(max_iters = 500)

m4.optimize_restarts(10, optimizer = "bfgs",  max_iters = 1000)

end = time.time()
print "Training GP in %f seconds" % (end - start)

mu4, v4 = m4.predict(Xtest)

error = np.linalg.norm(Exact - mu2)/np.linalg.norm(Exact)
gpe = np.linalg.norm(Exact - mu4)/np.linalg.norm(Exact)

print "N1 = %d, N2 = %d, error = %e, GP error = %e" % (N1, N2, error, gpe)


if (plot == 1):
    plt.close('all')

    # Scater plot
    sns.set(style="white", palette="muted", color_codes=True)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    mmin = np.min(Exact);
    mmax = np.max(Exact);
    line = np.linspace(mmin, mmax, 10);

    ax.plot(line, line, 'k--', label = 'Exact')
    plt.scatter(Exact, mu2, facecolors='none', edgecolors='k', label = 'AR-GP')
    plt.scatter(Exact, mu4, facecolors='none', edgecolors='r', label = 'GP')

    plt.legend(numpoints=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(mmin, mmax)
    plt.ylim(mmin, mmax)

    ''' Density plot '''
    sns.set(style="white", palette="muted", color_codes=True)

    fig = plt.figure(2)
    ax = fig.add_subplot(111)

    # Plot a filled kernel density estimate of the test observation
    sns.distplot(Exact, hist=False, color="b", kde_kws={"shade": True}, label = 'Exact')
    # Plot a kernel density estimate and rug plot of the predictions
    sns.distplot(mu2, hist=False, color="r", label = 'AR-GP')
    sns.distplot(mu4, hist=False, color="g", label = 'GP')
    sns.distplot(Y2, kde = False, hist=False, rug = True, color="r")

    plt.legend(numpoints=1)

    # Tail Density plot
    # sns.set(style="white", palette="muted", color_codes=True)

    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    ax.set(yscale="log")
    # Plot a filled kernel density estimate of the test observation
    sns.distplot(Exact, hist=False, color="b", label = 'Exact')
    # Plot a kernel density estimate and rug plot of the predictions
    sns.distplot(mu2, hist=False, color="r", label = 'AR-GP')
    sns.distplot(mu4, hist=False, color="g", label = 'GP')
    sns.distplot(Y2, kde = False, hist=False, rug = True, color="r")
    plt.grid()
    plt.legend(numpoints=1)


if (save == 1):
            np.savetxt("conduction_X1.txt", X1)
            np.savetxt("conduction_X2.txt", X2)
            np.savetxt("conduction_Y1.txt", Y1)
            np.savetxt("conduction_Y2.txt", Y2)
            np.savetxt("conduction_ARGP_mean.txt", mu2)
            np.savetxt("conduction_ARGP_var.txt", v2)
            np.savetxt("conduction_GP_mu4.txt", mu4)
            np.savetxt("conduction_GP_var.txt", v4)
            np.savetxt("conduction_Exact.txt", Exact)
            np.savetxt("conduction_Xtest.txt", Xtest)
            np.savetxt("conduction_lb.txt", lb)
            np.savetxt("conduction_ub.txt", lb)
# error = np.linalg.norm(Exact - mu3)/np.linalg.norm(Exact)
# print "error = %e" % (error)

plt.show()
