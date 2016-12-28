import GPy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as ml
import matplotlib.patches as mpatches

import time

np.random.seed(11)

def high(x):
    x1 = x[:,0]
    x2 = x[:,1]
    return (-1.275*x1**2 / np.pi**2 + 5.0*x1/np.pi + x2 - 6.0)**2 + (10.0 - 5.0/(4.0*np.pi))*np.cos(x1) + 10.0

def medium(x):
    x1 = x[:,0]
    x2 = x[:,1]
    return 10.0*np.sqrt(high(x-2.0)) + 2.0*(x1-0.5)-3.0*(3.0*x2-1.0) - 1.0

def low(x):
    x1 = x[:,0]
    x2 = x[:,1]
    return medium(1.2*(x+2.0)) - 3.0*x2 + 1.0

def scale_range(x,ub,lb):
    Np = x.shape[0]
    dim = x.shape[1]
    for i in range(0,Np):
        for j in range(0,dim):
            tmp = ub[j] -lb[j]
            x[i][j] = tmp*x[i][j] + lb[j]
    return x

def rmse(pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()
    return np.sqrt(np.mean((pred-truth)**2))


''' Create training set '''
N1 = 80
N2 = 40
N3 = 20

plot = 1
save = 0

dim = 2
lb = np.array([-5.0, 0.0])
ub = np.array([10.0, 15.0])

tmp = np.random.rand(1000,dim)
Xtrain = scale_range(tmp,ub,lb)
idx = np.random.permutation(1000)
X1 = Xtrain[idx[0:N1], :]
X2 = Xtrain[idx[0:N2], :]
X3 = Xtrain[idx[0:N3], :]

Y1 = low(X1)[:,None]
Y2 = medium(X2)[:,None]
Y3 = high(X3)[:,None]

nn = 40
lb = np.array([-5.0, 0.0])
ub = np.array([10.0, 15.0])
x1 = np.linspace(lb[0], ub[0], 50)
x2 = np.linspace(lb[1], ub[1], 50)
X, Y = np.meshgrid(x1, x2)

tmp = np.random.rand(1000,2)
Xtest = scale_range(tmp,ub,lb)

Exact = high(Xtest)
Medium = medium(Xtest)
Low = low(Xtest)

Exactplot = ml.griddata(Xtest[:,0],Xtest[:,1], Exact, X, Y, interp = 'linear')
Medplot = ml.griddata(Xtest[:,0],Xtest[:,1], Medium, X, Y, interp = 'linear')
Lowplot = ml.griddata(Xtest[:,0],Xtest[:,1], Low, X, Y, interp = 'linear')

active_dimensions = np.arange(0,dim)

# if plot == 1:
#     fig = plt.figure(1)
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Exactplot, color = '#377eb8', rstride=2, cstride=2,
#                                     linewidth=0, antialiased=True, shade = True, alpha = 0.6)
#     ax.plot_surface(X, Y, Medplot, color = 'magenta', rstride=2, cstride=2,
#                                     linewidth=0, antialiased=True, shade = True, alpha = 0.6)
#     ax.plot_surface(X, Y, Lowplot, color = 'green', rstride=2, cstride=2,
#                                     linewidth=0, antialiased=True, shade = True, alpha = 0.6)

#     fig = plt.figure(2)
#     plt.pcolor(X, Y, Exactplot, cmap='jet')
#     plt.colorbar()

#     fig = plt.figure(3)
#     plt.plot(Low,Exact,'.')

#     fig = plt.figure(4)
#     plt.plot(Medium,Exact,'.')

#     fig = plt.figure(5)
#     plt.plot(Low,Medium,'.')

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


# Prepare for level 3: sample f_1 at X3
nsamples = 100
ntest = X3.shape[0]
mu0, C0 = m1.predict(X3, full_cov=True)
Z = np.random.multivariate_normal(mu0.flatten(),C0,nsamples)
tmp_m = np.zeros((nsamples,ntest))
tmp_v = np.zeros((nsamples,ntest))

# push samples through f_2
for i in range(0,nsamples):
    mu, v = m2.predict(np.hstack((X3, Z[i,:][:,None])))
    tmp_m[i,:] = mu.flatten()
    tmp_v[i,:] = v.flatten()

# get mean and variance at X3
mu2 = np.mean(tmp_m, axis = 0)
v2 = np.mean(tmp_v, axis = 0) + np.var(tmp_m, axis = 0)
mu2 = mu2[:,None]
v3 = np.abs(v2[:,None])


''' Train level 3 '''
XX = np.hstack((X3, mu2))

k3 = GPy.kern.RBF(1, active_dims = [dim])*GPy.kern.RBF(dim, active_dims = active_dimensions, ARD = True) \
    + GPy.kern.RBF(dim, active_dims = active_dimensions, ARD = True)

m3 = GPy.models.GPRegression(X=XX, Y=Y3, kernel=k3)

m3[".*Gaussian_noise"] = m3.Y.var()*0.01
m3[".*Gaussian_noise"].fix()

m3.optimize(max_iters = 500)

m3[".*Gaussian_noise"].unfix()
m3[".*Gaussian_noise"].constrain_positive()

m3.optimize_restarts(30, optimizer = "bfgs",  max_iters = 1000)

end = time.time()
print "Training done in %f seconds" % (end - start)

# Compute posterior mean and variance for level 3 evaluated at the test points

# sample f_1 at Xtest
nsamples = 100
ntest = Xtest.shape[0]
mu0, C0 = m1.predict(Xtest, full_cov=True)
Z = np.random.multivariate_normal(mu0.flatten(),C0,nsamples)

# push samples through f_2 and f_3
tmp_m = np.zeros((nsamples**2,ntest))
tmp_v = np.zeros((nsamples**2,ntest))
cnt = 0
for i in range(0,nsamples):
    mu, C = m2.predict(np.hstack((Xtest, Z[i,:][:,None])), full_cov=True)
    Q = np.random.multivariate_normal(mu.flatten(),C,nsamples)
    for j in range(0,nsamples):
        mu, v = m3.predict(np.hstack((Xtest, Q[j,:][:,None])))
        tmp_m[cnt,:] = mu.flatten()
        tmp_v[cnt,:] = v.flatten()
        cnt = cnt + 1


# get f_2 posterior mean and variance at Xtest
mu3 = np.mean(tmp_m, axis = 0)
v3 = np.mean(tmp_v, axis = 0) + np.var(tmp_m, axis = 0)
mu3 = mu3[:,None]
v3 = np.abs(v3[:,None])

start = time.time()

''' Standard GP '''
k4 = GPy.kern.RBF(dim, ARD = True)
m4 = GPy.models.GPRegression(X=X2, Y=Y2, kernel=k4)

m4[".*Gaussian_noise"] = m4.Y.var()*0.01
m4[".*Gaussian_noise"].fix()

m4.optimize(max_iters = 500)

# m4[".*Gaussian_noise"].unfix()
# m4[".*Gaussian_noise"].constrain_positive()

m4.optimize_restarts(10, optimizer = "bfgs",  max_iters = 1000)

end = time.time()
print "Training GP in %f seconds" % (end - start)

mu4, v4 = m4.predict(Xtest)

Exact = Exact[:,None]
error = np.linalg.norm(Exact - mu3)/np.linalg.norm(Exact)
gpe = np.linalg.norm(Exact - mu4)/np.linalg.norm(Exact)

print "N1 = %d, N2 = %d, N3 = %d, error = %e, GP error = %e" % (N1, N2, N3, error, gpe)


if plot == 1:
    Predplot = ml.griddata(Xtest[:,0],Xtest[:,1], mu3.flatten(), X, Y, interp = 'linear')
    Varplot = ml.griddata(Xtest[:,0],Xtest[:,1], v3.flatten(), X, Y, interp = 'linear')
    GPplot = ml.griddata(Xtest[:,0],Xtest[:,1], mu4.flatten(), X, Y, interp = 'linear')

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Exactplot, color = '#377eb8', rstride=2, cstride=2,
                                    linewidth=0, antialiased=True, shade = True, alpha = 0.6)
    ax.plot_surface(X, Y, Medplot, color = 'magenta', rstride=2, cstride=2,
                                    linewidth=0, antialiased=True, shade = True, alpha = 0.6)
    ax.plot_surface(X, Y, Lowplot, color = 'green', rstride=2, cstride=2,
                                    linewidth=0, antialiased=True, shade = True, alpha = 0.6)
    ax.plot_surface(X, Y, Predplot, color = 'red', rstride=2, cstride=2,
                                    linewidth=0, antialiased=True, shade = True, alpha = 0.6)

    fig = plt.figure(2)
    plt.pcolor(X, Y, Exactplot, cmap='jet')
    plt.colorbar()

    fig = plt.figure(3)
    plt.pcolor(X, Y, Predplot, cmap='jet')
    plt.plot(X2[:,0], X2[:,1], marker='o', linestyle = '')
    plt.colorbar()

    fig = plt.figure(4)
    plt.pcolor(X, Y, GPplot, cmap='jet')
    plt.plot(X2[:,0], X2[:,1], marker='o', linestyle = '')
    plt.colorbar()

    fig = plt.figure(5)
    plt.pcolor(X, Y, Varplot, cmap='jet')
    plt.colorbar()

    fig = plt.figure(6)
    plt.plot(Low, Exact, '.', label = "Exact correlation")
    plt.plot(mu0, mu3, '.', label = "Predicted correlation")
    plt.legend()

if (save == 1):
            np.savetxt("Branin_X1.txt", X1)
            np.savetxt("Branin_X2.txt", X2)
            np.savetxt("Branin_X3.txt", X3)
            np.savetxt("Branin_Y1.txt", Y1)
            np.savetxt("Branin_Y2.txt", Y2)
            np.savetxt("Branin_Y3.txt", Y3)
            np.savetxt("Branin_ARGP_mean.txt", mu3)
            np.savetxt("Branin_ARGP_var.txt", v3)
            np.savetxt("Branin_GP_mu4.txt", mu4)
            np.savetxt("Branin_GP_var.txt", v4)
            np.savetxt("Branin_Exact.txt", Exact)
            np.savetxt("Branin_Medium.txt", Medium)
            np.savetxt("Branin_Low.txt", Low)

# error = np.linalg.norm(Exact - mu3)/np.linalg.norm(Exact)
# print "error = %e" % (error)

plt.show()
