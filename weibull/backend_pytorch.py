import torch
import numpy as np


def fit(x, iters=100, eps=1e-6, use_cuda=True):
    """
    Fits a 2-parameter Weibull distribution to the given data using maximum-likelihood estimation.
    :param x: 1d-ndarray of samples from an (unknown) distribution. Each value must satisfy x > 0.
    :param iters: Maximum number of iterations
    :param eps: Stopping criterion. Fit is stopped ff the change within two iterations is smaller than eps.
    :param use_cuda: Use gpu
    :return: Tuple (Shape, Scale) which can be (NaN, NaN) if a fit is impossible.
        Impossible fits may be due to 0-values in x.
    """
    if use_cuda:
        dtype = torch.cuda.DoubleTensor
    else:
        dtype = torch.DoubleTensor
    k = 1.0
    k_t_1 = k
    xvar = torch.from_numpy(x).type(dtype)
    ln_x = torch.log(xvar)

    for t in xrange(iters):
        # Partial derivative df/dk
        x_k = xvar ** k
        x_k_ln_x = x_k * ln_x
        ff = torch.sum(x_k_ln_x)
        fg = torch.sum(x_k)
        f1 = torch.mean(ln_x)
        f = ff/fg - f1 - (1.0 / k)

        ff_prime = torch.sum(x_k_ln_x * ln_x)
        fg_prime = ff
        f_prime = (ff_prime / fg - (ff / fg * fg_prime / fg)) + (1. / (k * k))

        # Newton-Raphson method k = k - f(k;x)/f'(k;x)
        k -= f / f_prime
        # print('f=% 7.5f, dk=% 7.5f, k=% 7.5f' % (f.data[0], k.grad.data[0], k.data[0]))
        if np.isnan(f):
            return np.nan, np.nan
        if abs(k - k_t_1) < eps:
            break

        k_t_1 = k

    # Lambda (scale) can be calculated directly
    lam = torch.mean(xvar ** k) ** (1.0 / k)

    return k, lam  # Shape (SC), Scale (FE)


import matplotlib.pyplot as plt


def compare_fits(x):
    shape, scale = fit(x)
    app_shape, app_scale = x.mean() / x.std(), x.mean()
    # _, np_shape, _, np_scale = exponweib.fit(x, floc=0)

    # # Plot
    # def weib(x, n, a): # a == shape
    # 	return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)
    #
    # count, _, _ = plt.hist(x, 100)
    # xx = np.linspace(x.min(), x.max(), 10000)
    # yy = weib(xx, scale, shape)
    # yy_app = weib(xx, app_scale, app_shape)
    # yy_np = weib(xx, np_scale, np_shape)
    # plt.plot(xx, yy*(count.max() / yy.max()), label='MLE')
    # plt.plot(xx, yy_app*(count.max() / yy_app.max()), label='App')
    # plt.plot(xx, yy_np*(count.max() / yy_np.max()), label='Scipy')
    # plt.legend()
    # plt.show()

    return (shape, scale), (app_shape, app_scale)


def test_nn():
    from nnadapter.torchadapter import TorchAdapter
    from tqdm import tqdm
    import os
    from scipy.stats import exponweib
    import matplotlib.pyplot as plt

    def recursivelistims(path):
        if os.path.isfile(path):
            yield path
        else:
            for dirpath, dirnames, filenames in os.walk(path):
                for fname in filenames:
                    if fname.lower().endswith('.jpg') or fname.lower().endswith('.jpeg') or fname.lower().endswith(
                            '.png'):
                        yield os.path.join(path, dirpath, fname)

    nn = TorchAdapter('alexnet', mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]),
                      inputsize=(3, 224, 224), use_gpu=True)
    # SRC = '/media/max/Googolplex/Hydra/HydraStimuliResizedRenamed'
    SRC = '/home/max/workspace/torch/complexity_analysis/stimuli'
    # SRC = '/media/max/Googolplex/Databases/MTL/raw/val'
    # SRC = '/media/max/Googolplex/Databases/test2014'
    images = [fname for fname in recursivelistims(SRC)]
    print('Evaluating %d images' % len(images))
    batchsize = 480

    err_shape = np.zeros(len(images))
    err_scale = np.zeros(len(images))

    shapes = []
    scales = []

    for bi in tqdm(range(0, len(images), batchsize)):
        batch = nn.preprocess(images[bi:(bi + batchsize)])
        nn.forward(batch)
        lout = nn.get_layeroutput('features.0')

        # # normalize layer output via sum of abs(kernel)
        # weight, _ = nn.get_layerparams('features.0')
        # for u in xrange(weight.shape[0]):
        # 	lout[:, u] /= np.sum(np.abs(weight[u]))

        for i in tqdm(range(batch.shape[0]), leave=False):
            x = lout[i].ravel()

            x = np.abs(x)
            x = x[x > 0]  # quick fix, 0s are illegal in weibull distributions
            (mle_shape, mle_scale), (app_shape, app_scale) = compare_fits(x)
            # print('Image: %s. Shape=%7.5f, Scale=%7.5f' % (images[bi+i], mle_shape, mle_scale))
            # plt.hist(x, 100)
            # plt.show()
            # return 0

            if np.isnan(mle_shape):
                print('Shape is nan for image: %s' % images[bi + i])
                continue
            if np.isnan(mle_shape):
                print('Scale is nan for image: %s' % images[bi + i])
                continue
            if np.isnan(app_shape):
                print('Approximated shape is nan for image: %s' % images[bi + i])
                continue
            if np.isnan(app_scale):
                print('Approximated scale is nan for image: %s' % images[bi + i])
                continue

            err_shape[bi + i] = mle_shape - app_shape
            err_scale[bi + i] = mle_scale - app_scale

            shapes.append(mle_shape)
            scales.append(mle_scale)

        # print('RMSE shape: %7.5f, RMSE scale: %7.5f' % (np.sqrt(np.mean(err_shape**2)), np.sqrt(np.mean(err_scale**2))))
        # print('Max Err shape: %7.5f, Max Err scale: %7.5f' % (np.max(np.abs(err_shape)), np.max(np.abs(err_scale))))

    print('MSE shape: %7.5f, MSE scale: %7.5f' % (np.sqrt(np.mean(err_shape ** 2)), np.sqrt(np.mean(err_scale ** 2))))
    print('Max Err shape: %7.5f, Max Err scale: %7.5f' % (np.max(np.abs(err_shape)), np.max(np.abs(err_scale))))

    print('Shape range: %7.5f - %7.5f' % (np.min(shapes), np.max(shapes)))
    print('Scale range: %7.5f - %7.5f' % (np.min(scales), np.max(scales)))

    plt.plot(scales[:180], shapes[:180], 'r.')
    plt.plot(scales[180:360], shapes[180:360], 'g.')
    plt.plot(scales[360:], shapes[360:], 'b.')
    plt.xlabel('Scale (FE)')
    plt.ylabel('Shape (SC)')
    plt.xlim(0, plt.xlim()[1])
    plt.ylim(0, plt.ylim()[1])
    plt.show()


def test_rndweibull():
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    shapes = np.linspace(0.5, 20, 1000)
    scales = np.linspace(0.1, 5, 20)

    errs = np.zeros((2, shapes.shape[0], scales.shape[0]))

    for i, shape in enumerate(tqdm(shapes)):
        dist = np.random.weibull(shape, 10000)
        for j, scale in enumerate(scales):
            x = scale * dist
            x = np.abs(x)
            (mle_shape, mle_scale), (app_shape, app_scale) = compare_fits(x)

            if np.isnan(shape):
                print('Shape is nan for shape=%5.3f, scale=%5.3f' % (shape, scale))
                continue
            if np.isnan(scale):
                print('Scale is nan for shape=%5.3f, scale=%5.3f' % (shape, scale))
                continue
            if np.isnan(app_shape):
                print('Approximated shape is nan for shape=%5.3f, scale=%5.3f' % (shape, scale))
                continue
            if np.isnan(app_scale):
                print('Approximated scale is nan for shape=%5.3f, scale=%5.3f' % (shape, scale))
                continue

            errs[0, i, j] = shape - mle_shape
            errs[1, i, j] = scale - mle_scale

    print('RMSE shape: %7.5f, RMSE scale: %7.5f' % (np.sqrt(np.mean(errs[0] ** 2)), np.sqrt(np.mean(errs[1] ** 2))))
    print('Max Err shape: %7.5f, Max Err scale: %7.5f' % (np.max(np.abs(errs[0])), np.max(np.abs(errs[1]))))

    errs = np.sqrt(np.mean(errs ** 2, axis=0))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(scales, shapes)
    ax.plot_surface(X, Y, errs)
    ax.set_xlabel('Scale parameter')
    ax.set_ylabel('Shape parameter')
    ax.set_zlabel('RMSE Error')
    plt.show()


# plt.plot(shapes, scales, markersize=5.0*errs/errs.max())


if __name__ == '__main__':
    test_rndweibull()
    # test_nn()
