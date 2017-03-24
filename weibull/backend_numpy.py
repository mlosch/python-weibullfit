import numpy as np


def fit(x, iters=100, eps=1e-6):
    """
    Fits a 2-parameter Weibull distribution to the given data using maximum-likelihood estimation.
    :param x: 1d-ndarray of samples from an (unknown) distribution. Each value must satisfy x > 0.
    :param iters: Maximum number of iterations
    :param eps: Stopping criterion. Fit is stopped ff the change within two iterations is smaller than eps.
    :return: Tuple (Shape, Scale) which can be (NaN, NaN) if a fit is impossible.
        Impossible fits may be due to 0-values in x.
    """
    # fit k via MLE
    ln_x = np.log(x)
    k = 1.
    k_t_1 = k

    for t in xrange(iters):
        x_k = x ** k
        x_k_ln_x = x_k * ln_x
        ff = np.sum(x_k_ln_x)
        fg = np.sum(x_k)
        f = ff / fg - np.mean(ln_x) - (1. / k)

        # Calculate second derivative d^2f/dk^2
        ff_prime = np.sum(x_k_ln_x * ln_x)
        fg_prime = ff
        f_prime = (ff_prime/fg - (ff/fg * fg_prime/fg)) + (1. / (k*k))

        # Newton-Raphson method k = k - f(k;x)/f'(k;x)
        k -= f/f_prime

        if np.isnan(f):
            return np.nan, np.nan
        if abs(k - k_t_1) < eps:
            break

        k_t_1 = k

    lam = np.mean(x ** k) ** (1.0 / k)

    return k, lam


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
            # x = np.abs(x)
            mle_shape, mle_scale = fit(x)

            if np.isnan(mle_shape):
                print('Shape is nan for shape=%5.3f, scale=%5.3f' % (shape, scale))
                continue
            if np.isnan(mle_scale):
                print('Scale is nan for shape=%5.3f, scale=%5.3f' % (shape, scale))
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
    ax.set_xlabel('\lambda (Scale)')
    ax.set_ylabel('k (Shape)')
    ax.set_zlabel('RMSE Error')
    plt.show()

if __name__ == '__main__':
    test_rndweibull()
