from time import time

import numpy as np

from menpo.visualize import bytes_str


def rpca_missing(X, M, lambda_=None, tol=1e-6, max_iter=1000, verbose=False):
    r"""
    Robust PCA with Missing Values using the inexact augmented Lagrange
    multiplier method.
    Parameters
    ----------
    X : ``(n_samples, n_features)`` `ndarray`
        Data matrix.
    M : ``(n_components, n_features)`` `ndarray` of type `np.bool`
        Mask matrix. For each element, if ``True`` indicates that the
        corresponding element in ``X`` is meaningful data. If ``False``
        the corresponding element in ``X`` is not considered in the
        calculation.
    lambda_ : float, optional
        The weight on sparse error term in the cost function. If ``None``,
        the heuristic value of ``1 / np.sqrt(n_samples)`` is used.
    tol : `float`, optional
        The tolerance for the stopping criterion.
    max_iter : `float`, optional
        The maximum allowed number of iterations.
    verbose : `boolean`, optional
        If ``True``, details of the progress of the algorithm will be printed
        every 10 iterations.
    Returns
    -------
    A : ``(n_samples, n_features)`` `ndarray`
        Low rank reconstruction
    E : ``(n_samples, n_features)`` `ndarray`
        Sparse reconstruction
    """
    m, n = X.shape

    if verbose:
        print('X {} of type {}: {}'.format(
            X.shape, X.dtype, bytes_str(X.nbytes)))
        # Have to allocate 7 arrays (X, Y, A, E, T, Z, V) all of this size
        # + 4 in temp computations
        print('Estimated total memory required: {}'.format(bytes_str(X.nbytes
                                                                     * 11)))
        t = time()

    if lambda_ is None:
        lambda_ = 1. / np.sqrt(m)

    norm_fro = np.linalg.norm(X, ord='fro')
    norm_two = np.linalg.norm(X, ord=2)
    norm_inf = np.linalg.norm(X.ravel(), ord=np.inf) / lambda_
    dual_norm_inv = 1.0 / max(norm_two, norm_inf)
    Y = X * dual_norm_inv

    A = np.zeros_like(X)
    notM = ~M
    mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    sv = 10
    for i in range(1, max_iter + 1):
        T = X - A + (1 / mu) * Y
        E = (np.maximum(T - (lambda_ / mu), 0) +
             np.minimum(T + (lambda_ / mu), 0))
        E = E * M + T * notM
        U, s, V = np.linalg.svd(X - E + (1 / mu) * Y, full_matrices=False)

        svp = (s > 1 / mu).sum()
        sv = min(svp + 1 if svp < sv else svp + round(0.05 * n), n)

        S_svp = np.diag(s[:svp] - 1 / mu)
        A = np.dot(U[:, :svp], np.dot(S_svp, V[:svp]))

        Z = X - A - E
        Y += mu * Z
        mu = min(mu * rho, mu_bar)

        stopping_criterion = np.linalg.norm(Z, ord='fro') / norm_fro

        if verbose and (time() - t > 1):
            print('{i:02d} ({time:.1f} sec/iter) r(A): {r_A} |E|_0: {E_0} '
                  'criterion/tol: {sc:.0f} '.format(
                i=i, r_A=np.linalg.matrix_rank(A), time=time() - t,
                E_0=(np.abs(E) > 0).sum(), sc=stopping_criterion / tol))
            t = time()
        if stopping_criterion < tol:
            if verbose:
                print('Converged after {} iterations'.format(i))
            break
    else:
        if verbose:
            print('Maximum iterations ({}) reached without '
                  'convergence'.format(max_iter))

    return A, E
