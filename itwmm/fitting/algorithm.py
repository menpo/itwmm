from collections import defaultdict
from datetime import timedelta
from time import time

import numpy as np
from scipy import sparse as sp

from menpo.visualize import bytes_str, print_progress

from .base import gradient_xy, camera_parameters_update
from .hessian import (initialize_hessian_and_JTe, insert_frame_to_H,
                      insert_frame_to_JTe)
from .jacobian import jacobians


def increment_parameters(images, mm, id_indices, exp_indices, template_camera,
                         p, qs, cs,
                         c_f=1, c_l=1, c_id=1, c_exp=1, c_sm=1,
                         lm_group=None, n_samples=1000,
                         compute_costs=True):

    n_frames = len(images)
    n_points = mm.shape_model.template_instance.n_points
    n_p = len(id_indices)
    n_q = len(exp_indices)
    n_c = cs.shape[1] - 2  # sub one for quaternion, one for focal length

    print('Precomputing....')
    # Rescale shape components to have size:
    # n_points x (n_components * n_dims)
    # and to be scaled by the relevant standard deviation.
    shape_pc = (
            mm.shape_model.components.T *
            np.sqrt(mm.shape_model.eigenvalues)
    ).reshape([n_points, -1])
    # include std.dev in principal components
    shape_pc_lms = shape_pc.reshape([n_points, 3, -1])[mm.model_landmarks_index]

    print('Initializing Hessian/JTe for frame...')
    H, JTe = initialize_hessian_and_JTe(c_id, c_exp, c_sm, n_p, n_q, n_c, p, qs,
                                        n_frames)
    print('H: {} ({})'.format(H.shape, bytes_str(H.nbytes)))

    if compute_costs:
        costs = defaultdict(list)

    for (f, image), c, q in zip(enumerate(print_progress(
            images, prefix='Incrementing H/JTe')), cs, qs):

        # Form the overall shape parameter: [p, q]
        s = np.zeros(mm.shape_model.n_active_components)
        s[id_indices] = p
        s[exp_indices] = q

        # In our error we consider landmarks stored [x, y] - so flip here.
        lms_points_xy = image.landmarks[lm_group].points[:, [1, 0]]

        # Compute input image gradient
        grad_x, grad_y = gradient_xy(image)

        j = jacobians(s, c, image, lms_points_xy, mm, id_indices, exp_indices,
                      template_camera, grad_x, grad_y, shape_pc, shape_pc_lms,
                      n_samples, compute_costs=compute_costs)
        insert_frame_to_H(H, j, f, n_p, n_q, n_c, c_f, c_l, n_frames)
        insert_frame_to_JTe(JTe, j, f, n_p, n_q, n_c, c_f, c_l, n_frames)

        if compute_costs:
            for cost, val in j['costs'].items():
                costs[cost].append(val)

    print('Converting Hessian to sparse format')
    H = sp.csr_matrix(H)
    print("Sparsity (prop. 0's) of H: {:.2%}".format(
        1 - (H.count_nonzero() / np.prod(np.array(H.shape)))))
    print('Solving for parameter update')
    d = sp.linalg.spsolve(H, JTe)
    dp = d[:n_p]
    dqs = d[n_p:(n_p + (n_frames * n_q))].reshape([n_frames, n_q])
    dcs = d[-(n_frames * n_c):].reshape([n_frames, n_c])
    # Add the focal length and degenerate quaternion parameters back on as
    # null delta updates
    dcs = np.hstack([np.tile(np.array([0, 1]), (n_frames, 1)), dcs])

    new_p = p + dp
    new_qs = qs + dqs
    new_cs = np.array([camera_parameters_update(c, dc)
                       for c, dc in zip(cs, dcs)])

    params = {
        'p': new_p,
        'qs': new_qs,
        'cs': new_cs,
        'dp': dp,
        'dqs': dqs,
        'dcs': dcs,
    }

    if compute_costs:
        c = {k: np.array(v) for k, v in costs.items()}

        err_s_id = (p ** 2).sum()
        err_s_exp = (qs ** 2).sum()
        err_sm = ((qs[:-2] - 2 * qs[1:-1] + qs[2:]) ** 2).sum()

        err_f_tot = c['err_f'].sum() * c_f / (n_c * n_samples)
        err_l_tot = c['err_l'].sum()

        total_energy = (err_f_tot +
                        c_l * err_l_tot +
                        c_id * err_s_id +
                        c_exp * err_s_exp +
                        c_sm * err_sm)

        c['total_energy'] = total_energy
        c['err_s_id'] = (c_id, err_s_id)
        c['err_s_exp'] = (c_exp, err_s_exp)
        c['err_sm'] = (c_sm, err_sm)
        c['err_f_tot'] = err_f_tot
        c['err_l_tot'] = (c_l, err_l_tot)

        print_cost_dict(c)

        params['costs'] = c
    return params


def fit_video(images, mm, id_indices, exp_indices, template_camera,
              p, qs, cs, c_f=1, c_l=1, c_id=1, c_exp=1, c_sm=1, lm_group=None,
              n_samples=1000, n_iters=10, compute_costs=True):
    params = [
        {
            "p": p,
            "qs": qs,
            "cs": cs
        }]

    for i in range(1, n_iters + 1):
        print('{} / {}'.format(i, n_iters))
        # retrieve the last used parameters and pass them into the increment
        l = params[-1]
        t1 = time()
        incs = increment_parameters(images, mm, id_indices, exp_indices,
                                    template_camera, l['p'], l['qs'], l['cs'],
                                    c_f=c_f, c_l=c_l, c_id=c_id, c_exp=c_exp,
                                    c_sm=c_sm,
                                    lm_group=lm_group, n_samples=n_samples,
                                    compute_costs=compute_costs)
        # update the parameter list
        params.append(incs)
        # And report the time taken for the iteration.
        dt = int(time() - t1)
        print('Iteration {} complete in {}\n'.format(i, timedelta(seconds=dt)))
    return params


def fit_image(image, mm, id_indices, exp_indices, template_camera,
              p, q, c, c_f=1, c_l=1, c_id=1, c_exp=1, lm_group=None,
              n_samples=1000, n_iters=10, compute_costs=True):
    # fit image is the same as fit_video, just for a single length video.
    return fit_video(
        [image], mm, id_indices, exp_indices, template_camera,
        p, q[None, :], c[None, :], c_f=c_f, c_l=c_l,
        c_id=c_id, c_exp=c_exp, c_sm=0, lm_group=lm_group,
        n_samples=n_samples, n_iters=n_iters, compute_costs=compute_costs
    )


def print_single_cost(k, c, tot):
    if isinstance(c, tuple):
        key = '{:03.0%} | {:>12}'.format((c[0] * c[1]) / tot, k)
        val = '{:>12.2f} x {:>12.2f} = {:.2f}'.format(c[0], c[1], c[0] * c[1])
    else:
        key = '{:03.0%} | {:>12}'.format(c / tot, k)
        val = '{:.2f}'.format(c)
    print('{:>20}: {}'.format(key, val))


def print_cost_dict(d):
    print('------------------------------------------------------------------')
    print_single_cost('total_energy', d['total_energy'], d['total_energy'])
    print('------------------------------------------------------------------')
    for k in ['err_f_tot', 'err_l_tot', 'err_s_id',
              'err_s_exp', 'err_sm']:
        print_single_cost(k, d[k], d['total_energy'])
    print('------------------------------------------------------------------')
    for k in ['err_f', 'err_l']:
        print('{} (median over frames): {:.2f}'.format(k, np.median(d[k])))
    print('------------------------------------------------------------------')
