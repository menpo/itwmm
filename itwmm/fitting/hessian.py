import numpy as np


def smoothing_kernel(n_frames):
    A = np.eye(n_frames - 2, n_frames)
    i, j = np.diag_indices(n_frames - 2)

    A[i, j + 1] = -2
    A[i, j + 2] = 1
    return A


# -------------------------- HESSIAN INITIALIZATION ------------------------- #
# Functions needed to construct the per-iteration ITW-V video Hessian object.
def insert_id_constraint(H, c_id, n_p):
    H_id = H[:n_p, :n_p]
    np.fill_diagonal(H_id, np.diag(H_id) + c_id)


def insert_exp_constraint(H, c_exp, n_p, n_q, n_frames):
    i_offset = n_p
    j_offset = n_p

    size = n_q * n_frames
    exp_const = np.diag(np.repeat(c_exp, size))
    H[i_offset:i_offset + size, j_offset:j_offset + size] = exp_const


def insert_smoothness_constraint(H, c_sm, n_p, n_q, n_frames):
    s = smoothing_kernel(n_frames)
    s_h = s.T.dot(s)

    B = np.eye(n_q) * c_sm
    smoothing = np.kron(s_h, B)
    H[n_p:(n_p + n_q * n_frames), n_p:(n_p + n_q * n_frames)] += smoothing


# ----------------------- J.T.dot(e) INITIALIZATION  ------------------------- #
def insert_id_constraint_to_JTe(JTe, p, c_id, n_p):
    JTe[:n_p] += -c_id * p


def insert_exp_constraint_to_JTe(JTe, qs, c_exp, n_p, n_q, n_frames):
    size = n_q * n_frames
    JTe[n_p:n_p + size] += - c_exp * qs.ravel()


def insert_smoothness_constraint_to_JTe(JTe, qs, c_sm, n_p, n_q, n_frames):
    s = smoothing_kernel(n_frames)
    # Analytically we can observe that the pattern of selection of terms in
    # JTe for the smoothing is the 2nd order central difference scheme formed
    # in the Hessian (s.T.dot(s)) - form this matrix here to select the qs as
    #  we want
    s_h = s.T.dot(s)
    # n_frames x n_frames @ n_frames x n_q = n_frames x n_q
    # ravelling this back preserves per-frame ordering of q in JTe.
    JTe_smoothing = s_h.dot(qs).ravel()
    JTe[n_p:n_p + n_frames * n_q] += - c_sm * JTe_smoothing


# -------------------------- TOTAL INITIALIZATION ---------------------------- #
def initialize_hessian_and_JTe(c_id, c_exp, c_sm, n_p, n_q, n_c, p, qs,
                               n_frames):
    n = n_p + n_frames * n_q + n_frames * n_c
    H = np.zeros((n, n))
    insert_id_constraint(H, c_id, n_p)
    insert_exp_constraint(H, c_exp, n_p, n_q, n_frames)
    if n_frames >= 3:
        insert_smoothness_constraint(H, c_sm, n_p, n_q, n_frames)
    else:
        print('This video has only {} frames, cannot include smoothing term '
              'in H/JTe.'
              .format(n_frames))

    # The J.T.dot(e) term is always the size of the Hessian w/h (n total params)
    JTe = np.zeros(n)
    insert_id_constraint_to_JTe(JTe, p, c_id, n_p)
    insert_exp_constraint_to_JTe(JTe, qs, c_exp, n_p, n_q, n_frames)


    if n_frames >= 3:
        insert_smoothness_constraint_to_JTe(JTe, qs, c_sm, n_p, n_q, n_frames)
    return H, JTe


# -------------------------- PER-FRAME UPDATES  ------------------------------ #

def insert_frame_to_H(H, j, f, n_p, n_q, n_c, c_f, c_l, n_frames):

    # normalize data term contributions by n_channels x n_pixels
    # otherwise scales of all other constants need to be n_channels x n_pixels
    # to compensate
    c_f = c_f / j['J_f_p'].shape[0]

    pp = c_f * j['J_f_p'].T @ j['J_f_p'] + c_l * j['J_l_p'].T @ j['J_l_p']
    qq = c_f * j['J_f_q'].T @ j['J_f_q'] + c_l * j['J_l_q'].T @ j['J_l_q']
    cc = c_f * j['J_f_c'].T @ j['J_f_c'] + c_l * j['J_l_c'].T @ j['J_l_c']

    pq = c_f * j['J_f_p'].T @ j['J_f_q'] + c_l * j['J_l_p'].T @ j['J_l_q']
    pc = c_f * j['J_f_p'].T @ j['J_f_c'] + c_l * j['J_l_p'].T @ j['J_l_c']

    qc = c_f * j['J_f_q'].T @ j['J_f_c'] + c_l * j['J_l_q'].T @ j['J_l_c']

    # Find the right offset into the Hessian for Q/C (which are per-frame
    # terms). Note that p terms always sum, so no per-frame offset needed.
    offset_q = n_p + f * n_q
    offset_c = n_p + n_frames * n_q + f * n_c

    # 1. Update the terms along the block diagonal of the Hessian
    H[:n_p, :n_p] += pp
    H[offset_q:offset_q + n_q, offset_q:offset_q + n_q] += qq
    H[offset_c:offset_c + n_c, offset_c:offset_c + n_c] += cc

    # 2. Update terms at the off diagonal. Each time we can immediately fill
    # in the symmetric part of the term (as we know the Hessian is symmetric).
    H[:n_p, offset_q:offset_q + n_q] += pq
    H[offset_q:offset_q + n_q, :n_p] += pq.T

    H[:n_p, offset_c:offset_c + n_c] += pc
    H[offset_c:offset_c + n_c, :n_p] += pc.T

    H[offset_q:offset_q + n_q, offset_c:offset_c + n_c] += qc
    H[offset_c:offset_c + n_c, offset_q:offset_q + n_q] += qc.T


def insert_frame_to_JTe(JTe, j, f, n_p, n_q, n_c, c_f, c_l, n_frames):

    # normalize data term contributions by n_channels x n_pixels
    # otherwise scales of all other constants need to be n_channels x n_pixels
    # to compensate
    c_f = c_f / j['J_f_p'].shape[0]

    JTe_p = c_f * j['J_f_p'].T @ j['e_f'] + c_l * j['J_l_p'].T @ j['e_l']
    JTe_q = c_f * j['J_f_q'].T @ j['e_f'] + c_l * j['J_l_q'].T @ j['e_l']
    JTe_c = c_f * j['J_f_c'].T @ j['e_f'] + c_l * j['J_l_c'].T @ j['e_l']

    JTe[:n_p] += JTe_p

    offset_q = n_p + f * n_q
    JTe[offset_q:offset_q + n_q] += JTe_q

    offset_c = n_p + n_frames * n_q + f * n_c
    JTe[offset_c:offset_c + n_c] += JTe_c
