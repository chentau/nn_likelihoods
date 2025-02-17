import numpy as np
import scipy as scp
import ddm_data_simulation as ddm_data_simulator
import scipy.integrate as integrate

from libc.math cimport sin, exp, sqrt, M_PI, fmax, fmin, log, fabs

# WFPT NAVARROS FUSS -------------
# Large-time approximation to fpt distribution
cdef double fptd_large(double t, double w, int k):
    cdef double fptd_sum = 0
    cdef int i
    for i in range(1, k + 1):
        fptd_sum += i * exp( - ((i**2) * (M_PI ** 2) * t) / 2) * sin(i * M_PI * w)
    return fptd_sum * M_PI

# Small-time approximation to fpt distribution
cdef double fptd_small(double t, double w, int k):
    cdef double temp = (k - 1) / 2
    cdef int floor = np.floor(temp)
    cdef int ceiling = - np.ceil(temp)
    cdef double fptd_sum = 0
    cdef int i

    for i in range(ceiling, floor + 1, 1):
        fptd_sum += (w + 2 * i) * exp(- ((w + 2 * i)**2) / (2 * t))
    return fptd_sum * (1 / sqrt(2 * M_PI * (t**3)))

# Leading term (shows up for both large and small time)
cdef double calculate_leading_term(double t, double v, double a, double w):
    return 1 / (a**2) * exp( - (v * a * w) - (((v**2) * t) / 2))

# Choice function to determine which approximation is appropriate (small or large time)
cdef choice_function(double t, double eps):
    eps_l = fmin(eps, 1 / (t * M_PI))
    eps_s = fmin(eps, 1 / (2 * sqrt(2 * M_PI * t)))
    k_l = np.ceil(fmax(sqrt( - (2 * log(M_PI * t * eps_l)) / (M_PI**2 * t)), 1 / (M_PI * sqrt(t))))
    k_s = np.ceil(fmax(2 + sqrt( - 2 * t * log(2 * eps_s * sqrt(2 * M_PI * t))), 1 + sqrt(t)))
    return k_s - k_l, k_l, k_s

# Actual fptd (first-passage-time-distribution) algorithm
def fptd(t, v, a, w, eps=1e-10):
    # negative reaction times signify upper boundary crossing
    # we have to change the parameters as suggested by navarro & fuss (2009)
    if t < 0:
        v = - v
        w = 1 - w
        t = np.abs(t)

    #print('lambda: ' + str(sgn_lambda))
    #print('k_s: ' + str(k_s))
    if t != 0:
        leading_term = calculate_leading_term(t, v, a, w)
        t_adj = t / (a**2)
        sgn_lambda, k_l, k_s = choice_function(t_adj, eps)
        if sgn_lambda >= 0:
            return max(eps, leading_term * fptd_large(t_adj, w, k_l))
        else:
            return max(eps, leading_term * fptd_small(t_adj, w, k_s))
    else:
        return 1e-29
# --------------------------------

# def batch_fptd(t, double v, double a, double w, double eps=1e-10):
#     # Use when rts and choices vary, but parameters are held constant
#     cdef int i
#     cdef double[:] t_view = t
#     cdef int n = t.shape[0]

#     likelihoods = np.zeros(n)
#     cdef double[:] likelihoods_view = likelihoods

#     for i in range(n):
#         if t_view[i] == 0:
#             likelihoods_view[i] = 1e-48
#         elif t_view[i] < 0:
#             t = fabs(t)

#             leading_term = calculate_leading_term(t_view[i], (-1) * v, a, 1 - w)
#             sgn_lambda, k_l, k_s = choice_function(t_view[i], eps)
#             if sgn_lambda >= 0:
#                 likelihoods_view[i] = fmax(1e-48, leading_term * fptd_large(t_view[i] / (a**2),
#                     1 - w, k_l))
#             else:
#                 likelihoods_view[i] = fmax(1e-48, leading_term * fptd_small(t_view[i] / (a**2),
#                     1 - w, k_s))
#         elif t_view[i] > 0:
#             sgn_lambda, k_l, k_s = choice_function(t_view[i], eps)
#             leading_term = calculate_leading_term(t_view[i], v, a, w)
#             if sgn_lambda >= 0:
#                 likelihoods_view[i] = fmax(1e-48, leading_term * fptd_large(t_view[i] / (a**2),
#                     w, k_l))
#             else:
#                 likelihoods_view[i] = fmax(1e-48, leading_term * fptd_small(t_view[i] / (a**2),
#                     w, k_s))

#     return likelihoods

def batch_fptd(t, double v, double a, double w, double eps=1e-10):
    # Use when rts and choices vary, but parameters are held constant
    cdef int i
    cdef double[:] t_view = t
    cdef int n = t.shape[0]
    cdef double temp
    cdef double t_adj

    likelihoods = np.zeros(n)
    cdef double[:] likelihoods_view = likelihoods

    for i in range(n):
        if t_view[i] == 0:
            likelihoods_view[i] = 1e-48
        elif t_view[i] < 0:
            temp = fabs(t_view[i])
            t_adj = temp / (a ** 2)

            leading_term = calculate_leading_term(temp, (-1) * v, a, 1 - w)
            sgn_lambda, k_l, k_s = choice_function(t_adj, eps)
            if sgn_lambda >= 0:
                likelihoods_view[i] = fmax(1e-48, leading_term * fptd_large(t_adj, 1 - w, k_l))
            else:
                likelihoods_view[i] = fmax(1e-48, leading_term * fptd_small(t_adj, 1 - w, k_s))
        elif t_view[i] > 0:
            t_adj = t_view[i] / (a**2)

            leading_term = calculate_leading_term(t_view[i], v, a, w)
            sgn_lambda, k_l, k_s = choice_function(t_adj, eps)
            if sgn_lambda >= 0:
                likelihoods_view[i] = fmax(1e-48, leading_term * fptd_large(t_adj, w, k_l))
            else:
                likelihoods_view[i] = fmax(1e-48, leading_term * fptd_small(t_adj, w, k_s))

    return likelihoods


def array_fptd(t, v, a, w, double eps=1e-10):
    # Use when all inputs vary, but we want to feed in an array of them
    cdef int i
    cdef int n = t.shape[0]
    cdef double temp
    cdef double t_adj
    cdef double[:] t_view = t
    cdef double[:] v_view = v
    cdef double[:] a_view = a
    cdef double[:] w_view = w

    likelihoods = np.zeros(n)
    cdef double[:] likelihoods_view = likelihoods

    for i in range(n):
        if t_view[i] == 0:
            likelihoods_view[i] = 1e-29
        elif t_view[i] < 0:
            temp = fabs(t_view[i])
            t_adj = temp / (a_view[i] ** 2)

            leading_term = calculate_leading_term(temp, (-1) * v_view[i], a_view[i],
                    1 - w_view[i])
            sgn_lambda, k_l, k_s = choice_function(t_adj, eps)
            if sgn_lambda >= 0:
                likelihoods_view[i] = fmax(1e-29, leading_term * fptd_large(t_adj, 
                    1 - w_view[i], k_l))
            else:
                likelihoods_view[i] = fmax(1e-29, leading_term * fptd_small(t_adj, 
                    1 - w_view[i], k_s))
        elif t_view[i] > 0:
            t_adj = t_view[i] / (a_view[i] ** 2)

            leading_term = calculate_leading_term(t_view[i], v_view[i], 
                    a_view[i], w_view[i])
            sgn_lambda, k_l, k_s = choice_function(t_adj, eps)
            if sgn_lambda >= 0:
                likelihoods_view[i] = fmax(1e-29, leading_term * fptd_large(
                    t_adj, w_view[i], k_l))
            else:
                likelihoods_view[i] = fmax(1e-29, leading_term * fptd_small(
                    t_adj, w_view[i], k_s))

    return likelihoods

