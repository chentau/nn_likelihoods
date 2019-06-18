import numpy as np

from timeit import timeit

# print("Note: all simulations run with 5000 samples with default arguments, and gamma_bnd as the boundary function")
# print("All simulations repeated 10 times")
# print("testing ddm_simple")
# print("-" * 50)
# print("testing base python implementation")
# print(str(timeit("ddm_simulate(n_samples=5000)", setup="from ddm_data_simulation import ddm_simulate", 
#         number = 10) / 10) + " seconds on average")

# print("testing naive cython implementation")
# print(str(timeit("ddm_simulate1(n_samples=5000)", setup="from ddm_data_simulation1 import ddm_simulate1", 
#         number = 10) / 10) + " seconds on average")

# print("testing typed cython implementation")
# print(str(timeit("ddm_simulate2(n_samples=5000)", setup="from ddm_data_simulation1 import ddm_simulate2", 
#         number = 10) / 10) + " seconds on average")

# print("testing typed, non-vectorized cython implementation")
# print(str(timeit("ddm_simulate3(n_samples=5000)", setup="from ddm_data_simulation1 import ddm_simulate3", 
#         number = 10) / 10) + " seconds on average")

# print("-" * 50)
# print("testing ddm_flexbound")
# print("-" * 50)
# print("testing base python implementation")
# print(str(timeit("ddm_flexbound_simulate(n_samples=5000, boundary_fun=gamma_bnd, boundary_multiplicative=False, boundary_params={})", 
#  setup="from ddm_data_simulation import ddm_flexbound_simulate; from boundary_functions import gamma_bnd", number=5) / 5) + " seconds on average")

# print("testing naive cython implementation")
# print(str(timeit("ddm_flexbound_simulate1(n_samples=5000, boundary_fun=gamma_bnd, boundary_multiplicative=False, boundary_params={})", 
#  setup="from ddm_data_simulation1 import ddm_flexbound_simulate1; from boundary_functions import gamma_bnd", number=10) / 10) + " seconds on average")

# print("testing non-vectorized cython implementation")
# print(str(timeit("ddm_flexbound_simulate2(n_samples=5000, boundary_fun=gamma_bnd, boundary_multiplicative=False, boundary_params={})", 
#  setup="from ddm_data_simulation1 import ddm_flexbound_simulate2; from boundary_functions import gamma_bnd", number=10) / 10) + " seconds on average")

print("-" * 50)
print("testing race_model")
print("-" * 50)
print("note - base python version does not work")

print("testing naive cython implementation")
print(str(timeit("race_model1(n_samples=5000, boundary_fun=gamma_bnd)",
    setup="from ddm_data_simulation1 import race_model1; from boundary_functions import gamma_bnd",
    number=10) / 10) + " seconds on average")

# print("testing non-vectorized cython implementation")
# print(str(timeit("race_model2(n_samples=5000, boundary_fun=gamma_bnd)",
#     setup="from ddm_data_simulation1 import race_model2; from boundary_functions import gamma_bnd",
#     number=10) / 10) + " seconds on average")

# print("testing ornstein_uhlenbeck")
# # print("-" * 50)
# # print("testing python implementation")
# # print(str(timeit("ornstein_uhlenbeck(n_samples = 5000)",
# #     setup="from ddm_data_simulation import ornstein_uhlenbeck; from boundary_functions import gamma_bnd",
# #     number=10) / 10) + "seconds on average")

# print("testing naive cython implementation")
# print(str(timeit("ornstein_uhlenbeck1(n_samples = 5000)",
#     setup="from ddm_data_simulation1 import ornstein_uhlenbeck1; from boundary_functions import gamma_bnd",
#     number=10) / 10) + " seconds on average")

# print("testing non-vectorized cython implementation")
# print(str(timeit("ornstein_uhlenbeck2(n_samples = 5000)",
#     setup="from ddm_data_simulation1 import ornstein_uhlenbeck2; from boundary_functions import gamma_bnd",
#     number=10) / 10) + " seconds on average")


# print("-" * 50)
# print("testing ornstein_uhlenbeck_flexbound")
# print("-" * 50)
# print("note - base python version does not work")
# # rts, choices = ornstein_uhlenbeck_flexbnd(n_samples = 5000, boundary_fun=gamma_bnd,
# #         boundary_params={})

# print("testing naive cython implementation")
# print(str(timeit("ornstein_uhlenbeck_flexbnd1(n_samples = 5000, boundary_fun=gamma_bnd)",
#     setup="from ddm_data_simulation1 import ornstein_uhlenbeck_flexbnd1; from boundary_functions \
#             import gamma_bnd",
#     number=10) / 10) + "seconds on average")

# print("testing non-vectorized cython implementation")
# print(str(timeit("ornstein_uhlenbeck_flexbnd2(n_samples = 5000, boundary_fun=gamma_bnd)",
#     setup="from ddm_data_simulation1 import ornstein_uhlenbeck_flexbnd2; from boundary_functions \
#             import gamma_bnd",
#     number=10) / 10) + "seconds on average")
