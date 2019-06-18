# Functions for DDM data simulation
import cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, sqrt

import numpy as np
import pandas as pd
from time import time
import inspect

# Convert all variables to c types, and compare to previous benchmarks
# Compare vectorized approach with C loop approach
DTYPE = np.float32

# Method to draw random samples from a gaussian
cdef float random_uniform():
    cdef float r = rand()
    return r / RAND_MAX

cdef float random_gaussian():
    cdef float x1, x2, w
    w = 2.0
    
    while(w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2
        
    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w
    
@cython.boundscheck(False)
cdef void assign_random_gaussian_pair(float[:] out, int assign_ix):
    cdef float x1, x2, w
    w = 2.0
    
    while(w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2
        
    w = ((-2.0 * log(w)) / w) ** 0.5
    out[assign_ix] = x1 * w
    out[assign_ix + 1] = x2 * 2
    
@cython.boundscheck(False)
cdef float[:] draw_gaussian(int n):
    # Draws standard normal variables - need to have the variance rescaled
    cdef int i
    cdef float[:] result = np.zeros(n, dtype=DTYPE)
    for i in range(n // 2):
        assign_random_gaussian_pair(result, i * 2)
    if n % 2 == 1:
        result[n - 1] = random_gaussian()
    return result

# Returns a numpy array instead of a memoryview
@cython.boundscheck(False)
def draw_gaussian1(int n):
    # Draws standard normal variables - need to have the variance rescaled
    cdef int i
    result = np.zeros(n, dtype=DTYPE)
    cdef float[:] result_view = result
    for i in range(n // 2):
        assign_random_gaussian_pair(result_view, i * 2)
    if n % 2 == 1:
        result_view[n - 1] = random_gaussian()
    return result
# Simulate (rt, choice) tuples from: SIMPLE DDM -----------------------------------------------

# Simplest algorithm
def ddm_simulate1(v = 0, # drift by timestep 'delta_t'
                  a = 1, # boundary separation
                  w = 0.5,  # between -1 and 1
                  s = 1, # noise sigma
                  delta_t = 0.001, # timesteps fraction of seconds
                  max_t = 20, # maximum rt allowed
                  n_samples = 20000, # number of samples considered
                  print_info = True # timesteps fraction of seconds
                  ):

    delta_t_sqrt = np.sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s

    y = np.repeat(w * a, n_samples)
    t = np.zeros(n_samples)
    finished = np.zeros(n_samples)

    while finished.sum() < n_samples and np.max(t) <= max_t:
        finished = (y > a) + (y < 0) # boolean array that indexes whether a step is finished or not
        y += (v * delta_t + sqrt_st * draw_gaussian(n_samples)) * (1 - finished) # only update the runs that have not finished
        t += delta_t * (1 - finished)

    # Store choice and reaction time
    rts = t[:, np.newaxis]
    # Note that for purposes of consistency with Navarro and Fuss, the choice corresponding the lower barrier is +1, higher barrier is -1
    choices = ((-1) * np.sign(y))[:, np.newaxis]

    # if print_info == True:
    #     if n % 1000 == 0:
    #         print(n, ' datapoints sampled')

    return (rts, choices, {'v': v,
                           'a': a,
                           'w': w,
                           's': s,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm',
                           'boundary_fun_type': 'constant',
                           'possible_choices': [-1, 1]})

# Add in typing of variables
def ddm_simulate2(float v = 0, # drift by timestep 'delta_t'
                  float a = 1, # boundary separation
                  float w = 0.5,  # between -1 and 1
                  float s = 1, # noise sigma
                  float delta_t = 0.001, # timesteps fraction of seconds
                  float max_t = 20, # maximum rt allowed
                  int n_samples = 20000, # number of samples considered
                  print_info = True # timesteps fraction of seconds
                  ):

    cdef float delta_t_sqrt = np.sqrt(delta_t)
    cdef float sqrt_st = delta_t_sqrt * s

    y = np.repeat(w * a, n_samples)
    t = np.zeros(n_samples, dtype=DTYPE)
    finished = np.zeros(n_samples, dtype=DTYPE)

    while finished.sum() < n_samples and np.max(t) <= max_t:
        finished = (y > a) + (y < 0) # boolean array that indexes whether a step is finished or not
        y += (v * delta_t + sqrt_st * draw_gaussian1(n_samples)) * (1 - finished) # only update the runs that have not finished
        t += delta_t * (1 - finished)

    # Store choice and reaction time
    rts = t[:, np.newaxis]
    # Note that for purposes of consistency with Navarro and Fuss, the choice corresponding the lower barrier is +1, higher barrier is -1
    choices = ((-1) * np.sign(y))[:, np.newaxis]

    # if print_info == True:
    #     if n % 1000 == 0:
    #         print(n, ' datapoints sampled')

    return (rts, choices, {'v': v,
                           'a': a,
                           'w': w,
                           's': s,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm',
                           'boundary_fun_type': 'constant',
                           'possible_choices': [-1, 1]})
    
# replace the vectorized code with C loops
def ddm_simulate3(float v = 0, # drift by timestep 'delta_t'
                  float a = 1, # boundary separation
                  float w = 0.5,  # between -1 and 1
                  float s = 1, # noise sigma
                  float delta_t = 0.001, # timesteps fraction of seconds
                  float max_t = 20, # maximum rt allowed
                  int n_samples = 20000, # number of samples considered
                  print_info = True # timesteps fraction of seconds
                  ):

    rts = np.zeros((n_samples, 1), dtype=DTYPE)
    choices = np.zeros((n_samples, 1), dtype=np.intc)
    cdef float[:,:] rts_view = rts
    cdef int[:,:] choices_view = choices
    
    cdef float delta_t_sqrt = np.sqrt(delta_t)
    cdef float sqrt_st = delta_t_sqrt * s

    cdef float y, t

    cdef int n
    cdef int m = 0
    cdef int num_draws = int(max_t / delta_t + 1)
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    
    for n in range(n_samples):
        y = w * a
        t = 0.0
        while y <= a and y >= 0 and t <= max_t:
            y += v * delta_t + sqrt_st * gaussian_values[m] # only update the runs that have not finished
            t += delta_t
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0
        rts_view[n, 0] = t
        choices_view[n, 0] = (-1) * np.sign(y)


    # if print_info == True:
    #     if n % 1000 == 0:
    #         print(n, ' datapoints sampled')

    return (rts, choices, {'v': v,
                           'a': a,
                           'w': w,
                           's': s,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm',
                           'boundary_fun_type': 'constant',
                           'possible_choices': [-1, 1]})
    
def ddm_flexbound_simulate1(v = 0,
                           a = 1,
                           w = 0.5,
                           s = 1,
                           delta_t = 0.001,
                           max_t = 20,
                           n_samples = 20000,
                           print_info = True,
                           boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                           boundary_multiplicative = True,
                           boundary_params = {'p1': 0, 'p2':0}
                          ):

    # Initializations
    # print({'boundary_fun': boundary_fun})
    
    delta_t_sqrt = np.sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage: 0 index for lower bound, 1 index for upper bound
    # Instead of keeping two vectors for uppper and lower bounds, only store one vector
    boundaries = np.zeros(((int((max_t/delta_t)), 2)))
    
    if boundary_multiplicative:
        tmp = np.array([a * boundary_fun(t = i * delta_t, **boundary_params)
            for i in range(0, int((max_t/delta_t)), 1)])
        boundaries[:, 1] = np.where(tmp > 0, tmp, 0)
        boundaries[:, 0] = np.where(tmp > 0, (-1) * tmp, 0)
    else:
        tmp = np.array([a + boundary_fun(t = i * delta_t, **boundary_params)
            for i in range(0, int((max_t/delta_t)), 1)])
        boundaries[:, 1] = np.where(tmp > 0, tmp, 0)
        boundaries[:, 0] = np.where(tmp > 0, (-1) * tmp, 0)
        
    y = np.repeat((boundaries[0, 0] + (w * (boundaries[0, 1] - boundaries[0, 0]))), n_samples)
    t = np.zeros(n_samples)
    
    ix = 0
    finished = np.zeros(n_samples)

    while finished.sum() < n_samples and np.max(t) <= max_t:
        finished = (y > boundaries[ix, 1]) + (y < boundaries[ix, 0])
        y += (v * delta_t + sqrt_st * draw_gaussian(n_samples)) * (1 - finished)
        t += delta_t * (1 - finished)
        ix += 1
        
    rts = t[:, np.newaxis]
    choices = np.sign(y)[:, np.newaxis]

    # if print_info == True:
    #     if n % 1000 == 0:
    #         print(n, ' datapoints sampled')
            
    return (rts, choices,  {'v': v,
                           'a': a,
                           'w': w,
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm_flexbound',
                           'boundary_fun_type': boundary_fun.__name__,
                           'possible_choices': [-1, 1]})

# Replace vectorized code with c loops
def ddm_flexbound_simulate2(float v = 0,
                           float a = 1,
                           float w = 0.5,
                           float s = 1,
                           float delta_t = 0.001,
                           float max_t = 20,
                           int n_samples = 20000,
                           print_info = True,
                           boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                           boundary_multiplicative = True,
                           boundary_params = {'p1': 0, 'p2':0}
                          ):

    rts = np.zeros((n_samples, 1), dtype=DTYPE)
    choices = np.zeros((n_samples, 1), dtype=np.intc)
    
    cdef float[:,:] rts_view = rts
    cdef int[:,:] choices_view = choices
    
    cdef float delta_t_sqrt = np.sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = delta_t_sqrt * s # scalar to ensure the correct variance for the gaussian step

    # Boundary storage for the upper bound
    cdef int num_draws = int((max_t / delta_t) + 1)
    boundaries = np.zeros(num_draws, dtype=DTYPE)
    cdef float[:] boundaries_view = boundaries
    cdef int i
    cdef float tmp
    
    if boundary_multiplicative:
        for i in range(num_draws):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundaries_view[i] = tmp
    else:
        for i in range(num_draws):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundaries_view[i] = tmp
        
    cdef float y, t
    cdef int n, ix
    cdef int m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)

    for n in range(n_samples):
        y = w * a
        t = 0
        ix = 0
        while y >= (-1) * boundaries_view[ix] and y <= boundaries_view[ix] and t <= max_t:
            y += v * delta_t + sqrt_st * gaussian_values[m]
            t += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0
        
        rts_view[n, 0] = t
        choices_view[n, 0] = np.sign(y)
        
    # if print_info == True:
    #     if n % 1000 == 0:
    #         print(n, ' datapoints sampled')
            
    return (rts, choices,  {'v': v,
                           'a': a,
                           'w': w,
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm_flexbound',
                           'boundary_fun_type': boundary_fun.__name__,
                           'possible_choices': [-1, 1]})

# Simulate (rt, choice) tuples from: RACE MODEL WITH N SAMPLES ----------------------------------
cdef bint check_finished(float[:] particles, float boundary):
    cdef int i,n
    n = particles.shape[0]
    for i in range(n):
        if i > boundary:
            return True
        
def test_check():
    temp = np.random.normal(0,1, 10).astype(DTYPE)
    cdef float[:] temp_view = temp
    start = time()
    [check_finished(temp_view, 3) for _ in range(1000000)]
    print(check_finished(temp_view, 3))
    end = time()
    print("cython check: {}".format(start - end))
    start = time()
    [(temp > 3).any() for _ in range(1000000)]
    end = time()
    print("numpy check: {}".format(start - end))

def race_model1(v = np.array([0, 0, 0]), # np.array expected in fact, one column of floats
               w = np.array([0, 0, 0]), # np.array expected in fact, one column of floats
               s = np.array([1, 1, 1]), # np.array expected in fact, one column of floats
               delta_t = 0.001,
               max_t = 20,
               n_samples = 2000,
               print_info = True,
               boundary_fun = None,
               **boundary_params):

    # Initializations
    n_particles = len(v)
    delta_t_sqrt = np.sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s
    # particles = np.zeros((n_particles, 1))

    # We just care about an upper boundary here: (more complicated things possible)
    boundaries = np.array([boundary_fun(t = i * delta_t, **boundary_params)
        for i in range(int(max_t/delta_t + 1))])

    particles = np.tile((w * boundaries[0])[:, np.newaxis], n_samples)
    t = np.zeros(n_samples)
    finished = np.zeros(n_samples)
    ix = 0
    
    while finished.sum() < n_samples and np.max(t) < max_t:
        try:
            finished = (particles > boundaries[ix]).any(axis=0)
        except:
            print(ix)
            print(np.max(t))
            print(finished.sum())
        particles += ((v * delta_t)[:, np.newaxis] + sqrt_st[:, np.newaxis] * 
                draw_gaussian1(n_samples * n_particles).reshape(n_particles,
                    n_samples)) * (1 - finished)
        t += delta_t * (1 - finished)
        ix += 1

    rts = t[:, np.newaxis]
    choices = particles.argmax(axis=0)[:, np.newaxis]

    # if print_info == True:
    #     if n % 1000 == 0:
    #         print(n, ' datapoints sampled')
    return (rts, choices)
        
@cython.boundscheck(False)
def race_model2(v = np.array([0, 0, 0], dtype=DTYPE), # np.array expected, one column of floats
               w = np.array([0, 0, 0], dtype=DTYPE), # np.array expected, one column of floats
               s = np.array([1, 1, 1], dtype=DTYPE), # np.array expected, one column of floats
               float delta_t = 0.001,
               float max_t = 20,
               int n_samples = 2000,
               print_info = True,
               boundary_fun = None,
               **boundary_params):

    # Initializations
    cdef float[:] v_view = v
    cdef float[:] w_view = w
    cdef float delta_t_sqrt = np.sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s
    cdef float[:] sqrt_st_view = sqrt_st
    
    cdef int n_particles = len(v)
    rts = np.zeros((n_samples, 1), dtype=DTYPE)
    cdef float[:,:] rts_view = rts
    choices = np.zeros((n_samples, 1), dtype=np.intc)
    cdef int[:,:] choices_view = choices
    cdef float [:] particles_view
    
    # We just care about an upper boundary here: (more complicated things possible)
    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef int i
    
    boundaries = np.zeros(num_steps, dtype=DTYPE)
    cdef float[:] boundaries_view = boundaries
    for i in range(num_steps):
        boundaries_view[i] = boundary_fun(t = i * delta_t, **boundary_params)
    
    cdef float t
    cdef int n, ix, j
    cdef int m = 0
    
    cdef int num_draws = num_steps * n_particles
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    
    for n in range(n_samples):
        particles = np.repeat(w * boundaries_view[0], n_particles)
        particles_view = particles
        ix = 0
        while not check_finished(particles_view, boundaries_view[ix]) and t <= max_t:
            for j in range(n_particles):
                particles_view[j] += (v_view[j] * delta_t) + sqrt_st_view[j] * gaussian_values[m]
                m += 1
                if m == num_draws:
                    m = 0
                    gaussian_values = draw_gaussian(num_draws)
            t += delta_t
            ix += 1
            
        rts_view[n, 0] = t
        choices_view[n, 0] = np.argmax(particles)

    return (rts, choices)
# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Ornstein-Uhlenbeck -------------------------------------------
def ornstein_uhlenbeck1(v = 1, # drift parameter
                       a = 1, # boundary separation parameter
                       w = 0.5, # starting point bias
                       g = 0.1, # decay parameter
                       s = 1, # standard deviation
                       delta_t = 0.001, # size of timestamp
                       max_t = 20, # maximal time in trial
                       n_samples = 2000, # number of samples from process
                       print_info = True): # whether or not ot print periodic update on number of samples generated

    # Initializations
    delta_t_sqrt = np.sqrt(delta_t)
    sqrt_st = delta_t_sqrt * s

    y = np.repeat(w * a, n_samples)
    t = np.zeros(n_samples)
    finished = np.zeros(n_samples)
    
    while finished.sum() < n_samples and np.max(t) < max_t:
        finished = (y > a) + (y < 0)
        y += (((v * delta_t) - (delta_t * g * y)) + sqrt_st * \
                draw_gaussian1(n_samples)) * (1 - finished)
        t += delta_t * (1 - finished)
        
    
    rts = t[:, np.newaxis]
    # Note that for purposes of consistency with Navarro and Fuss, the choice corresponding the lower barrier is +1, higher barrier is -1
    choices = np.sign(y)[:, np.newaxis]

    # if print_info == True:
    #     if n % 1000 == 0:
    #         print(n, ' datapoints sampled')

    return (rts, choices)
# -------------------------------------------------------------------------------------------------

def ornstein_uhlenbeck2(float v = 1, # drift parameter
                       float a = 1, # boundary separation parameter
                       float w = 0.5, # starting point bias
                       float g = 0.1, # decay parameter
                       float s = 1, # standard deviation
                       float delta_t = 0.001, # size of timestamp
                       float max_t = 20, # maximal time in trial
                       int n_samples = 2000, # number of samples from process
                       print_info = True): # whether or not ot print periodic update on number of samples generated

    # Initializations
    rts = np.zeros((n_samples, 1), dtype=DTYPE)
    choices = np.zeros((n_samples, 1), dtype=np.intc)
    
    cdef float[:,:] rts_view = rts
    cdef int[:,:] choices_view = choices
    
    cdef float delta_t_sqrt = np.sqrt(delta_t)
    cdef float sqrt_st = delta_t_sqrt * s

    cdef int n
    cdef int m = 0
    cdef float y, t
    cdef int num_draws = int(max_t / delta_t + 1)
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    
    for n in range(n_samples):
        y = w * a
        t = 0.0
        while y <= a and y >= 0 and t < max_t:
            y += ((v * delta_t) - (delta_t * g * y)) + sqrt_st * gaussian_values[m]
            t += delta_t
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0
            
        rts_view[n, 0] = t
    # Note that for purposes of consistency with Navarro and Fuss, the choice corresponding the lower barrier is +1, higher barrier is -1
        choices_view[n, 0] = np.sign(y)

    # if print_info == True:
    #     if n % 1000 == 0:
    #         print(n, ' datapoints sampled')

    return (rts, choices)

# Simulate (rt, choice) tuples from: Onstein-Uhlenbeck with flexible bounds -----------------------
def ornstein_uhlenbeck_flexbnd1(v = 0, 
                                w = 0.5, 
                                g = 0.1, 
                                s = 1, 
                                delta_t = 0.001, 
                                max_t = 20, 
                                n_samples = 20000, 
                                print_info = True, 
                                boundary_fun = None, 
                                **boundary_params):
    delta_t_sqrt = np.sqrt(delta_t)
    sqrt_st = s * delta_t_sqrt

    tmp = np.array([boundary_fun(t = i * delta_t, **boundary_params) 
    for i in range(int(max_t / delta_t + 1))])
    boundaries = np.where(tmp > 0, tmp, 0)

    y = np.repeat((-1) * boundaries[0] + (w * 2 * boundaries[0]), n_samples)
    t = np.zeros(n_samples)
    finished = np.zeros(n_samples)
    ix = 0

    while finished.sum() < n_samples and np.max(t) <= max_t:
        finished = (y < (-1) * boundaries[ix]) + (y > boundaries[ix])
        y += (((v * delta_t) - (delta_t * g * y)) + sqrt_st * draw_gaussian(n_samples)) * (1 - finished)
        t += (delta_t) * (1 - finished)
        ix += 1
    
    rts = t[:, np.newaxis]
    choices = np.sign(y)[:, np.newaxis]

    return rts, choices

def ornstein_uhlenbeck_flexbnd2(float v = 0, # drift parameter
                               float w = 0.5, # starting point bias
                               float g = 0.1, # decay parameter
                               float s = 1, # standard deviation
                               float delta_t = 0.001, # size of timestep
                               float max_t = 20, # maximal time in trial
                               int n_samples = 20000, # number of samples from process
                               print_info = True, # whether or not to print periodic update on number of samples generated
                               boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                               **boundary_params):
    # Initializations
    rts = np.zeros((n_samples,1), dtype=DTYPE) # rt storage
    choices = np.zeros((n_samples,1), dtype=np.intc) # choice storage
    
    cdef float[:,:] rts_view = rts
    cdef int[:,:] choices_view = choices
    
    cdef float delta_t_sqrt = np.sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion
    cdef float sqrt_st = s * delta_t_sqrt

    # Boundary storage:
    cdef int num_draws = int((max_t / delta_t) + 1)
    boundaries = np.zeros(int(max_t/delta_t + 1), dtype=DTYPE)
    cdef float[:] boundaries_view = boundaries
    cdef int i
    cdef float tmp

    for i in range(num_draws):
        tmp = boundary_fun(t = i * delta_t, **boundary_params)
        if tmp > 0:
            boundaries_view[i] = tmp
                
    cdef float y, t
    cdef int n, ix
    cdef int m = 0
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    
    for n in range(n_samples):
        y = (-1) * boundaries_view[0] + (w * 2 * boundaries_view[0])
        t = 0
        ix = 0
        while y >= (-1) * boundaries_view[ix] and y <= boundaries_view[ix] and t <= max_t:
            y += ((v * delta_t) - (delta_t * g * y)) + sqrt_st * gaussian_values[m]
            t += delta_t
            ix += 1
            m += 1
            if m == num_draws:
                gaussian_values = draw_gaussian(num_draws)
                m = 0
                
        rts_view[n, 0] = t
        choices_view[n, 0] = np.sign(y)
        
        # Note that for purposes of consistency with Navarro and Fuss, the choice corresponding the lower barrier is +1, higher barrier is -1
        # This is kind of a legacy issue at this point (plan is to flip this around, after appropriately reformulating navarro fuss wfpd function)

        # if print_info == True:
        #     if n % 1000 == 0:
        #         print(n, ' datapoints sampled')
    return (rts, choices)
# --------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Leaky Competing Accumulator Model -----------------------------
def lca1(v = [0, 0, 0], # drift parameters (np.array expect: one column of floats)
        w = [0, 0, 0], # initial bias parameters (np.array expect: one column of floats)
        a = 1, # criterion height
        g = 0, # decay parameter
        b = 1, # inhibition parameter
        s = 1, # variance (can be one value or np.array of size as v and w)
        delta_t = 0.001, # time-step size in simulator
        max_t = 20, # maximal time
        n_samples = 2000, # number of samples to produce
        print_info = True): # whether or not to periodically report the number of samples generated thus far

    # Initializations
    n_particles = len(v)
    rts = np.zeros((n_samples, 1))
    choices = np.zeros((n_samples, 1))
    delta_t_sqrt = np.sqrt(delta_t)
    particles = np.zeros((n_particles, 1))

    for n in range(0, n_samples, 1):

        # initialize y, t and time_counter
        particles_reduced_sum = particles
        particles = w * a
        t = 0

        while np.less_equal(particles, a).all() and t <= max_t:
            particles_reduced_sum[:,] = - particles + np.sum(particles)
            particles += ((v - (g * particles) - (b * particles_reduced_sum)) * delta_t) + \
                         (delta_t_sqrt * np.random.normal(loc = 0, scale = s, size = (n_particles, 1)))
            particles = np.maximum(particles, 0.0)
            t += delta_t

        rts[n] = t
        choices[n] = particles.argmax()

        if print_info == True:
            if n % 1000 == 0:
                print(n, ' datapoints sampled')
    return (rts, choices)
# --------------------------------------------------------------------------------------------------

