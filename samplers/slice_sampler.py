import numpy as np

class SliceSampler:
    def __init__(self, bounds, target, w = 1 / 256, p = 8):
        self.dims = bounds.shape[0]
        self.bounds = bounds
        self.target = target
        self.w = w
        self.p = p

#     def _slice_sample(self, prev, prev_lp):
#         out = prev.copy()
#         for dim in range(self.dims):
#             z = prev_lp - np.random.exponential()
#             lp = -np.inf
#             cnt = 0
#             while lp < z:
#                 if cnt == 5000:
#                     print("could not find adequate sample!")
#                     break
#                 out[dim] = np.random.uniform(self.bounds[dim][0], self.bounds[dim][1])
#                 lp = self.target(out, self.data)
#                 cnt += 1
#         return (out, lp)

    def _find_interval(self, z, prev, dim):
        left = prev.copy()
        right = prev.copy()
        u = np.random.uniform()

        left[dim] = prev[dim] - self.w * u
        right[dim] = left[dim] + self.w
        k = self.p

        lp_l = self.target(left, self.data)
        lp_r = self.target(right, self.data)

        while k > 0 and (z < lp_l or z < lp_r):
            v = np.random.uniform()
            if v < .5:
                left[dim] += left[dim] - right[dim]
                left[dim] = np.clip(left[dim], self.bounds[dim][0], self.bounds[dim][1])
                lp_l = self.target(left, self.data)
            else:
                right[dim] += right[dim] - left[dim]
                right[dim] = np.clip(right[dim], self.bounds[dim][0], self.bounds[dim][1])
                lp_r = self.target(right, self.data)
            k -= 1
        return left[dim], right[dim]

    def _slice_sample(self, prev, prev_lp):
        out = prev.copy()
        for dim in range(self.dims):
            z = prev_lp - np.random.exponential()
            left, right = self._find_interval(z, prev, dim)

            # Adaptively shrink the interval
            while not np.isclose(left,right, 1e-2):
                u = np.random.uniform()
                out[dim] = left + u * (right - left)
                lp = self.target(out, self.data)
                if z < lp:
                    break
                else:
                    if out[dim] < prev[dim]:
                        left = out[dim]
                    else:
                        right = out[dim]
        return (out, lp)

    def sample(self, data, num_samples=1000):
        self.data = data
        self.samples = np.zeros((num_samples, self.dims))

        init = np.zeros(self.dims)
        for dim in range(self.dims):
            init[dim] = np.random.uniform(self.bounds[dim][0], self.bounds[dim][1])
        init_lp = self.target(init, self.data)
        
        self.samples[0], prev_lp = self._slice_sample(init, init_lp)

        print("Beginning sampling")

        for i in range(1, num_samples):
            if i % 100 == 0:
                print("Iteration {}".format(i))
            self.samples[i], prev_lp = self._slice_sample(self.samples[i-1], prev_lp)

