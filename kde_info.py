import numpy as np
import pickle
import cddm_data_simulation as cd
import boundary_functions as bf

temp = {"ddm":
    {
    "dgp": cd.ddm_flexbound,
    "boundary": bf.constant,
    "data_folder": "/users/afengler/data/kde/ddm/train_test_data_20000",
#     custom_objects: {"huber_loss": tf.losses.huber_loss}
#     fcn_path: "/users/afengler/data/tony/kde/ddm/keras_models/\
# deep_inference08_12_19_11_15_06/model.h5"
#    fcn_custom_objects: {"heteroscedastic_loss": tf.losses.huber_loss}
    "output_folder": "/users/afengler/data/tony/kde/ddm/method_comparison/",
    "param_names": ["v", "a", "w"],
    "boundary_param_names": [],
    "param_bounds": np.array([[-2, .6, .3], [2, 1.5, .7]]),
    "boundary_param_bounds": []
    },
"linear_collapse":
    {
    "dgp": cd.ddm_flexbound,
    "boundary": bf.linear_collapse,
    "data_folder": "/users/afengler/data/kde/linear_collapse/train_test_data_20000",
    "output_folder": "/users/afengler/data/tony/kde/linear_collapse/method_comparison/",
    "param_names": ["v", "a", "w"],
    "boundary_param_names": ["node", "theta"],
    "param_bounds": np.array([[-2, .6, .3], [2, 1.5, .7]]),
    "boundary_param_bounds": np.array([[1, 0], [2, 1.37]])
    },
"ornstein":
    {
    "output_folder": "/users/afengler/data/tony/kde/ornstein_uhlenbeck/method_comparison/",
    "dgp": cd.ornstein_uhlenbeck,
    "data_folder": "/users/afengler/data/kde/ornstein_uhlenbeck/train_test_data_20000",
    "boundary": bf.constant,
    "param_names": ["v", "a", "w", "g"],
    "boundary_param_names": [],
    "boundary_param_bounds": [],
    "param_bounds": np.array([[-2, .6, .3, -1], [2, 1.5, .7, 1]])
    },
"full":
    {
    "dgp": cd.full_ddm,
    "output_folder": "/users/afengler/data/tony/kde/full_ddm/method_comparison/",
    "data_folder": "/users/afengler/data/kde/full_ddm/train_test_data_20000",
    "boundary": bf.constant,
    "param_names": ["v", "a", "w", "dw", "sdv"],
    "boundary_param_names": [],
    "boundary_param_bounds": [],
    "param_bounds": np.array([[-2, .6, .3, 0, 0], [2, 1.5, .7, .1, .5]])
    },
"ddm_fcn":
    {
    "dgp": cd.ddm_flexbound,
    "boundary": bf.constant,
    "data_folder": "/users/afengler/data/tony/kde/ddm/train_test_data_fcn",
    "output_folder": "/users/afengler/data/tony/kde/ddm/method_comparison_fcn/",
    "param_names": ["v", "a", "w"],
    "boundary_param_names": [],
    "param_bounds": np.array([[-2, .6, .3], [2, 1.5, .7]]),
    "boundary_param_bounds": []
    }
}

pickle.dump(temp, open("kde_stats.pickle", "wb"))
