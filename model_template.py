from __future__ import print_function
import tensorflow as tf


class ModelTemplate:
    """Template for supported models. Supported models are built via the `build()` method.

    The `build()` method must not assume that tensorflow variables/operations
    which were instantiated within `__init__` are live. The initialization code of
    `build()` must be **entirely** self-contained.
    """

    def __init__(self, **kwargs):
        pass

    def build(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor: tensorflow.Tensor
            A tensor specifying the input to the model

        Returns
        -------
        None
        """