import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


# book-keeping corresponding crop function, easy for scaling
crop_fn = {'1d': crop1d, '2d': crop2d}


def crop_and_concat(self, x1, x2):
    """
    crop tensor x1 to match x2, x2 shape is the target shape,
    then concatenate them along feature dimension
    """
    shape_x1 = x1.get_shape().as_list()
    shape_x2 = x2.get_shape().as_list()
    assert shape_x1 == shape_x2
    if x2 is None:
        return x1
    crop_helper_fn = crop_fn['1d']
    x1 = crop_helper_fn(x1, x2.get_shape().as_list())
    return tf.concat([x1, x2], axis=shape_x1 - 1)


def crop2d(self, tensor, target_shape):
    """
    crop tensor to match target_shape,
    remove the diff/2 items at the start and at the end,
    keep only the central part of the vector
    """
    # the tensor flow in model is of shape (batch, row, col, channels)
    shape = tensor.get_shape().as_list()
    diff_row = shape[1] - target_shape[1]
    diff_col = shape[2] - target_shape[2]
    assert diff_row >= 0 and diff_col >= 0  # Only positive difference allowed
    if diff_row == 0 and diff_col == 0:
        return tensor
    # calculate new cropped row index
    row_crop_start = diff_row // 2
    row_crop_end = diff_row - row_crop_start
    # calculate new cropped column index
    col_crop_start = diff_col // 2
    col_crop_end = diff_row - col_crop_start
    return tensor[:, row_crop_start:-row_crop_end, col_crop_start:-col_crop_end, :]


def crop1d(self, tensor, target_shape):
    """
    crop tensor to match target_shape,
    remove the diff/2 items at the start and at the end,
    keep only the central part of the vector
    """
    # the tensor flow in model is of shape (batch, freq_bins, time_frames)
    shape = tensor.get_shape().as_list()
    diff = shape[1] - target_shape[1]
    assert diff >= 0  # Only positive difference allowed
    if diff == 0:
        return tensor
    crop_start = diff // 2
    crop_end = diff - crop_start
    return tensor[:, crop_start:-crop_end, :]


def tile(self, tensor, multiples):
    shape = tensor.get_shape().as_list()
    row_multiple, col_multiple = multiples
    if len(shape) == 3:
        # the tensor flow in model is of shape (batch, row, column)
        multiples = tf.constant((1, row_multiple, col_multiple), tf.int32)
    elif len(shape) == 4:
        # the tensor flow in model is of shape (batch, row, column, channels)
        multiples = tf.constant((1, row_multiple, col_multiple, 1), tf.int32)
    else:
        raise ValueError("check the tensor shape!")
    return tf.tile(tensor, multiples)
