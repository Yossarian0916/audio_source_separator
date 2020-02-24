import tensorflow as tf
from tensorflow import keras


def crop_and_concat(x1, x2):
    """
    crop tensor x1 to match x2, x2 shape is the target shape,
    then concatenate them along feature dimension
    """
    if x2 is None:
        return x1
    x1 = crop(x1, x2.get_shape().as_list())
    return tf.concat([x1, x2], axis=-1)


def crop(tensor, target_shape):
    """
    crop tensor to match target_shape,
    remove the diff/2 items at the start and at the end,
    keep only the central part of the vector
    """
    # the tensor flow in model is of shape (batch, freq_bins, time_frames, channels)
    shape = tensor.get_shape().as_list()
    diff_row = shape[1] - target_shape[1]
    diff_col = shape[2] - target_shape[2]
    diff_channel = shape[3] - target_shape[3]
    if diff_row < 0 or diff_col < 0 or diff_channel < 0:
        raise ValueError('input tensor cannot be shaped as target_shape')
    row_slice = slice(0, None)
    col_slice = slice(0, None)
    channel_slice = slice(0, None)
    if diff_row == 0 and diff_col == 0 and diff_channel == 0:
        return tensor
    if diff_row != 0:
        # calculate new cropped row index
        row_crop_start = diff_row // 2
        row_crop_end = diff_row - row_crop_start
        row_slice = slice(row_crop_start, -row_crop_end)
    if diff_col != 0:
        # calculate new cropped column index
        col_crop_start = diff_col // 2
        col_crop_end = diff_col - col_crop_start
        col_slice = slice(col_crop_start, -col_crop_end)
    if diff_channel != 0:
        # calculate new cropped channel axis index
        channel_crop_start = diff_channel // 2
        channel_crop_end = diff_channel - channel_crop_start
        channel_slice = slice(channel_crop_start, -channel_crop_end)
    return tensor[:, row_slice, col_slice, channel_slice]


def tile(tensor, multiples):
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
