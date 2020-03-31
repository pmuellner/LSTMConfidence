import tensorflow as tf
from tensorflow.python.ops import math_ops, init_ops
from tensorflow.python.layers import base as base_layer
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class LSTM(tf.nn.rnn_cell.LSTMCell):
    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=False, activation=None, reuse=None, name=None):
        super(LSTM, self).__init__(num_units=num_units, reuse=reuse, name=name)
        self.input_spec = base_layer.InputSpec(ndim=2)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units) if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME, shape=[input_depth + h_depth, 4 * self._num_units])
        self._bias = self.add_variable(_BIAS_VARIABLE_NAME, shape=[4 * self._num_units],
                                       initializer=init_ops.zeros_initializer(dtype=self.dtype))
        self.built = True

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        add = math_ops.add
        multiply = math_ops.multiply
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))), multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))

        input_gate = tf.identity(sigmoid(i), name="input_gate")
        forget_gate = tf.identity(sigmoid(add(f, forget_bias_tensor)), name="forget_gate")
        output_gate = tf.identity(sigmoid(o), name="output_gate")
        candidate = tf.identity(self._activation(j), name="candidate")

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)

        return new_h, new_state