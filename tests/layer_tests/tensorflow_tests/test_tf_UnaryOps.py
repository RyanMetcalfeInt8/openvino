# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
import sys
from common.tf_layer_test_class import CommonTFLayerTest


class TestUnaryOps(CommonTFLayerTest):
    current_op_type = None

    def _prepare_input(self, inputs_dict):
        non_negative = ['Sqrt', 'Log']
        narrow_borders = ["Tanh"]
        within_one = ['Asin', 'Acos', 'Atanh']
        from_one = ['Acosh']

        logical_type = ['LogicalNot']
        bitwise_type = ["BitwiseNot"]

        # usual function domain
        lower = -256
        upper = 256

        # specific domains
        if self.current_op_type in non_negative:
            lower = 0
        elif self.current_op_type in narrow_borders:
            lower = -16
            upper = 16
        elif self.current_op_type in from_one:
            lower = 1
        elif self.current_op_type in within_one:
            lower = -1
            upper = 1

        for input in inputs_dict.keys():
            if self.current_op_type in logical_type:
                inputs_dict[input] = np.random.randint(0, 1, inputs_dict[input]).astype(bool)
            elif self.current_op_type in bitwise_type:
                inputs_dict[input] = np.random.randint(lower, upper, inputs_dict[input]).astype(np.int32)
            else:
                inputs_dict[input] = np.random.uniform(lower, upper, inputs_dict[input]).astype(
                    np.float32)

        return inputs_dict

    def create_net_with_mish(self, shape, ir_version):
        import tensorflow as tf
        import tensorflow_addons as tfa

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, shape, 'Input')
            tfa.activations.mish(input)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    def create_net_with_unary_op(self, shape, ir_version, op_type):
        import tensorflow as tf

        self.current_op_type = op_type
        op_type_to_tf = {
            'Abs': tf.math.abs,
            'Acos': tf.math.acos,
            'Acosh': tf.math.acosh,
            'Asin': tf.math.asin,
            'Asinh': tf.math.asinh,
            'Atan': tf.math.atan,
            'BitwiseNot': tf.bitwise.invert,
            'Ceiling': tf.math.ceil,
            'Floor': tf.math.floor,
            'Log': tf.math.log,
            'LogicalNot': tf.math.logical_not,
            # 'Mish': tfa.activations.mish,  # temporarily moved to `create_net_with_mish()`
            'Negative': tf.math.negative,
            'Sign': tf.math.sign,
            'Square': tf.math.square,
            'Tan': tf.math.tan,
            'Tanh': tf.math.tanh,
            'ReLU': tf.nn.relu,
        }

        tf.compat.v1.reset_default_graph()

        type = tf.float32
        if op_type == "LogicalNot":
            type = tf.bool
        elif op_type == "BitwiseNot":
            type = tf.int32
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(type, shape, 'Input')
            if self.current_op_type == 'Mish':
                # Mish has no attribute name
                op_type_to_tf[self.current_op_type](input)
            else:
                op_type_to_tf[self.current_op_type](input, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    test_data_precommit = [dict(shape=[4, 6, 8, 10, 12])]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.parametrize("op_type", ['Abs',
                                         'Negative',
                                         'Tan',
                                         'Tanh',
                                         'Floor',
                                         'ReLU',
                                         'Ceiling',
                                         'Asin',
                                         'Acos',
                                         'Atan',
                                         'Log',
                                         'Sign',
                                         'Acosh',
                                         'Asinh',
                                         'LogicalNot',
                                         'Square',
                                         'BitwiseNot'])
    @pytest.mark.nightly
    def test_unary_op_precommit(self, params, ie_device, precision, ir_version, temp_dir, op_type):
        if ie_device == 'GPU':
            pytest.skip("5D tensors is not supported on GPU")
        self._test(*self.create_net_with_unary_op(**params, ir_version=ir_version, op_type=op_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.xfail(sys.version_info > (3, 10),
                       reason="tensorflow_addons package is not available for Python 3.11 and higher")
    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.nightly
    def test_unary_op_mish_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        """
        TODO: Move to `test_unary_op_precommit()` once tensorflow_addons package is available for Python 3.11
        """
        if ie_device == 'GPU':
            pytest.skip("5D tensors is not supported on GPU")
        self._test(*self.create_net_with_mish(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data = [pytest.param(dict(shape=[10, 12]), marks=pytest.mark.precommit),
                 dict(shape=[8, 10, 12]),
                 dict(shape=[6, 8, 10, 12]),
                 dict(shape=[4, 6, 8, 10, 12])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("op_type", ['Abs',
                                         'Negative',
                                         'Tan',
                                         'Tanh',
                                         'Floor',
                                         'ReLU',
                                         'Ceiling',
                                         'Asin',
                                         'Acos',
                                         'Atan',
                                         'Log',
                                         'Sign',
                                         'Acosh',
                                         'Asinh',
                                         'LogicalNot',
                                         'Square',
                                         'BitwiseNot'])
    @pytest.mark.nightly
    @pytest.mark.skipif(sys.platform == 'darwin', reason="Ticket - 122182")
    @pytest.mark.xfail(platform.machine() in ["aarch64", "arm64", "ARM64"], reason='Ticket - 122716')
    def test_unary_op(self, params, ie_device, precision, ir_version, temp_dir, op_type):
        if ie_device == 'GPU':
            pytest.skip("5D tensors is not supported on GPU")
        self._test(*self.create_net_with_unary_op(**params, ir_version=ir_version, op_type=op_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.xfail(sys.version_info > (3, 10),
                       reason="tensorflow_addons package is not available for Python 3.11 and higher")
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_unary_op_mish(self, params, ie_device, precision, ir_version, temp_dir):
        """
        TODO: Move to `test_unary_op()` once tensorflow_addons package is available for Python 3.11
        """
        if ie_device == 'GPU':
            pytest.skip("5D tensors is not supported on GPU")
        self._test(*self.create_net_with_mish(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
