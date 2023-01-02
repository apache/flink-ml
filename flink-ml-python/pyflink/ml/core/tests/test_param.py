################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import unittest
from typing import Dict, Any

from pyflink.ml.core.param import Param
from pyflink.ml.lib.param import HasDistanceMeasure, HasFeaturesCol, HasGlobalBatchSize, \
    HasHandleInvalid, HasInputCols, HasLabelCol, HasLearningRate, HasMaxIter, HasMultiClass, \
    HasOutputCols, HasPredictionCol, HasRawPredictionCol, HasReg, HasSeed, HasTol, HasWeightCol, \
    HasWindows, HasRelativeError, HasFlatten, HasModelVersionCol, HasMaxAllowedModelDelayMs

from pyflink.ml.core.windows import GlobalWindows, CountTumblingWindows


class TestParams(HasDistanceMeasure, HasFeaturesCol, HasGlobalBatchSize, HasHandleInvalid,
                 HasInputCols, HasLabelCol, HasLearningRate, HasMaxIter, HasMultiClass,
                 HasOutputCols, HasPredictionCol, HasRawPredictionCol, HasReg, HasSeed, HasTol,
                 HasWeightCol, HasWindows, HasRelativeError, HasFlatten, HasModelVersionCol,
                 HasMaxAllowedModelDelayMs):
    def __init__(self):
        self._param_map = {}

    def get_param_map(self) -> Dict['Param[Any]', Any]:
        return self._param_map


class ParamTests(unittest.TestCase):
    def test_distance_measure_param(self):
        param = TestParams()
        distance_measure = param.DISTANCE_MEASURE
        self.assertEqual(distance_measure.name, "distance_measure")
        self.assertEqual(distance_measure.description,
                         "Distance measure. Supported options: "
                         "'euclidean', 'manhattan' and 'cosine'.")
        self.assertEqual(distance_measure.default_value, "euclidean")

        param.set_distance_measure("cosine")
        self.assertEqual(param.get_distance_measure(), "cosine")

    def test_feature_col_param(self):
        param = TestParams()
        feature_col = param.FEATURES_COL
        self.assertEqual(feature_col.name, "features_col")
        self.assertEqual(feature_col.description, "Features column name.")
        self.assertEqual(feature_col.default_value, "features")

        param.set_features_col("test_features")
        self.assertEqual(param.get_features_col(), "test_features")

    def test_global_batch_size_param(self):
        param = TestParams()
        global_batch_size = param.GLOBAL_BATCH_SIZE
        self.assertEqual(global_batch_size.name, "global_batch_size")
        self.assertEqual(global_batch_size.description,
                         "Global batch size of training algorithms.")
        self.assertEqual(global_batch_size.default_value, 32)

        param.set_global_batch_size(100)
        self.assertEqual(param.get_global_batch_size(), 100)

    def test_handle_invalid_param(self):
        param = TestParams()
        handle_invalid = param.HANDLE_INVALID
        self.assertEqual(handle_invalid.name, "handle_invalid")
        self.assertEqual(handle_invalid.description, "Strategy to handle invalid entries.")
        self.assertEqual(handle_invalid.default_value, "error")

        param.set_handle_invalid("skip")
        self.assertEqual(param.get_handle_invalid(), "skip")

    def test_input_cols_param(self):
        param = TestParams()
        input_cols = param.INPUT_COLS
        self.assertEqual(input_cols.name, "input_cols")
        self.assertEqual(input_cols.description, "Input column names.")
        self.assertEqual(input_cols.default_value, None)

        param.set_input_cols('a', 'b', 'c')
        self.assertEqual(param.get_input_cols(), ('a', 'b', 'c'))

    def test_label_col_param(self):
        param = TestParams()
        label_col = param.LABEL_COL
        self.assertEqual(label_col.name, "label_col")
        self.assertEqual(label_col.description, "Label column name.")
        self.assertEqual(label_col.default_value, "label")

        param.set_label_col('test_label')
        self.assertEqual(param.get_label_col(), 'test_label')

    def test_learning_rate_param(self):
        param = TestParams()
        learning_rate = param.LEARNING_RATE
        self.assertEqual(learning_rate.name, "learning_rate")
        self.assertEqual(learning_rate.description, "Learning rate of optimization method.")
        self.assertEqual(learning_rate.default_value, 0.1)

        param.set_learning_rate(0.2)
        self.assertEqual(param.get_learning_rate(), 0.2)

    def test_max_iter_param(self):
        param = TestParams()
        max_iter = param.MAX_ITER
        self.assertEqual(max_iter.name, "max_iter")
        self.assertEqual(max_iter.description, "Maximum number of iterations.")
        self.assertEqual(max_iter.default_value, 20)

        param.set_max_iter(50)
        self.assertEqual(param.get_max_iter(), 50)

    def test_multi_class_param(self):
        param = TestParams()
        multi_class = param.MULTI_CLASS
        self.assertEqual(multi_class.name, "multi_class")
        self.assertEqual(multi_class.description,
                         "Classification type. Supported options: "
                         "'auto', 'binomial' and 'multinomial'.")
        self.assertEqual(multi_class.default_value, 'auto')

        param.set_multi_class('binomial')
        self.assertEqual(param.get_multi_class(), 'binomial')

    def test_output_cols_param(self):
        param = TestParams()
        output_cols = param.OUTPUT_COLS
        self.assertEqual(output_cols.name, "output_cols")
        self.assertEqual(output_cols.description, "Output column names.")
        self.assertEqual(output_cols.default_value, None)

        param.set_output_cols('a', 'b')
        self.assertEqual(param.get_output_cols(), ('a', 'b'))

    def test_prediction_col_param(self):
        param = TestParams()
        prediction_col = param.PREDICTION_COL
        self.assertEqual(prediction_col.name, "prediction_col")
        self.assertEqual(prediction_col.description, "Prediction column name.")
        self.assertEqual(prediction_col.default_value, "prediction")

        param.set_prediction_col('test_prediction')
        self.assertEqual(param.get_prediction_col(), 'test_prediction')

    def test_raw_prediction_col_param(self):
        param = TestParams()
        raw_prediction_col = param.RAW_PREDICTION_COL
        self.assertEqual(raw_prediction_col.name, "raw_prediction_col")
        self.assertEqual(raw_prediction_col.description, "Raw prediction column name.")
        self.assertEqual(raw_prediction_col.default_value, "raw_prediction")

        param.set_raw_prediction_col('test_raw_prediction')
        self.assertEqual(param.get_raw_prediction_col(), 'test_raw_prediction')

    def test_reg_param(self):
        param = TestParams()
        reg = param.REG
        self.assertEqual(reg.name, "reg")
        self.assertEqual(reg.description, "Regularization parameter.")
        self.assertEqual(reg.default_value, 0.)

        param.set_reg(0.4)
        self.assertEqual(param.get_reg(), 0.4)

    def test_seed_param(self):
        param = TestParams()
        seed = param.SEED
        self.assertEqual(seed.name, "seed")
        self.assertEqual(seed.description, "The random seed.")
        self.assertEqual(seed.default_value, None)

        param.set_seed(1)
        self.assertEqual(param.get_seed(), 1)

    def test_tol(self):
        param = TestParams()
        tol = param.TOL
        self.assertEqual(tol.name, "tol")
        self.assertEqual(tol.description, "Convergence tolerance for iterative algorithms.")
        self.assertEqual(tol.default_value, 1e-6)

        param.set_tol(1e-5)
        self.assertEqual(param.get_tol(), 1e-5)

    def test_weight_col(self):
        param = TestParams()
        weight_col = param.WEIGHT_COL
        self.assertEqual(weight_col.name, "weight_col")
        self.assertEqual(weight_col.description, "Weight column name.")
        self.assertEqual(weight_col.default_value, None)

        param.set_weight_col('test_weight_col')
        self.assertEqual(param.get_weight_col(), 'test_weight_col')

    def test_windows(self):
        param = TestParams()
        windows = param.WINDOWS
        self.assertEqual(windows.name, "windows")
        self.assertEqual(windows.description,
                         "Windowing strategy that determines how to create "
                         "mini-batches from input data.")
        self.assertEqual(windows.default_value, GlobalWindows())

        param.set_windows(CountTumblingWindows.of(100))
        self.assertEqual(param.get_windows(), CountTumblingWindows.of(100))

    def test_relative_error(self):
        param = TestParams()
        relative_error = param.RELATIVE_ERROR
        self.assertEqual(relative_error.name, "relative_error")
        self.assertEqual(relative_error.description,
                         "The relative target precision for the approximate"
                         " quantile algorithm.")
        self.assertEqual(relative_error.default_value, 0.001)

        param.set_relative_error(0.1)
        self.assertEqual(param.get_relative_error(), 0.1)

    def test_flatten(self):
        param = TestParams()
        flatten = param.FLATTEN
        self.assertEqual(flatten.name, "flatten")
        self.assertEqual(flatten.description,
                         "If false, the returned table contains only a "
                         "single row, otherwise, one row per feature.")
        self.assertFalse(flatten.default_value)

        param.set_flatten(True)
        self.assertTrue(param.get_flatten())

    def test_model_version_col(self):
        param = TestParams()
        model_version_col = param.MODEL_VERSION_COL
        self.assertEqual(model_version_col.name, "model_version_col")
        self.assertEqual(model_version_col.description,
                         "The name of the column which contains the version of "
                         "the model data that the input data is predicted with. "
                         "The version should be a 64-bit integer.",)
        self.assertEqual(model_version_col.default_value, "version")

        param.set_model_version_col("test_version")
        self.assertEqual(param.get_model_version_col(), "test_version")

    def test_max_allowed_model_delay_ms(self):
        param = TestParams()
        max_allowed_model_delay_ms = param.MAX_ALLOWED_MODEL_DELAY_MS
        self.assertEqual(max_allowed_model_delay_ms.name, "max_allowed_model_delay_ms")
        self.assertEqual(max_allowed_model_delay_ms.description,
                         "The maximum difference allowed between the timestamps of the "
                         "input record and the model data that is used to predict that "
                         "input record. This param only works when the input contains "
                         "event time.")
        self.assertEqual(max_allowed_model_delay_ms.default_value, 0)

        param.set_max_allowed_model_delay_ms(100)
        self.assertEqual(param.get_max_allowed_model_delay_ms(), 100)
