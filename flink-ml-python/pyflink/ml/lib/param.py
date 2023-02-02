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
from abc import ABC
from typing import Tuple

from pyflink.ml.core.param import WithParams, Param, ParamValidators, StringParam, IntParam, \
    StringArrayParam, FloatParam, WindowsParam, BooleanParam
from pyflink.ml.core.windows import Windows, GlobalWindows


class HasDistanceMeasure(WithParams, ABC):
    """
    Base class for the shared distance_measure param.
    """
    DISTANCE_MEASURE: Param[str] = StringParam(
        "distance_measure",
        "Distance measure. Supported options: 'euclidean', 'manhattan' and 'cosine'.",
        "euclidean",
        ParamValidators.in_array(['euclidean', 'manhattan', 'cosine']))

    def set_distance_measure(self, distance_measure: str):
        return self.set(self.DISTANCE_MEASURE, distance_measure)

    def get_distance_measure(self) -> str:
        return self.get(self.DISTANCE_MEASURE)

    @property
    def distance_measure(self) -> str:
        return self.get_distance_measure()


class HasFeaturesCol(WithParams, ABC):
    """
    Base class for the shared feature_col param.

    `HasFeaturesCol` is typically used for `Stage`s that implement `HasLabelCol`. It is preferred
    to use `HasInputCol` for other cases.
    """
    FEATURES_COL: Param[str] = StringParam(
        "features_col",
        "Features column name.",
        "features",
        ParamValidators.not_null())

    def set_features_col(self, col):
        return self.set(self.FEATURES_COL, col)

    def get_features_col(self) -> str:
        return self.get(self.FEATURES_COL)

    @property
    def features_col(self) -> str:
        return self.get_features_col()


class HasGlobalBatchSize(WithParams, ABC):
    """
    Base class for the shared global_batch_size param.
    """
    GLOBAL_BATCH_SIZE: Param[int] = IntParam(
        "global_batch_size",
        "Global batch size of training algorithms.",
        32,
        ParamValidators.gt(0))

    def set_global_batch_size(self, global_batch_size: int):
        return self.set(self.GLOBAL_BATCH_SIZE, global_batch_size)

    def get_global_batch_size(self) -> int:
        return self.get(self.GLOBAL_BATCH_SIZE)

    @property
    def global_batch_size(self) -> int:
        return self.get_global_batch_size()


class HasHandleInvalid(WithParams, ABC):
    """
    Base class for the shared handle_invalid param.

    Supported options and the corresponding behavior to handle invalid entries is listed as follows.

    <ul>
        <li>error: raise an exception.
        <li>skip: filter out rows with bad values.
    </ul>
    """
    HANDLE_INVALID: Param[str] = StringParam(
        "handle_invalid",
        "Strategy to handle invalid entries.",
        "error",
        ParamValidators.in_array(['error', 'skip']))

    def set_handle_invalid(self, value: str):
        return self.set(self.HANDLE_INVALID, value)

    def get_handle_invalid(self) -> str:
        return self.get(self.HANDLE_INVALID)

    @property
    def handle_invalid(self) -> str:
        return self.get_handle_invalid()


class HasInputCol(WithParams, ABC):
    """
    Base class for the shared input col param.
    """
    INPUT_COL: Param[str] = StringParam(
        "input_col",
        "Input column name.",
        "input",
        ParamValidators.not_null())

    def set_input_col(self, col: str):
        return self.set(self.INPUT_COL, col)

    def get_input_col(self) -> str:
        return self.get(self.INPUT_COL)

    @property
    def input_col(self) -> str:
        return self.get_input_col()


class HasInputCols(WithParams, ABC):
    """
    Base class for the shared input cols param.
    """
    INPUT_COLS: Param[Tuple[str, ...]] = StringArrayParam(
        "input_cols",
        "Input column names.",
        None,
        ParamValidators.non_empty_array())

    def set_input_cols(self, *cols: str):
        return self.set(self.INPUT_COLS, cols)

    def get_input_cols(self) -> Tuple[str, ...]:
        return self.get(self.INPUT_COLS)

    @property
    def input_cols(self) -> Tuple[str, ...]:
        return self.get_input_cols()


class HasCategoricalCols(WithParams, ABC):
    """
    Base class for the shared categorical cols param.
    """
    CATEGORICAL_COLS: Param[Tuple[str, ...]] = StringArrayParam(
        "categorical_cols",
        "Categorical column names.",
        [],
        ParamValidators.not_null())

    def set_categorical_cols(self, *cols: str):
        return self.set(self.CATEGORICAL_COLS, cols)

    def get_categorical_cols(self) -> Tuple[str, ...]:
        return self.get(self.CATEGORICAL_COLS)

    @property
    def categorical_cols(self) -> Tuple[str, ...]:
        return self.get_categorical_cols()


class HasNumFeatures(WithParams, ABC):
    """
    Base class for the shared numFeatures param.
    """
    NUM_FEATURES: Param[int] = IntParam(
        "num_features",
        "Number of features.",
        262144,
        ParamValidators.gt(0))

    def set_num_features(self, num_features: int):
        return self.set(self.NUM_FEATURES, num_features)

    def get_num_features(self) -> int:
        return self.get(self.NUM_FEATURES)

    @property
    def num_features(self) -> int:
        return self.get_num_features()


class HasLabelCol(WithParams, ABC):
    """
    Base class for the shared label column param.
    """
    LABEL_COL: Param[str] = StringParam(
        "label_col",
        "Label column name.",
        "label",
        ParamValidators.not_null())

    def set_label_col(self, col: str):
        return self.set(self.LABEL_COL, col)

    def get_label_col(self) -> str:
        return self.get(self.LABEL_COL)

    @property
    def label_col(self) -> str:
        return self.get_label_col()


class HasLearningRate(WithParams, ABC):
    """
    Base class for the shared learning rate param.
    """

    LEARNING_RATE: Param[float] = FloatParam(
        "learning_rate",
        "Learning rate of optimization method.",
        0.1,
        ParamValidators.gt(0))

    def set_learning_rate(self, learning_rate: float):
        return self.set(self.LEARNING_RATE, learning_rate)

    def get_learning_rate(self) -> float:
        return self.get(self.LEARNING_RATE)

    @property
    def learning_rate(self) -> float:
        return self.get_learning_rate()


class HasMaxIter(WithParams, ABC):
    """
    Base class for the shared maxIter param.
    """
    MAX_ITER: Param[int] = IntParam(
        "max_iter",
        "Maximum number of iterations.",
        20,
        ParamValidators.gt(0))

    def set_max_iter(self, max_iter: int):
        return self.set(self.MAX_ITER, max_iter)

    def get_max_iter(self) -> int:
        return self.get(self.MAX_ITER)

    @property
    def max_iter(self) -> int:
        return self.get_max_iter()


class HasMultiClass(WithParams, ABC):
    """
    Base class for the shared multi class param.

    Supported options:
        <li>auto: selects the classification type based on the number of classes:
            If the number of unique label values from the input data is one or two,
            set to "binomial". Otherwise, set to "multinomial".
        <li>binomial: binary logistic regression.
        <li>multinomial: multinomial logistic regression.
    """
    MULTI_CLASS: Param[str] = StringParam(
        "multi_class",
        "Classification type. Supported options: 'auto', 'binomial' and 'multinomial'.",
        'auto',
        ParamValidators.in_array(['auto', 'binomial', 'multinomial']))

    def set_multi_class(self, class_type: str):
        return self.set(self.MULTI_CLASS, class_type)

    def get_multi_class(self) -> str:
        return self.get(self.MULTI_CLASS)

    @property
    def multi_class(self) -> str:
        return self.get_multi_class()


class HasOutputCol(WithParams, ABC):
    """
    Base class for the shared output_col param.
    """
    OUTPUT_COL: Param[str] = StringParam(
        "output_col",
        "Output column name.",
        "output",
        ParamValidators.not_null())

    def set_output_col(self, col: str):
        return self.set(self.OUTPUT_COL, col)

    def get_output_col(self) -> str:
        return self.get(self.OUTPUT_COL)

    @property
    def output_col(self) -> str:
        return self.get_output_col()


class HasOutputCols(WithParams, ABC):
    """
    Base class for the shared output_cols param.
    """
    OUTPUT_COLS: Param[Tuple[str, ...]] = StringArrayParam(
        "output_cols",
        "Output column names.",
        None,
        ParamValidators.non_empty_array())

    def set_output_cols(self, *cols: str):
        return self.set(self.OUTPUT_COLS, cols)

    def get_output_cols(self) -> Tuple[str, ...]:
        return self.get(self.OUTPUT_COLS)

    @property
    def output_cols(self) -> Tuple[str, ...]:
        return self.get_output_cols()


class HasPredictionCol(WithParams, ABC):
    """
    Base class for the shared prediction column param.
    """
    PREDICTION_COL: Param[str] = StringParam(
        "prediction_col",
        "Prediction column name.",
        "prediction",
        ParamValidators.not_null())

    def set_prediction_col(self, col: str):
        return self.set(self.PREDICTION_COL, col)

    def get_prediction_col(self) -> str:
        return self.get(self.PREDICTION_COL)

    @property
    def prediction_col(self) -> str:
        return self.get_prediction_col()


class HasRawPredictionCol(WithParams, ABC):
    """
    Base class for the shared raw prediction column param.
    """
    RAW_PREDICTION_COL: Param[str] = StringParam(
        "raw_prediction_col",
        "Raw prediction column name.",
        "raw_prediction")

    def set_raw_prediction_col(self, col: str):
        return self.set(self.RAW_PREDICTION_COL, col)

    def get_raw_prediction_col(self):
        return self.get(self.RAW_PREDICTION_COL)

    @property
    def raw_prediction_col(self) -> str:
        return self.get_raw_prediction_col()


class HasReg(WithParams, ABC):
    """
    Base class for the shared regularization param.
    """
    REG: Param[float] = FloatParam(
        "reg",
        "Regularization parameter.",
        0.,
        ParamValidators.gt_eq(0.))

    def set_reg(self, value: float):
        return self.set(self.REG, value)

    def get_reg(self) -> float:
        return self.get(self.REG)

    @property
    def reg(self) -> float:
        return self.get_reg()


class HasSeed(WithParams, ABC):
    """
    Base class for the shared seed param.
    """
    SEED: Param[int] = IntParam(
        "seed",
        "The random seed.",
        None)

    def set_seed(self, seed: int):
        return self.set(self.SEED, seed) if seed is not None else hash(self.__class__.__name__)

    def get_seed(self) -> int:
        return self.get(self.SEED)

    @property
    def seed(self) -> int:
        return self.get_seed()


class HasTol(WithParams, ABC):
    """
    Base class for the shared tolerance param.
    """
    TOL: Param[float] = FloatParam(
        "tol",
        "Convergence tolerance for iterative algorithms.",
        1e-6,
        ParamValidators.gt_eq(0))

    def set_tol(self, value: float):
        return self.set(self.TOL, value)

    def get_tol(self) -> float:
        return self.get(self.TOL)

    @property
    def tol(self) -> float:
        return self.get_tol()


class HasWeightCol(WithParams, ABC):
    """
    Base class for the shared weight column param. If this is not set, we treat all instance weights
    as 1.0.
    """
    WEIGHT_COL: Param[str] = StringParam(
        "weight_col",
        "Weight column name.",
        None)

    def set_weight_col(self, col: str):
        return self.set(self.WEIGHT_COL, col)

    def get_weight_col(self) -> str:
        return self.get(self.WEIGHT_COL)

    @property
    def weight_col(self):
        return self.get_weight_col()


class HasBatchStrategy(WithParams, ABC):
    """
    Base class for the shared batch strategy param.
    """
    BATCH_STRATEGY: Param[str] = StringParam(
        "batch_strategy",
        "Strategy to create mini batch from online train data.",
        "count",
        ParamValidators.in_array(["count"]))

    def get_batch_strategy(self) -> str:
        return self.get(self.BATCH_STRATEGY)

    @property
    def batch_strategy(self):
        return self.get_batch_strategy()


class HasDecayFactor(WithParams, ABC):
    """
    Base class for the shared decay factor param.
    """
    DECAY_FACTOR: Param[float] = FloatParam(
        "decay_factor",
        "The forgetfulness of the previous centroids.",
        0.,
        ParamValidators.in_range(0, 1))

    def set_decay_factor(self, value: float):
        return self.set(self.DECAY_FACTOR, value)

    def get_decay_factor(self) -> float:
        return self.get(self.DECAY_FACTOR)

    @property
    def decay_factor(self):
        return self.get(self.DECAY_FACTOR)


class HasElasticNet(WithParams, ABC):
    """
    Base class for the shared decay factor param.
    """
    ELASTIC_NET: Param[float] = FloatParam(
        "elastic_net",
        "ElasticNet parameter.",
        0.,
        ParamValidators.in_range(0.0, 1.0))

    def set_elastic_net(self, value: float):
        return self.set(self.ELASTIC_NET, value)

    def get_elastic_net(self) -> float:
        return self.get(self.ELASTIC_NET)

    @property
    def elastic_net(self):
        return self.get(self.ELASTIC_NET)


class HasWindows(WithParams, ABC):
    """
    Base class for the shared windows param.
    """
    WINDOWS: Param[Windows] = WindowsParam(
        "windows",
        "Windowing strategy that determines how to create mini-batches from input data.",
        GlobalWindows(),
        ParamValidators.not_null())

    def set_windows(self, value: Windows):
        self.set(self.WINDOWS, value)
        return self

    def get_windows(self) -> Windows:
        return self.get(self.WINDOWS)

    @property
    def windows(self):
        return self.get(self.WINDOWS)


class HasRelativeError(WithParams, ABC):
    """
    Interface for shared param relativeError.
    """
    RELATIVE_ERROR: Param[float] = FloatParam(
        "relative_error",
        "The relative target precision for the approximate quantile algorithm.",
        0.001,
        ParamValidators.in_range(0.0, 1.0))

    def set_relative_error(self, value: float):
        return self.set(self.RELATIVE_ERROR, value)

    def get_relative_error(self) -> float:
        return self.get(self.RELATIVE_ERROR)

    @property
    def relative_error(self):
        return self.get(self.RELATIVE_ERROR)


class HasFlatten(WithParams, ABC):
    """
    Interface for shared flatten param.
    """
    FLATTEN: Param[bool] = BooleanParam(
        "flatten",
        "If false, the returned table contains only a single row, otherwise, one row per feature.",
        False
    )

    def set_flatten(self, value: bool):
        return self.set(self.FLATTEN, value)

    def get_flatten(self) -> bool:
        return self.get(self.FLATTEN)

    @property
    def flatten(self):
        return self.get(self.FLATTEN)


class HasModelVersionCol(WithParams, ABC):
    """
    Interface for the shared model version column param.
    """
    MODEL_VERSION_COL: Param[str] = StringParam(
        "model_version_col",
        "The name of the column which contains the version of the model data that "
        "the input data is predicted with. The version should be a 64-bit integer.",
        "version"
    )

    def set_model_version_col(self, value: str):
        return self.set(self.MODEL_VERSION_COL, value)

    def get_model_version_col(self) -> str:
        return self.get(self.MODEL_VERSION_COL)

    @property
    def model_version_col(self):
        return self.get_model_version_col()


class HasMaxAllowedModelDelayMs(WithParams, ABC):
    """
    Interface for the shared max allowed model delay in milliseconds param.
    """
    MAX_ALLOWED_MODEL_DELAY_MS: Param[int] = IntParam(
        "max_allowed_model_delay_ms",
        "The maximum difference allowed between the timestamps of the input record "
        "and the model data that is used to predict that input record. "
        "This param only works when the input contains event time.",
        0,
        ParamValidators.gt_eq(0)
    )

    def set_max_allowed_model_delay_ms(self, value: int):
        return self.set(self.MAX_ALLOWED_MODEL_DELAY_MS, value)

    def get_max_allowed_model_delay_ms(self) -> int:
        return self.get(self.MAX_ALLOWED_MODEL_DELAY_MS)

    @property
    def max_allowed_model_delay_ms(self):
        return self.get_max_allowed_model_delay_ms()
