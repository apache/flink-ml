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
import typing

from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.common.param import HasInputCol, HasOutputCol, HasRelativeError
from pyflink.ml.param import BooleanParam, Param, FloatParam, ParamValidators

from pyflink.ml.feature.common import JavaFeatureModel, JavaFeatureEstimator


class _RobustScalerModelParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
    Params for :class `RobustScalerModel`.
    """
    WITH_CENTERING: Param[bool] = BooleanParam(
        "with_centering",
        "Whether to center the data with median before scaling.",
        False
    )

    WITH_SCALING: Param[bool] = BooleanParam(
        "with_scaling",
        "Whether to scale the data to quantile range.",
        True
    )

    def __init__(self, java_params):
        super(_RobustScalerModelParams, self).__init__(java_params)

    def set_with_centering(self, value: bool):
        return typing.cast(_RobustScalerModelParams, self.set(self.WITH_CENTERING, value))

    def get_with_centering(self):
        return self.get(self.WITH_CENTERING)

    def set_with_scaling(self, value: bool):
        return typing.cast(_RobustScalerModelParams, self.set(self.WITH_SCALING, value))

    def get_with_scaling(self):
        return self.get(self.WITH_SCALING)

    @property
    def with_centering(self):
        return self.get_with_centering()

    @property
    def with_scaling(self):
        return self.get_with_scaling()


class _RobustScalerParams(HasRelativeError, _RobustScalerModelParams):
    """
    Params for :class `RobustScaler`.
    """
    LOWER: Param[float] = FloatParam(
        "lower",
        "Lower quantile to calculate quantile range.",
        0.25,
        ParamValidators.in_range(0.0, 1.0, False, False)
    )

    UPPER: Param[float] = FloatParam(
        "upper",
        "Upper quantile to calculate quantile range.",
        0.75,
        ParamValidators.in_range(0.0, 1.0, False, False)
    )

    def __init__(self, java_params):
        super(_RobustScalerParams, self).__init__(java_params)

    def set_lower(self, value: float):
        return typing.cast(_RobustScalerParams, self.set(self.LOWER, value))

    def get_lower(self):
        return self.get(self.LOWER)

    def set_upper(self, value: float):
        return typing.cast(_RobustScalerParams, self.set(self.UPPER, value))

    def get_upper(self):
        return self.get(self.UPPER)

    @property
    def lower(self):
        return self.get_lower()

    @property
    def upper(self):
        return self.get_upper()


class RobustScalerModel(JavaFeatureModel, _RobustScalerModelParams):
    """
    A Model which transforms data using the model data computed by :class::RobustScaler.
    """

    def __init__(self, java_model=None):
        super(RobustScalerModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "robustscaler"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "RobustScalerModel"


class RobustScaler(JavaFeatureEstimator, _RobustScalerParams):
    """
    An Estimator which scales features using statistics that are robust to outliers.

    This Scaler removes the median and scales the data according to the quantile
    range (defaults to IQR: Interquartile Range). The IQR is the range between the 1st
    quartile (25th quantile) and the 3rd quartile (75th quantile) but can be configured.

    Centering and scaling happen independently on each feature by computing the relevant
    statistics on the samples in the training set. Median and quantile range are then
    stored to be used on later data using the transform method.

    Standardization of a dataset is a common requirement for many machine learning estimators.
    Typically this is done by removing the mean and scaling to unit variance. However, outliers can
    often influence the sample mean / variance in a negative way. In such cases, the median and the
    interquartile range often give better results.

    Note that NaN values are ignored in the computation of medians and ranges.
    """

    def __init__(self):
        super(RobustScaler, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> RobustScalerModel:
        return RobustScalerModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "robustscaler"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "RobustScaler"
