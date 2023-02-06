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

from pyflink.ml.param import Param, FloatParam, ParamValidators
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureModel, JavaFeatureEstimator
from pyflink.ml.common.param import HasInputCol, HasOutputCol


class _MinMaxScalerParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
    Params for :class:`MinMaxScaler`.
    """

    MIN: Param[float] = FloatParam(
        "min",
        "Lower bound of the output feature range.",
        0.0,
        ParamValidators.not_null())

    MAX: Param[float] = FloatParam(
        "max",
        "Upper bound of the output feature range.",
        1.0,
        ParamValidators.not_null())

    def __init__(self, java_params):
        super(_MinMaxScalerParams, self).__init__(java_params)

    def set_min(self, value: float):
        return typing.cast(_MinMaxScalerParams, self.set(self.MIN, value))

    def set_max(self, value: float):
        return typing.cast(_MinMaxScalerParams, self.set(self.MAX, value))

    def get_min(self) -> bool:
        return self.get(self.MIN)

    def get_max(self) -> bool:
        return self.get(self.MAX)

    @property
    def min(self):
        return self.get_min()

    @property
    def max(self):
        return self.get_max()


class MinMaxScalerModel(JavaFeatureModel, _MinMaxScalerParams):
    """
    A Model which transforms data using the model data computed by :class:`MinMaxScaler`.
    """

    def __init__(self, java_model=None):
        super(MinMaxScalerModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "minmaxscaler"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "MinMaxScalerModel"


class MinMaxScaler(JavaFeatureEstimator, _MinMaxScalerParams):
    """
    An Estimator which implements the MinMaxScaler algorithm. This algorithm rescales feature values
    to a common range [min, max] which defined by user.

    $$ Rescaled(value) = frac{value - E_{min}}{E_{max} - E_{min}} * (max - min) + min $$

    For the case (E_{max} == E_{min}), (Rescaled(value) = 0.5 * (max + min)).

    See https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization).
    """

    def __init__(self):
        super(MinMaxScaler, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> MinMaxScalerModel:
        return MinMaxScalerModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "minmaxscaler"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "MinMaxScaler"
