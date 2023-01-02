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

from pyflink.ml.core.param import Param, BooleanParam
from pyflink.ml.core.wrapper import JavaWithParams
from pyflink.ml.lib.feature.common import JavaFeatureEstimator, JavaFeatureModel
from pyflink.ml.lib.param import HasInputCol, HasOutputCol


class _StandardScalerParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
    Params for :class:`StandardScaler`.
    """

    WITH_MEAN: Param[bool] = BooleanParam(
        "with_mean",
        "Whether centers the data with mean before scaling.",
        False)

    WITH_STD: Param[bool] = BooleanParam(
        "with_std",
        "Whether scales the data with standard deviation.",
        True)

    def __init__(self, java_params):
        super(_StandardScalerParams, self).__init__(java_params)

    def set_with_mean(self, value: bool):
        return typing.cast(_StandardScalerParams, self.set(self.WITH_MEAN, value))

    def set_with_std(self, value: bool):
        return typing.cast(_StandardScalerParams, self.set(self.WITH_STD, value))

    def get_with_mean(self) -> bool:
        return self.get(self.WITH_MEAN)

    def get_with_std(self) -> bool:
        return self.get(self.WITH_STD)

    @property
    def with_mean(self):
        return self.get_with_mean()

    @property
    def with_std(self):
        return self.get_with_std()


class StandardScalerModel(JavaFeatureModel, _StandardScalerParams):
    """
    A Model which classifies data using the model data computed by :class:`StandardScaler`.
    """

    def __init__(self, java_model=None):
        super(StandardScalerModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "standardscaler"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "StandardScalerModel"


class StandardScaler(JavaFeatureEstimator, _StandardScalerParams):
    """
    An Estimator which implements the standard scaling algorithm.

    Standardization is a common requirement for machine learning training because they may behave
    badly if the individual features of an input do not look like standard normally distributed data
    (e.g. Gaussian with 0 mean and unit variance).

    This estimator standardizes the input features by removing the mean and scaling each dimension
    to unit variance.
    """

    def __init__(self):
        super(StandardScaler, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> StandardScalerModel:
        return StandardScalerModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "standardscaler"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "StandardScaler"
