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


class _VarianceThresholdSelectorModelParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
    Params for :class:`VarianceThresholdSelectorModel`.
    """

    def __init__(self, java_params):
        super(_VarianceThresholdSelectorModelParams, self).__init__(java_params)


class _VarianceThresholdSelectorParams(_VarianceThresholdSelectorModelParams):
    """
    Params for :class:`VarianceThresholdSelector`.
    """

    VARIANCE_THRESHOLD: Param[float] = FloatParam(
        "variance_threshold",
        "Features with a variance not greater than this threshold will be removed.",
        0.0,
        ParamValidators.gt_eq(0.0))

    def __init__(self, java_params):
        super(_VarianceThresholdSelectorParams, self).__init__(java_params)

    def set_variance_threshold(self, value: float):
        return typing.cast(_VarianceThresholdSelectorParams,
                           self.set(self.VARIANCE_THRESHOLD, value))

    def get_variance_threshold(self):
        return self.get(self.VARIANCE_THRESHOLD)

    @property
    def variance_threshold(self):
        return self.get_variance_threshold()


class VarianceThresholdSelectorModel(JavaFeatureModel, _VarianceThresholdSelectorModelParams):
    """
    A Model which transforms data using the model data
    computed by :class:`VarianceThresholdSelector`.
    """

    def __init__(self, java_model=None):
        super(VarianceThresholdSelectorModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "variancethresholdselector"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "VarianceThresholdSelectorModel"


class VarianceThresholdSelector(JavaFeatureEstimator, _VarianceThresholdSelectorParams):
    """
    An Estimator which implements the VarianceThresholdSelector algorithm. The algorithm
    removes all low-variance features. Features with a variance not greater than the
    threshold will be removed. The default is to keep all features with non-zero variance,
    i.e. remove the features that have the same value in all samples.
    """

    def __init__(self):
        super(VarianceThresholdSelector, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> VarianceThresholdSelectorModel:
        return VarianceThresholdSelectorModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "variancethresholdselector"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "VarianceThresholdSelector"
