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
from abc import ABC

from pyflink.ml.param import Param, ParamValidators, FloatParam
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.classification.common import (JavaClassificationModel,
                                              JavaClassificationEstimator)
from pyflink.ml.common.param import (HasFeaturesCol, HasPredictionCol, HasLabelCol,
                                     HasRawPredictionCol, HasWeightCol, HasMaxIter, HasReg,
                                     HasElasticNet, HasLearningRate, HasGlobalBatchSize, HasTol)


class _LinearSVCModelParams(
    JavaWithParams,
    HasFeaturesCol,
    HasPredictionCol,
    HasRawPredictionCol,
    ABC
):
    """
    Params for :class:`LinearSVCModel`.
    """

    THRESHOLD: Param[float] = FloatParam(
        "threshold",
        "Threshold in binary classification prediction applied to rawPrediction.",
        0.0,
        ParamValidators.not_null())

    def __init__(self, java_params):
        super(_LinearSVCModelParams, self).__init__(java_params)

    def set_threshold(self, value: int):
        return typing.cast(_LinearSVCModelParams, self.set(self.THRESHOLD, value))

    def get_threshold(self) -> int:
        return self.get(self.THRESHOLD)

    @property
    def threshold(self) -> int:
        return self.get_threshold()


class _LinearSVCParams(
    _LinearSVCModelParams,
    HasLabelCol,
    HasWeightCol,
    HasMaxIter,
    HasReg,
    HasElasticNet,
    HasLearningRate,
    HasGlobalBatchSize,
    HasTol,
):
    """
    Params for :class:`LinearSVC`.
    """

    def __init__(self, java_params):
        super(_LinearSVCParams, self).__init__(java_params)


class LinearSVCModel(JavaClassificationModel, _LinearSVCModelParams):
    """
    A Model which classifies data using the model data computed by :class:`LinearSVC`.
    """

    def __init__(self, java_model=None):
        super(LinearSVCModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "linearsvc"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "LinearSVCModel"


class LinearSVC(JavaClassificationEstimator, _LinearSVCParams):
    """
    An Estimator which implements the linear support vector classification.

    See: https://en.wikipedia.org/wiki/Support-vector_machine#Linear_SVM.
    """

    def __init__(self):
        super(LinearSVC, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> LinearSVCModel:
        return LinearSVCModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "linearsvc"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "LinearSVC"
