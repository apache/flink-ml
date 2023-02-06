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

import typing

from pyflink.ml.param import Param, StringParam, ParamValidators, FloatParam
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.classification.common import (JavaClassificationModel,
                                              JavaClassificationEstimator)
from pyflink.ml.common.param import HasFeaturesCol, HasPredictionCol, HasLabelCol


class _NaiveBayesModelParams(
    JavaWithParams,
    HasFeaturesCol,
    HasPredictionCol,
    ABC
):
    """
    Params for :class:`NaiveBayesModel`.
    """

    MODEL_TYPE: Param[str] = StringParam(
        "model_type",
        "The model type.",
        "multinomial",
        ParamValidators.in_array(["multinomial"]))

    def __init__(self, java_params):
        super(_NaiveBayesModelParams, self).__init__(java_params)

    def set_model_type(self, value: str):
        return self.set(self.MODEL_TYPE, value)

    def get_model_type(self) -> str:
        return self.get(self.MODEL_TYPE)

    @property
    def model_type(self) -> str:
        return self.get_model_type()


class _NaiveBayesParams(
    _NaiveBayesModelParams,
    HasLabelCol,
):
    """
    Params for :class:`NaiveBayes`.
    """

    SMOOTHING: Param[float] = FloatParam(
        "smoothing",
        "The smoothing parameter.",
        1.0,
        ParamValidators.gt_eq(0.0))

    def __init__(self, java_params):
        super(_NaiveBayesParams, self).__init__(java_params)

    def set_smoothing(self, value: float):
        return typing.cast(_NaiveBayesParams, self.set(self.SMOOTHING, value))

    def get_smoothing(self) -> float:
        return self.get(self.SMOOTHING)

    @property
    def smoothing(self) -> float:
        return self.get_smoothing()


class NaiveBayesModel(JavaClassificationModel, _NaiveBayesModelParams):
    """
    A Model which classifies data using the model data computed by :class:`NaiveBayes`.
    """

    def __init__(self, java_model=None):
        super(NaiveBayesModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "naivebayes"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "NaiveBayesModel"


class NaiveBayes(JavaClassificationEstimator, _NaiveBayesParams):
    """
    An Estimator which implements the naive bayes classification algorithm.

    See https://en.wikipedia.org/wiki/Naive_Bayes_classifier.
    """

    def __init__(self):
        super(NaiveBayes, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> NaiveBayesModel:
        return NaiveBayesModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "naivebayes"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "NaiveBayes"
