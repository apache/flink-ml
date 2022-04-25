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

from pyflink.ml.core.wrapper import JavaWithParams
from pyflink.ml.lib.classification.common import (JavaClassificationModel,
                                                  JavaClassificationEstimator)
from pyflink.ml.lib.param import (HasWeightCol, HasMaxIter, HasReg, HasLearningRate,
                                  HasGlobalBatchSize, HasTol, HasMultiClass, HasFeaturesCol,
                                  HasPredictionCol, HasRawPredictionCol, HasLabelCol)


class _LogisticRegressionModelParams(
    JavaWithParams,
    HasFeaturesCol,
    HasPredictionCol,
    HasRawPredictionCol,
    ABC
):
    """
    Params for :class:`LogisticRegressionModel`.
    """

    def __init__(self, java_params):
        super(_LogisticRegressionModelParams, self).__init__(java_params)


class _LogisticRegressionParams(
    _LogisticRegressionModelParams,
    HasLabelCol,
    HasWeightCol,
    HasMaxIter,
    HasReg,
    HasLearningRate,
    HasGlobalBatchSize,
    HasTol,
    HasMultiClass
):
    """
    Params for :class:`LogisticRegression`.
    """

    def __init__(self, java_params):
        super(_LogisticRegressionParams, self).__init__(java_params)


class LogisticRegressionModel(JavaClassificationModel, _LogisticRegressionModelParams):
    """
    A Model which classifies data using the model data computed by :class:`LogisticRegression`.
    """

    def __init__(self, java_model=None):
        super(LogisticRegressionModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "logisticregression"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "LogisticRegressionModel"


class LogisticRegression(JavaClassificationEstimator, _LogisticRegressionParams):
    """
    An Estimator which implements the logistic regression algorithm.

    See https://en.wikipedia.org/wiki/Logistic_regression.
    """

    def __init__(self):
        super(LogisticRegression, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> LogisticRegressionModel:
        return LogisticRegressionModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "logisticregression"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "LogisticRegression"
