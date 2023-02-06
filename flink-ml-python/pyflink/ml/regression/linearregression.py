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

from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.regression.common import (JavaRegressionModel, JavaRegressionEstimator)
from pyflink.ml.common.param import (HasWeightCol, HasMaxIter, HasReg, HasLearningRate,
                                     HasGlobalBatchSize, HasTol, HasFeaturesCol,
                                     HasPredictionCol, HasLabelCol, HasElasticNet)


class _LinearRegressionModelParams(
    JavaWithParams,
    HasFeaturesCol,
    HasPredictionCol,
    ABC
):
    """
    Params for :class:`LinearRegressionModel`.
    """

    def __init__(self, java_params):
        super(_LinearRegressionModelParams, self).__init__(java_params)


class _LinearRegressionParams(
    _LinearRegressionModelParams,
    HasLabelCol,
    HasWeightCol,
    HasMaxIter,
    HasReg,
    HasElasticNet,
    HasLearningRate,
    HasGlobalBatchSize,
    HasTol
):
    """
    Params for :class:`LinearRegression`.
    """

    def __init__(self, java_params):
        super(_LinearRegressionParams, self).__init__(java_params)


class LinearRegressionModel(JavaRegressionModel, _LinearRegressionModelParams):
    """
    A Model which classifies data using the model data computed by :class:`LinearRegression`.
    """

    def __init__(self, java_model=None):
        super(LinearRegressionModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "linearregression"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "LinearRegressionModel"


class LinearRegression(JavaRegressionEstimator, _LinearRegressionParams):
    """
    An Estimator which implements the linear regression algorithm.

    See https://en.wikipedia.org/wiki/Linear_regression.
    """

    def __init__(self):
        super(LinearRegression, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> LinearRegressionModel:
        return LinearRegressionModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "linearregression"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "LinearRegression"
