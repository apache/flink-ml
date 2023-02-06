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

from pyflink.ml.param import (ParamValidators, Param, StringParam, FloatParam)
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.classification.common import (JavaClassificationModel,
                                              JavaClassificationEstimator)
from pyflink.ml.common.param import (HasWeightCol, HasMaxIter, HasReg, HasLearningRate,
                                     HasGlobalBatchSize, HasTol, HasMultiClass, HasFeaturesCol,
                                     HasPredictionCol, HasRawPredictionCol, HasLabelCol,
                                     HasBatchStrategy, HasElasticNet)


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
    HasElasticNet,
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


class _OnlineLogisticRegressionModelParams(
    JavaWithParams,
    HasFeaturesCol,
    HasPredictionCol,
    HasRawPredictionCol,
    ABC
):
    """
    Params for :class:`OnlineLogisticRegressionModel`.
    """
    MODEL_VERSION_COL: Param[str] = StringParam(
        "model_version_col",
        "Model version column name.",
        "model_version",
        ParamValidators.not_null())

    def __init__(self, java_params):
        super(_OnlineLogisticRegressionModelParams, self).__init__(java_params)

    def set_model_version_col(self, value: str):
        return self.set(self.MODEL_VERSION_COL, value)

    def get_model_version_col(self) -> str:
        return self.get(self.MODEL_VERSION_COL)


class _OnlineLogisticRegressionParams(
    _OnlineLogisticRegressionModelParams,
    HasBatchStrategy,
    HasLabelCol,
    HasWeightCol,
    HasReg,
    HasElasticNet,
    HasGlobalBatchSize
):
    """
    Params for :class:`OnlineLogisticRegression`.
    """

    ALPHA: Param[float] = FloatParam(
        "alpha",
        "The alpha parameter of ftrl.",
        0.1,
        ParamValidators.gt(0))

    BETA: Param[float] = FloatParam(
        "beta",
        "The beta parameter of ftrl.",
        0.1,
        ParamValidators.gt(0))

    def __init__(self, java_params):
        super(_OnlineLogisticRegressionParams, self).__init__(java_params)

    def set_alpha(self, alpha: float):
        return self.set(self.ALPHA, alpha)

    def get_alpha(self) -> float:
        return self.get(self.ALPHA)

    @property
    def alpha(self) -> float:
        return self.get_alpha()

    def set_beta(self, beta: float):
        return self.set(self.BETA, beta)

    def get_beta(self) -> float:
        return self.get(self.BETA)

    @property
    def beta(self) -> float:
        return self.get_beta()


class OnlineLogisticRegressionModel(JavaClassificationModel,
                                    _OnlineLogisticRegressionModelParams):
    """
    A Model which classifies data using the model data computed by
    :class:`OnlineLogisticRegression`.
    """

    def __init__(self, java_model=None):
        super(OnlineLogisticRegressionModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "logisticregression"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "OnlineLogisticRegressionModel"


class OnlineLogisticRegression(JavaClassificationEstimator, _OnlineLogisticRegressionParams):
    """
    An Estimator which implements the online logistic regression algorithm.

    See H. Brendan McMahan et al., Ad click prediction: a view from the trenches.
    """

    def __init__(self):
        super(OnlineLogisticRegression, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> OnlineLogisticRegressionModel:
        return OnlineLogisticRegressionModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "logisticregression"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "OnlineLogisticRegression"
