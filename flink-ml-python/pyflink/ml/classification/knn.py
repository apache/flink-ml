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

from pyflink.ml.param import Param, IntParam, ParamValidators
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.classification.common import (JavaClassificationModel,
                                              JavaClassificationEstimator)
from pyflink.ml.common.param import HasFeaturesCol, HasPredictionCol, HasLabelCol


class _KNNModelParams(
    JavaWithParams,
    HasFeaturesCol,
    HasPredictionCol,
    ABC
):
    """
    Params for :class:`KNNModel`.
    """

    K: Param[int] = IntParam(
        "k",
        "The number of nearest neighbors",
        5,
        ParamValidators.gt(0))

    def __init__(self, java_params):
        super(_KNNModelParams, self).__init__(java_params)

    def set_k(self, value: int):
        return typing.cast(_KNNModelParams, self.set(self.K, value))

    def get_k(self) -> int:
        return self.get(self.K)

    @property
    def k(self) -> int:
        return self.get_k()


class _KNNParams(
    _KNNModelParams,
    HasLabelCol
):
    """
    Params for :class:`KNN`.
    """

    def __init__(self, java_params):
        super(_KNNParams, self).__init__(java_params)


class KNNModel(JavaClassificationModel, _KNNModelParams):
    """
    A Model which classifies data using the model data computed by :class:`KNN`.
    """

    def __init__(self, java_model=None):
        super(KNNModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "knn"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "KnnModel"


class KNN(JavaClassificationEstimator, _KNNParams):
    """
    An Estimator which implements the KNN algorithm.

    See: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm.
    """

    def __init__(self):
        super(KNN, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> KNNModel:
        return KNNModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "knn"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "Knn"
