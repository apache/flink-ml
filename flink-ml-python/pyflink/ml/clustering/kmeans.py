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

from pyflink.ml.param import ParamValidators, Param, IntParam, StringParam
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.clustering.common import JavaClusteringModel, JavaClusteringEstimator
from pyflink.ml.common.param import (HasDistanceMeasure, HasFeaturesCol, HasPredictionCol,
                                     HasBatchStrategy, HasGlobalBatchSize, HasDecayFactor, HasSeed,
                                     HasMaxIter)


class _KMeansModelParams(
    JavaWithParams,
    HasDistanceMeasure,
    HasFeaturesCol,
    HasPredictionCol,
    ABC
):
    """
    Params for :class:`KMeansModel`.
    """

    K: Param[int] = IntParam(
        "k",
        "The max number of clusters to create.",
        2,
        ParamValidators.gt(1))

    def __init__(self, java_params):
        super(_KMeansModelParams, self).__init__(java_params)

    def set_k(self, value: int):
        return typing.cast(_KMeansModelParams, self.set(self.K, value))

    def get_k(self) -> int:
        return self.get(self.K)

    @property
    def k(self) -> int:
        return self.get_k()


class _KMeansParams(
    _KMeansModelParams,
    HasSeed,
    HasMaxIter
):
    """
    Params for :class:`KMeans`.
    """
    INIT_MODE: Param[str] = StringParam(
        "init_mode",
        "The initialization algorithm. Supported options: 'random'.",
        "random",
        ParamValidators.in_array(["random"]))

    def __init__(self, java_params):
        super(_KMeansParams, self).__init__(java_params)

    def set_init_mode(self, value: str):
        return self.set(self.INIT_MODE, value)

    def get_init_mode(self) -> str:
        return self.get(self.INIT_MODE)

    @property
    def init_mode(self):
        return self.get_init_mode()


class _OnlineKMeansParams(
    _KMeansModelParams,
    HasBatchStrategy,
    HasGlobalBatchSize,
    HasDecayFactor,
    HasSeed,
):
    """
    Params of :class:OnlineKMeans.
    """

    def __init__(self, java_params):
        super(_OnlineKMeansParams, self).__init__(java_params)


class KMeansModel(JavaClusteringModel, _KMeansModelParams):
    """
    A Model which clusters data into k clusters using the model data computed by :class:`KMeans`.
    """

    def __init__(self, java_model=None):
        super(KMeansModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "kmeans"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "KMeansModel"


class OnlineKMeansModel(JavaClusteringModel, _KMeansModelParams):
    """
    OnlineKMeansModel can be regarded as an advanced :class:`KMeansModel` operator which can update
    model data in a streaming format, using the model data provided by :class:`OnlineKMeans`.
    """

    def __init__(self, java_model=None):
        super(OnlineKMeansModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "kmeans"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "OnlineKMeansModel"


class KMeans(JavaClusteringEstimator, _KMeansParams):
    """
    An Estimator which implements the k-means clustering algorithm.

    See https://en.wikipedia.org/wiki/K-means_clustering.
    """

    def __init__(self):
        super(KMeans, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> KMeansModel:
        return KMeansModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "kmeans"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "KMeans"


class OnlineKMeans(JavaClusteringEstimator, _OnlineKMeansParams):
    """
    OnlineKMeans extends the function of :class:`KMeans`, supporting to train a K-Means model
    continuously according to an unbounded stream of train data.

    OnlineKMeans makes updates with the "mini-batch" KMeans rule, generalized to incorporate
    forgetfulness (i.e. decay). After the centroids estimated on the current batch are acquired,
    OnlineKMeans computes the new centroids from the weighted average between the original and the
    estimated centroids. The weight of the estimated centroids is the number of points assigned to
    them. The weight of the original centroids is also the number of points, but additionally
    multiplying with the decay factor.

    The decay factor scales the contribution of the clusters as estimated thus far. If the decay
    factor is 1, all batches are weighted equally. If the decay factor is 0, new centroids are
    determined entirely by recent data. Lower values correspond to more forgetting.
    """

    def __init__(self):
        super(OnlineKMeans, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> KMeansModel:
        return KMeansModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "kmeans"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "OnlineKMeans"
