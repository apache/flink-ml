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

from pyflink.ml.param import Param, StringParam, IntParam, FloatParam, \
    BooleanParam, ParamValidators
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.clustering.common import JavaClusteringAlgoOperator
from pyflink.ml.common.param import HasDistanceMeasure, HasFeaturesCol, HasPredictionCol, HasWindows


class _AgglomerativeClusteringParams(
    JavaWithParams,
    HasDistanceMeasure,
    HasFeaturesCol,
    HasPredictionCol,
    HasWindows
):
    """
    Params for :class:`AgglomerativeClustering`.
    """
    NUM_CLUSTERS: Param[int] = IntParam("num_clusters",
                                        "The max number of clusters to create.",
                                        2)

    DISTANCE_THRESHOLD: Param[float] = \
        FloatParam("distance_threshold",
                   "Threshold to decide whether two clusters should be merged.",
                   None)

    """
    Supported options to compute the distance between two clusters. The
    algorithm will merge the pairs of cluster that minimize this criterion.
    <ul>
        <li>ward: the variance between the two clusters.
        <li>complete: the maximum distance between all observations of the two clusters.
        <li>single: the minimum distance between all observations of the two clusters.
        <li>average: the average distance between all observations of the two clusters.
    </ul>
    """
    LINKAGE: Param[str] = StringParam(
        "linkage",
        "Criterion for computing distance between two clusters.",
        "ward",
        ParamValidators.in_array(
            ["ward", "complete", "single", "average"]))

    COMPUTE_FULL_TREE: Param[bool] = BooleanParam(
        "compute_full_tree",
        "Whether computes the full tree after convergence.",
        False,
        ParamValidators.not_null())

    def __init__(self, java_params):
        super(_AgglomerativeClusteringParams, self).__init__(java_params)

    def set_num_clusters(self, value: int):
        return typing.cast(_AgglomerativeClusteringParams, self.set(self.NUM_CLUSTERS, value))

    def get_num_clusters(self) -> int:
        return self.get(self.NUM_CLUSTERS)

    def set_distance_threshold(self, value: float):
        return typing.cast(_AgglomerativeClusteringParams, self.set(self.DISTANCE_THRESHOLD, value))

    def get_distance_threshold(self) -> float:
        return self.get(self.DISTANCE_THRESHOLD)

    def set_linkage(self, value: str):
        return typing.cast(_AgglomerativeClusteringParams, self.set(self.LINKAGE, value))

    def get_linkage(self) -> str:
        return self.get(self.LINKAGE)

    def set_compute_full_tree(self, value: bool):
        return typing.cast(_AgglomerativeClusteringParams, self.set(self.COMPUTE_FULL_TREE, value))

    def get_compute_full_tree(self) -> bool:
        return self.get(self.COMPUTE_FULL_TREE)

    @property
    def num_clusters(self):
        return self.get_num_clusters()

    @property
    def distance_threshold(self):
        return self.get_distance_threshold()

    @property
    def linkage(self):
        return self.get_linkage()

    @property
    def compute_full_tree(self):
        return self.get_compute_full_tree()


class AgglomerativeClustering(JavaClusteringAlgoOperator, _AgglomerativeClusteringParams):
    """
    An AlgoOperator that performs a hierarchical clustering using a bottom-up approach. Each
    observation starts in its own cluster and the clusters are merged together one by one. Users can
    choose different strategies to merge two clusters by setting
    AgglomerativeClusteringParams#LINKAGE and different distance measures by setting
    AgglomerativeClusteringParams#DISTANCE_MEASURE.

    The output contains two tables. The first one assigns one cluster Id for each data point.
    The second one contains the information of merging two clusters at each step. The data format
    of the merging information is (clusterId1, clusterId2, distance, sizeOfMergedCluster).

    This AlgoOperator splits input stream into mini-batches of elements according to the windowing
    strategy specified by the HasWindows parameter, and performs the hierarchical clustering on each
    mini-batch independently. The clustering result of each element depends only on the elements in
    the same mini-batch.

    See https://en.wikipedia.org/wiki/Hierarchical_clustering.
    """

    def __init__(self, java_algo_operator=None):
        super(AgglomerativeClustering, self).__init__(java_algo_operator)

    @classmethod
    def _java_algo_operator_package_name(cls) -> str:
        return "agglomerativeclustering"

    @classmethod
    def _java_algo_operator_class_name(cls) -> str:
        return "AgglomerativeClustering"
