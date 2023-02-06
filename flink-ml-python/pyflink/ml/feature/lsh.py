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
#  limitations under the License.
################################################################################

import typing
from abc import ABC
from pyflink.java_gateway import get_gateway
from pyflink.table import Table
from pyflink.util.java_utils import to_jarray

from pyflink.ml.linalg import Vector, DenseVector, SparseVector
from pyflink.ml.param import Param, IntParam, ParamValidators
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureEstimator, JavaFeatureModel
from pyflink.ml.common.param import HasInputCol, HasOutputCol, HasSeed


class _LSHModelParams(JavaWithParams,
                      HasInputCol,
                      HasOutputCol):
    """
    Params for :class:`LSHModel`
    """

    def __init__(self, java_params):
        super(_LSHModelParams, self).__init__(java_params)


class _LSHParams(_LSHModelParams):
    """
    Params for :class:`LSH`
    """

    """
    Param for the number of hash tables used in LSH OR-amplification.

    OR-amplification can be used to reduce the false negative rate. Higher values of this
    param lead to a reduced false negative rate, at the expense of added computational
    complexity.
    """
    NUM_HASH_TABLES: Param[int] = IntParam(
        "num_hash_tables", "Number of hash tables.", 1, ParamValidators.gt_eq(1)
    )

    """
    Param for the number of hash functions per hash table used in LSH AND-amplification.

    AND-amplification can be used to reduce the false positive rate. Higher values of this
    param lead to a reduced false positive rate, at the expense of added computational
    complexity.
    """
    NUM_HASH_FUNCTIONS_PER_TABLE: Param[int] = IntParam(
        "num_hash_functions_per_table",
        "Number of hash functions per table.",
        1,
        ParamValidators.gt_eq(1))

    def __init__(self, java_params):
        super(_LSHParams, self).__init__(java_params)

    def set_num_hash_tables(self, value: int):
        return typing.cast(_LSHParams, self.set(self.NUM_HASH_TABLES, value))

    def get_num_hash_tables(self):
        return self.get(self.NUM_HASH_TABLES)

    @property
    def num_hash_tables(self):
        return self.get_num_hash_tables()

    def set_num_hash_functions_per_table(self, value: int):
        return typing.cast(_LSHParams, self.set(self.NUM_HASH_FUNCTIONS_PER_TABLE, value))

    def get_num_hash_functions_per_table(self):
        return self.get(self.NUM_HASH_FUNCTIONS_PER_TABLE)

    @property
    def num_hash_functions_per_table(self):
        return self.get_num_hash_functions_per_table()


class _MinHashLSHParams(_LSHParams, HasSeed):
    """
    Params for :class:`MinHashLSH`
    """

    def __init__(self, java_params):
        super(_MinHashLSHParams, self).__init__(java_params)


class _LSH(JavaFeatureEstimator, ABC):
    """
    Base class for estimators that support LSH (Locality-sensitive hashing) algorithm for different
    metrics (e.g., Jaccard distance).

    The basic idea of LSH is to use a set of hash functions to map input vectors into different
    buckets, where closer vectors are expected to be in the same bucket with higher probabilities.
    In detail, each input vector is hashed by all functions. To decide whether two input vectors
    are mapped into the same bucket, two mechanisms for assigning buckets are proposed as follows.

    <ul>
        <li>AND-amplification: The two input vectors are defined to be in the same bucket as long as
        ALL of the hash value matches.
        <li>OR-amplification: The two input vectors are defined to be in the same bucket as long as
        ANY of the hash value matches.
    </ul>

    See: <a href="https://en.wikipedia.org/wiki/Locality-sensitive_hashing">
    Locality-sensitive_hashing</a>.
    """

    def __init__(self):
        super(_LSH, self).__init__()

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "lsh"


class _LSHModel(JavaFeatureModel, ABC):
    """
    Base class for LSH model.

    In addition to transforming input feature vectors to multiple hash values, it also supports
    approximate nearest neighbors search within a dataset regarding a key vector and approximate
    similarity join between two datasets.
    """

    def __init__(self, java_model):
        super(_LSHModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "lsh"

    def approx_nearest_neighbors(self, dataset: Table, key: Vector, k: int,
                                 dist_col: str = 'distCol'):
        """
        Approximately finds at most k items from a dataset which have the closest distance to a
        given item. If the `outputCol` is missing in the given dataset, this method transforms the
        dataset with the model at first.

        :param dataset: The dataset in which to to search for nearest neighbors.
        :param key: The item to search for.
        :param k: The maximum number of nearest neighbors.
        :param dist_col: The output column storing the distance between each neighbor and the key.
        :return: A dataset containing at most k items closest to the key with a column named
                 `distCol` appended.
        """
        j_vectors = get_gateway().jvm.org.apache.flink.ml.linalg.Vectors
        if isinstance(key, (DenseVector,)):
            j_key = j_vectors.dense(to_jarray(get_gateway().jvm.double, key.values.tolist()))
        elif isinstance(key, (SparseVector,)):
            # noinspection PyProtectedMember
            j_key = j_vectors.sparse(
                key.size(),
                to_jarray(get_gateway().jvm.int, key._indices.tolist()),
                to_jarray(get_gateway().jvm.double, key._values.tolist())
            )
        else:
            raise TypeError(f'Key {key} must be an instance of Vector.')

        # noinspection PyProtectedMember
        return Table(self._java_obj.approxNearestNeighbors(
            dataset._j_table, j_key, k, dist_col), self._t_env)

    def approx_similarity_join(self, dataset_a: Table, dataset_b: Table, threshold: float,
                               id_col: str, dist_col: str = 'distCol'):
        """
        Joins two datasets to approximately find all pairs of rows whose distance are smaller than
        or equal to the threshold. If the `outputCol` is missing in either dataset, this method
        transforms the dataset at first.

        :param dataset_a: One dataset.
        :param dataset_b: The other dataset.
        :param threshold: The distance threshold.
        :param id_col: A column in the two datasets to identify each row.
        :param dist_col: The output column storing the distance between each pair of rows.
        :return: A joined dataset containing pairs of rows. The original rows are in columns
                 "dataset_a" and "dataset_b", and a column "distCol" is added to show the distance
                 between each pair.
        """
        # noinspection PyProtectedMember
        return Table(self._java_obj.approxSimilarityJoin(
            dataset_a._j_table, dataset_b._j_table,
            threshold, id_col, dist_col), self._t_env)


class MinHashLSHModel(_LSHModel, _LSHModelParams):
    """
    A Model which generates hash values using the model data computed by :class:`MinHashLSH`.
    """

    def __init__(self, java_model=None):
        super(MinHashLSHModel, self).__init__(java_model)

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "MinHashLSHModel"


class MinHashLSH(_LSH, _MinHashLSHParams):
    """
    An Estimator that implements the MinHash LSH algorithm, which supports LSH for Jaccard distance.

    The input could be dense or sparse vectors. Each input vector must have at least one non-zero
    index and all non-zero values are treated as binary "1" values. The sizes of input vectors
    should be same and not larger than a predefined prime (i.e., 2038074743).

    See: <a href="https://en.wikipedia.org/wiki/MinHash">MinHash</a>.
    """

    def __init__(self):
        super(MinHashLSH, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> MinHashLSHModel:
        return MinHashLSHModel(java_model)

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "MinHashLSH"
