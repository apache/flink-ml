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

from pyflink.ml.param import IntParam, ParamValidators
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureModel, JavaFeatureEstimator
from pyflink.ml.common.param import HasInputCol, HasOutputCol, HasHandleInvalid


class _VectorIndexerModelParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol,
    HasHandleInvalid
):
    """
    Params for :class:`VectorIndexerModel`.
    """

    def __init__(self, java_params):
        super(_VectorIndexerModelParams, self).__init__(java_params)


class _VectorIndexerParams(_VectorIndexerModelParams):
    """
    Params for :class:`VectorIndexer`.
    """

    MAX_CATEGORIES: IntParam = IntParam(
        "max_categories",
        "Threshold for the number of values a categorical feature can take (>= 2). "
        + "If a feature is found to have > maxCategories values, then it is declared continuous.",
        20,
        ParamValidators.gt_eq(2)
    )

    def __init__(self, java_params):
        super(_VectorIndexerParams, self).__init__(java_params)

    def set_max_categories(self, value: int):
        return typing.cast(_VectorIndexerParams, self.set(self.MAX_CATEGORIES, value))

    def get_max_categories(self) -> int:
        return self.get(self.MAX_CATEGORIES)

    @property
    def max_categories(self):
        return self.get_max_categories()


class VectorIndexerModel(JavaFeatureModel, _VectorIndexerModelParams):
    """
    A Model which encodes input vector to an output vector using the model data computed by
    :class::VectorIndexer.

    The `keep` option of {@link HasHandleInvalid} means that we put the invalid entries in a
    special bucket, whose index is the number of distinct values in this column.
    """

    def __init__(self, java_model=None):
        super(VectorIndexerModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "vectorindexer"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "VectorIndexerModel"


class VectorIndexer(JavaFeatureEstimator, _VectorIndexerParams):
    """
    An Estimator which implements the vector indexing algorithm.

    A vector indexer maps each column of the input vector into a continuous/categorical
    feature. Whether one feature is transformed into a continuous or categorical feature
    depends on the number of distinct values in this column. If the number of distinct
    values in one column is greater than a specified parameter (i.e., maxCategories),
    the corresponding output column is unchanged. Otherwise, it is transformed into
    a categorical value. For categorical outputs, the indices are
    in [0, numDistinctValuesInThisColumn].

    The output model is organized in ascending order except that 0.0 is always mapped
     to 0 (for sparsity). We list two examples here:

    <ul>
        <li>If one column contains {-1.0, 1.0}, then -1.0 should be encoded as 0
        and 1.0 will be encoded as 1.
        <li>If one column contains {-1.0, 0.0, 1.0}, then -1.0 should be encoded as 1,
         0.0 should be encoded as 0 and 1.0 should be encoded as 2.
    </ul>

    The `keep` option of {@link HasHandleInvalid} means that we put the invalid entries
    in a special bucket, whose index is the number of distinct values in this column.
    """

    def __init__(self):
        super(VectorIndexer, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> VectorIndexerModel:
        return VectorIndexerModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "vectorindexer"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "VectorIndexer"
