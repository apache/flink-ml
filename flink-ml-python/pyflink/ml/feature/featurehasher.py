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

from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureTransformer
from pyflink.ml.common.param import HasInputCols, HasOutputCol, HasCategoricalCols, HasNumFeatures


class _FeatureHasherParams(
    JavaWithParams,
    HasInputCols,
    HasCategoricalCols,
    HasOutputCol,
    HasNumFeatures
):
    """
    Params for :class:`FeatureHasher`.
    """

    def __init__(self, java_params):
        super(_FeatureHasherParams, self).__init__(java_params)


class FeatureHasher(JavaFeatureTransformer, _FeatureHasherParams):
    """
    A Transformer that transforms a set of categorical or numerical features into
    a sparse vector of a specified dimension. The rules of hashing categorical
    columns and numerical columns are as follows:

    For numerical columns, the index of this feature in the output vector is the
    hash value of the column name and its correponding value is the same as the
    input.

    For categorical columns, the index of this feature in the output vector is
    the hash value of the string "column_name=value" and the corresponding
    value is 1.0.

    If multiple features are projected into the same column, the output values
    are accumulated. For the hashing trick, see
    https://en.wikipedia.org/wiki/Feature_hashing for details.
    """

    def __init__(self, java_model=None):
        super(FeatureHasher, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "featurehasher"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "FeatureHasher"
