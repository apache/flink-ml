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
from typing import Tuple

from pyflink.ml.param import Param, FloatArrayArrayParam
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureTransformer
from pyflink.ml.common.param import HasInputCols, HasOutputCols, HasHandleInvalid


class _BucketizerParams(
    JavaWithParams,
    HasInputCols,
    HasOutputCols,
    HasHandleInvalid
):
    """
    Params for :class:`Bucketizer`.
    """

    SPLITS_ARRAY: Param[Tuple[float, ...]] = FloatArrayArrayParam(
        "splits_array",
        "Array of split points for mapping continuous features into buckets.",
        None)

    def __init__(self, java_params):
        super(_BucketizerParams, self).__init__(java_params)

    def set_splits_array(self, value: Tuple[Tuple[float, ...]]):
        return typing.cast(_BucketizerParams, self.set(self.SPLITS_ARRAY, value))

    def get_split_array(self) -> Tuple[Tuple[float, ...]]:
        return self.get(self.SPLITS_ARRAY)

    @property
    def split_array(self):
        return self.get_split_array()


class Bucketizer(JavaFeatureTransformer, _BucketizerParams):
    """
    A Transformer that maps multiple columns of continuous features to multiple
    columns of discrete features, i.e., buckets indices. The indices are in
    [0, numSplitsInThisColumn - 1].

    The `keep` option of HasHandleInvalid means that we put the invalid data in the last
    bucket of the splits, whose index is the number of the buckets.
    """

    def __init__(self, java_model=None):
        super(Bucketizer, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "bucketizer"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "Bucketizer"
