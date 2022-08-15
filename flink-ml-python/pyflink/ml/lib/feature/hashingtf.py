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

from pyflink.ml.core.param import BooleanParam
from pyflink.ml.core.wrapper import JavaWithParams
from pyflink.ml.lib.feature.common import JavaFeatureTransformer
from pyflink.ml.lib.param import HasInputCol, HasOutputCol, HasNumFeatures


class _HashingTFParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol,
    HasNumFeatures
):
    """
    Params for :class:`HashingTF`.
    """

    """
    Supported options to decide whether each dimension of the output vector is binary or not.
    <ul>
        <li>true: the value at one dimension is set as 1 if there are some features hashed to this
        column.
        <li>false: the value at one dimension is set as number of features that has been hashed to
        this column.
    </ul>
    """
    BINARY: BooleanParam = BooleanParam(
        "binary",
        "Whether each dimension of the output vector is binary or not.",
        False
    )

    def __init__(self, java_params):
        super(_HashingTFParams, self).__init__(java_params)

    def set_binary(self, value: bool):
        return typing.cast(_HashingTFParams, self.set(self.BINARY, value))

    def get_binary(self) -> bool:
        return self.get(self.BINARY)

    @property
    def binary(self) -> int:
        return self.get_binary()


class HashingTF(JavaFeatureTransformer, _HashingTFParams):
    """
    A Transformer that maps a sequence of terms(strings, numbers, booleans) to a sparse vector
    with a specified dimension using the hashing trick.

    <p>If multiple features are projected into the same column, the output values are accumulated
    by default. Users could also enforce all non-zero output values as 1 by setting {@link
    HashingTFParams#BINARY} as true.

    <p>For the hashing trick, see https://en.wikipedia.org/wiki/Feature_hashing for details.
    """

    def __init__(self, java_model=None):
        super(HashingTF, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "hashingtf"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "HashingTF"
