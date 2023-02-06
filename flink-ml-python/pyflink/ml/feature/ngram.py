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
from pyflink.ml.feature.common import JavaFeatureTransformer
from pyflink.ml.common.param import HasInputCol, HasOutputCol


class _NGramParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
    Params for :class:`NGram`.
    """

    N: IntParam = IntParam(
        "n",
        "Number of elements per n-gram (>=1).",
        2,
        ParamValidators.gt_eq(1)
    )

    def __init__(self, java_params):
        super(_NGramParams, self).__init__(java_params)

    def set_n(self, value: int):
        return typing.cast(_NGramParams, self.set(self.N, value))

    def get_n(self) -> int:
        return self.get(self.N)

    @property
    def n(self) -> int:
        return self.get_n()


class NGram(JavaFeatureTransformer, _NGramParams):
    """
    A Transformer that converts the input string array into an array of n-grams,
     where each n-gram is represented by a space-separated string of words. If
    the length of the input array is less than `n`, no n-grams are returned.

    See https://en.wikipedia.org/wiki/N-gram.
    """

    def __init__(self, java_model=None):
        super(NGram, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "ngram"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "NGram"
