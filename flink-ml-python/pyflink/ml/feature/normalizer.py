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

from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.param import FloatParam, ParamValidators
from pyflink.ml.feature.common import JavaFeatureTransformer
from pyflink.ml.common.param import HasInputCol, HasOutputCol, Param


class _NormalizerParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
    Params for :class:`Normalizer`.
    """

    P: Param[float] = FloatParam(
        "p",
        "The p norm value.",
        2.0,
        ParamValidators.gt_eq(1.0))

    def __init__(self, java_params):
        super(_NormalizerParams, self).__init__(java_params)

    def set_p(self, value: float):
        return typing.cast(_NormalizerParams, self.set(self.P, value))

    def get_p(self) -> float:
        return self.get(self.P)

    @property
    def p(self):
        return self.get_p()


class Normalizer(JavaFeatureTransformer, _NormalizerParams):
    """
    A Transformer that normalizes a vector to have unit norm using the given p-norm.
    """

    def __init__(self, java_model=None):
        super(Normalizer, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "normalizer"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "Normalizer"
