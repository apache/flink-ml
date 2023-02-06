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
from pyflink.ml.param import Param, FloatArrayParam, ParamValidator
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureTransformer
from pyflink.ml.common.param import HasSeed


class _RandomSplitterParams(
    JavaWithParams,
    HasSeed
):
    """
    Checks the weights parameter.
    """
    def weights_validator(self) -> ParamValidator[Tuple[float]]:
        class WeightsValidator(ParamValidator[Tuple[float]]):
            def validate(self, weights: Tuple[float]) -> bool:
                for val in weights:
                    if val <= 0.0:
                        return False
                return len(weights) > 1
        return WeightsValidator()

    """
    Params for :class:`RandomSplitter`.
    Weights should be a non-empty array with all elements greater than zero.
    The weights will be normalized such that the sum of all elements equals
    to one.
    """
    WEIGHTS: Param[Tuple[float]] = FloatArrayParam(
        "weights",
        "The weights of data splitting.",
        [1.0, 1.0],
        weights_validator(None))

    def __init__(self, java_params):
        super(_RandomSplitterParams, self).__init__(java_params)

    def set_weights(self, *value: float):
        return typing.cast(_RandomSplitterParams, self.set(self.WEIGHTS, value))

    def get_weights(self) -> Tuple[float, ...]:
        return self.get(self.WEIGHTS)

    @property
    def weights(self):
        return self.get_weights()


class RandomSplitter(JavaFeatureTransformer, _RandomSplitterParams):
    """
    An AlgoOperator which splits a table into N tables according to the given weights.
    """

    def __init__(self, java_model=None):
        super(RandomSplitter, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "randomsplitter"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "RandomSplitter"
