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
from typing import Tuple

from pyflink.ml.param import ParamValidators, Param, FloatArrayParam
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureTransformer
from pyflink.ml.common.param import HasInputCols, HasOutputCols


class _BinarizerParams(
    JavaWithParams,
    HasInputCols,
    HasOutputCols
):
    """
    Params for :class:`Binarizer`.
    """

    THRESHOLDS: Param[Tuple[float, ...]] = FloatArrayParam(
        "thresholds",
        "The thresholds used to binarize continuous features. Each threshold would be used "
        + "against one input column. If the value of a continuous feature is greater than the "
        + "threshold, it will be binarized to 1.0. If the value is equal to or less than the "
        + "threshold, it will be binarized to 0.0.",
        None,
        ParamValidators.non_empty_array())

    def set_thresholds(self, *thresholds: float):
        return self.set(self.THRESHOLDS, thresholds)

    def get_thresholds(self) -> Tuple[float, ...]:
        return self.get(self.THRESHOLDS)

    @property
    def thresholds(self) -> Tuple[float, ...]:
        return self.get_thresholds()


class Binarizer(JavaFeatureTransformer, _BinarizerParams):
    """
    A Transformer that binarizes the columns of continuous features by the given thresholds.
    The continuous features may be DenseVector, SparseVector, or Numerical Value.
    """

    def __init__(self, java_model=None):
        super(Binarizer, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "binarizer"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "Binarizer"
