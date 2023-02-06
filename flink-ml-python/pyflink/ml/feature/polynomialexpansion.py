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
from pyflink.ml.param import IntParam, ParamValidators
from pyflink.ml.feature.common import JavaFeatureTransformer
from pyflink.ml.common.param import HasInputCol, HasOutputCol, Param


class _PolynomialExpansionParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
    Params for :class:`PolynomialExpansion`.
    """

    DEGREE: Param[int] = IntParam(
        "degree",
        "Degree of the polynomial expansion.",
        2,
        ParamValidators.gt_eq(1))

    def __init__(self, java_params):
        super(_PolynomialExpansionParams, self).__init__(java_params)

    def set_degree(self, value: int):
        return typing.cast(_PolynomialExpansionParams, self.set(self.DEGREE, value))

    def get_degree(self) -> bool:
        return self.get(self.DEGREE)

    @property
    def degree(self):
        return self.get_degree()


class PolynomialExpansion(JavaFeatureTransformer, _PolynomialExpansionParams):
    """
    A Transformer that expands the input vectors in polynomial space.

    Take a 2-dimension vector as an example: `(x, y)`, if we want to expand it with degree 2, then
    we get `(x, x * x, y, x * y, y * y)`.

    For more information about the polynomial expansion, see
    http://en.wikipedia.org/wiki/Polynomial_expansion.
    """

    def __init__(self, java_model=None):
        super(PolynomialExpansion, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "polynomialexpansion"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "PolynomialExpansion"
