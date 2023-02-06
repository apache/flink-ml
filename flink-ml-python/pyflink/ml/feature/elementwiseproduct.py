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

from pyflink.ml.param import ParamValidators, Param, VectorParam
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureTransformer
from pyflink.ml.common.param import HasInputCol, HasOutputCol
from pyflink.ml.linalg import Vector


class _ElementwiseProductParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
    Params for :class:`ElementwiseProduct`.
    """

    SCALING_VEC: Param[Vector] = VectorParam(
        "scaling_vec",
        "the scaling vector to multiply with input vectors using hadamard product.",
        None,
        ParamValidators.not_null())

    def __init__(self, java_params):
        super(_ElementwiseProductParams, self).__init__(java_params)

    def set_scaling_vec(self, value: Vector):
        return self.set(self.SCALING_VEC, value)

    def get_scaling_vec(self) -> Vector:
        return self.get(self.SCALING_VEC)

    @property
    def scaling_vec(self) -> Vector:
        return self.get_scaling_vec()


class ElementwiseProduct(JavaFeatureTransformer, _ElementwiseProductParams):
    """
    A Transformer that multiplies each input vector with a given scaling vector using Hadamard
    product.

    If the size of the input vector does not equal the size of the scaling vector,
    the transformer will throw IllegalArgumentException.
    """

    def __init__(self, java_model=None):
        super(ElementwiseProduct, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "elementwiseproduct"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "ElementwiseProduct"
