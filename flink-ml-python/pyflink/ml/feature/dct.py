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

from pyflink.ml.param import Param, BooleanParam
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureTransformer
from pyflink.ml.common.param import HasInputCol, HasOutputCol


class _DCTParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
    Params for :class:`DCT`.
    """

    INVERSE: Param[bool] = BooleanParam(
        "inverse",
        "Whether to perform the inverse DCT (true) or forward DCT (false).",
        False)

    def __init__(self, java_params):
        super(_DCTParams, self).__init__(java_params)

    def set_inverse(self, value: bool):
        return typing.cast(_DCTParams, self.set(self.INVERSE, value))

    def get_inverse(self) -> bool:
        return self.get(self.INVERSE)

    @property
    def inverse(self):
        return self.get_inverse()


class DCT(JavaFeatureTransformer, _DCTParams):
    """
    A Transformer that takes the 1D discrete cosine transform of a real vector.
    No zero padding is performed on the input vector. It returns a real vector
    of the same length representing the DCT. The return vector is scaled such
    that the transform matrix is unitary (aka scaled DCT-II).

    See https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II (DCT-II
    in Discrete cosine transform).
    """

    def __init__(self, java_model=None):
        super(DCT, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "dct"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "DCT"
