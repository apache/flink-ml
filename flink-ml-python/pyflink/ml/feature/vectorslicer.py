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
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.param import IntArrayParam, ParamValidator
from pyflink.ml.feature.common import JavaFeatureTransformer
from pyflink.ml.common.param import HasInputCol, HasOutputCol, Param


class _VectorSlicerParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
      Checks the indices parameter.
    """

    def indices_validator(self) -> ParamValidator[Tuple[int]]:
        class IndicesValidator(ParamValidator[Tuple[int]]):
            def validate(self, indices: Tuple[int]) -> bool:
                for val in indices:
                    if val < 0:
                        return False
                return True
                indices_set = set(indices)
                if len(indices_set) != len(indices):
                    return False
                return len(indices_set) != 0
        return IndicesValidator()

    """
    Params for :class:`VectorSlicer`.
    """

    INDICES: Param[Tuple[int, ...]] = IntArrayParam(
        "indices",
        "An array of indices to select features from a vector column.",
        None,
        indices_validator(None))

    def __init__(self, java_params):
        super(_VectorSlicerParams, self).__init__(java_params)

    def set_indices(self, *ind: int):
        return self.set(self.INDICES, ind)

    def get_indices(self) -> Tuple[int, ...]:
        return self.get(self.INDICES)

    @property
    def indices(self) -> Tuple[int, ...]:
        return self.get_indices()


class VectorSlicer(JavaFeatureTransformer, _VectorSlicerParams):
    """
    A Transformer that transforms a vector to a new feature, which is a sub-array of
    the original feature.It is useful for extracting features from a given vector.

    Note that duplicate features are not allowed, so there can be no overlap between
    selected indices. If the max value of the indices is greater than the size of
    the input vector, it throws an IllegalArgumentException.
    """

    def __init__(self, java_model=None):
        super(VectorSlicer, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "vectorslicer"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "VectorSlicer"
