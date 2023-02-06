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
from pyflink.ml.common.param import HasInputCols, HasOutputCol, HasHandleInvalid, Param


class _VectorAssemblerParams(
    JavaWithParams,
    HasInputCols,
    HasOutputCol,
    HasHandleInvalid
):

    """
    Checks the inputSizes parameter.
    """
    def SizesValidator(self) -> ParamValidator[Tuple[int]]:
        class SizesValidator(ParamValidator[Tuple[int]]):
            def validate(self, indices: Tuple[int]) -> bool:
                if indices is None:
                    return False
                for val in indices:
                    if val <= 0:
                        return False
                return len(indices) != 0
        return SizesValidator()

    """
    Params for :class:`VectorAssembler`.
    """

    INPUT_SIZES: Param[Tuple[int, ...]] = IntArrayParam(
        "input_sizes",
        "Sizes of the input elements to be assembled.",
        None,
        SizesValidator(None))

    def __init__(self, java_params):
        super(_VectorAssemblerParams, self).__init__(java_params)

    def set_input_sizes(self, *sizes: int):
        return self.set(self.INPUT_SIZES, sizes)

    def get_input_sizes(self) -> Tuple[int, ...]:
        return self.get(self.INPUT_SIZES)

    @property
    def input_sizes(self) -> Tuple[int, ...]:
        return self.get_input_sizes()


class VectorAssembler(JavaFeatureTransformer, _VectorAssemblerParams):
    """
     A Transformer which combines a given list of input columns into a vector column. Input columns
     would be numerical or vectors whose sizes are specified by the :class:INPUT_SIZES parameter.
     Invalid input data with null values or values with wrong sizes would be dealt with according to
     the strategy specified by the :class:HasHandleInvalid parameter as follows:
     <ul>
       <li>keep: If the input column data is null, a vector would be created with the specified size
           and NaN values. The vector would be used in the assembling process to represent the input
           column data. If the input column data is a vector, the data would be used in the
           assembling process even if it has a wrong size.
       <li>skip: If the input column data is null or a vector with wrong size, the input row would
           be filtered out and not be sent to downstream operators.
       <li>error: If the input column data is null or a vector with wrong size, an exception would
           be thrown.
     </ul>
    """

    def __init__(self, java_model=None):
        super(VectorAssembler, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "vectorassembler"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "VectorAssembler"
