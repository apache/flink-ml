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

from pyflink.ml.core.wrapper import JavaWithParams
from pyflink.ml.lib.feature.common import JavaFeatureTransformer
from pyflink.ml.lib.param import HasInputCols, HasOutputCol, HasHandleInvalid


class _VectorAssemblerParams(
    JavaWithParams,
    HasInputCols,
    HasOutputCol,
    HasHandleInvalid
):
    """
    Params for :class:`VectorAssembler`.
    """

    def __init__(self, java_params):
        super(_VectorAssemblerParams, self).__init__(java_params)


class VectorAssembler(JavaFeatureTransformer, _VectorAssemblerParams):
    """
    A feature transformer that combines a given list of input columns into a vector column. Types of
    input columns must be either vector or numerical value.

    The `keep` option of :class:HasHandleInvalid means that we output bad rows with output column
    set to null.
    """

    def __init__(self, java_model=None):
        super(VectorAssembler, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "vectorassembler"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "VectorAssembler"
