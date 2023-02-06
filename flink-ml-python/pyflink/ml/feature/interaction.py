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

from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureTransformer
from pyflink.ml.common.param import HasInputCols, HasOutputCol


class _InteractionParams(
    JavaWithParams,
    HasInputCols,
    HasOutputCol
):
    """
    Params for :class:`Interaction`.
    """

    def __init__(self, java_params):
        super(_InteractionParams, self).__init__(java_params)


class Interaction(JavaFeatureTransformer, _InteractionParams):
    """
    A Transformer that takes vector or numerical columns, and generates a single vector column
    that contains the product of all combinations of one value from each input column.

    For example, when the input feature values are Double(2) and Vector(3, 4), the output would be
    Vector(6, 8). When the input feature values are Vector(1, 2) and Vector(3, 4), the output would
    be Vector(3, 4, 6, 8). If you change the position of these two input Vectors, the output would
    be Vector(3, 6, 4, 8).
    """

    def __init__(self, java_model=None):
        super(Interaction, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "interaction"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "Interaction"
