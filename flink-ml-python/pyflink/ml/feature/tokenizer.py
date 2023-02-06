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
from pyflink.ml.common.param import HasInputCol, HasOutputCol


class _TokenizerParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
    Params for :class:`Tokenizer`.
    """

    def __init__(self, java_params):
        super(_TokenizerParams, self).__init__(java_params)


class Tokenizer(JavaFeatureTransformer, _TokenizerParams):
    """
    A Transformer that converts the input string to lowercase and then splits it by white spaces.
    """

    def __init__(self, java_model=None):
        super(Tokenizer, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "tokenizer"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "Tokenizer"
