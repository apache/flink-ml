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

from pyflink.ml.param import IntParam, BooleanParam, StringParam, ParamValidators
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureTransformer
from pyflink.ml.common.param import HasInputCol, HasOutputCol


class _RegexTokenizerParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
    Params for :class:`RegexTokenizer`.
    """

    MIN_TOKEN_LENGTH: IntParam = IntParam(
        "min_token_length",
        "Minimum token length",
        1,
        ParamValidators.gt_eq(0)
    )

    GAPS: BooleanParam = BooleanParam(
        "gaps",
        "Set regex to match gaps or tokens",
        True
    )

    PATTERN: StringParam = StringParam(
        "pattern",
        "Regex pattern used for tokenizing",
        "\\s+"
    )

    TO_LOWERCASE: BooleanParam = BooleanParam(
        "to_lowercase",
        "Whether to convert all characters to lowercase before tokenizing",
        True
    )

    def __init__(self, java_params):
        super(_RegexTokenizerParams, self).__init__(java_params)

    def set_min_token_length(self, value: int):
        return typing.cast(_RegexTokenizerParams, self.set(self.MIN_TOKEN_LENGTH, value))

    def get_min_token_length(self) -> int:
        return self.get(self.MIN_TOKEN_LENGTH)

    def set_gaps(self, value: bool):
        return typing.cast(_RegexTokenizerParams, self.set(self.GAPS, value))

    def get_gaps(self) -> bool:
        return self.get(self.GAPS)

    def set_pattern(self, value: str):
        return typing.cast(_RegexTokenizerParams, self.set(self.PATTERN, value))

    def get_pattern(self) -> str:
        return self.get(self.PATTERN)

    def set_to_lowercase(self, value: bool):
        return typing.cast(_RegexTokenizerParams, self.set(self.TO_LOWERCASE, value))

    def get_to_lowertcase(self) -> bool:
        return self.get(self.TO_LOWERCASE)

    @property
    def min_token_length(self) -> int:
        return self.get_min_token_length()

    @property
    def gaps(self) -> bool:
        return self.get_gaps()

    @property
    def pattern(self) -> str:
        return self.get_pattern()

    @property
    def to_lowercase(self):
        return self.get_to_lowertcase()


class RegexTokenizer(JavaFeatureTransformer, _RegexTokenizerParams):
    """
    A Transformer which converts the input string to lowercase and then splits it by white spaces
    based on regex. It provides two options to extract tokens:

    <ul>
        <li>if "gaps" is true: uses the provided pattern to split the input string.
        <li>else: repeatedly matches the regex (the provided pattern) with the input string.
    </ul>

    Moreover, it provides parameters to filter tokens with a minimal length and converts input to
    lowercase. The output of each input string is an array of strings that can be empty.

    """

    def __init__(self, java_model=None):
        super(RegexTokenizer, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "regextokenizer"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "RegexTokenizer"
