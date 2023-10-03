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

from pyflink.ml.param import Param, StringParam, IntParam, FloatParam, ParamValidators
from pyflink.ml.recommendation.common import JavaRecommendationAlgoOperator
from pyflink.ml.wrapper import JavaWithParams


class _FPGrowthParams(
    JavaWithParams
):
    """
    Params for :class:`FPGrowth`.
    """

    ITEMS_COL: Param[str] = StringParam(
        "items_col",
        "Item sequence column name.",
        "items",
        ParamValidators.not_null())

    FIELD_DELIMITER: Param[str] = StringParam(
        "field_delimiter",
        "Field delimiter of item sequence, default delimiter is ','.",
        ",",
        ParamValidators.not_null())

    MIN_LIFT: Param[float] = FloatParam(
        "min_lift",
        "Minimal lift level for association rules.",
        1.0,
        ParamValidators.gt_eq(0))

    MIN_CONFIDENCE: Param[float] = FloatParam(
        "min_confidence",
        "Minimal confidence level for association rules.",
        0.6,
        ParamValidators.gt_eq(0))

    MIN_SUPPORT: Param[float] = FloatParam(
        "min_support",
        "Minimal support percent. The default value of MIN_SUPPORT is 0.02",
        0.02)

    MIN_SUPPORT_COUNT: Param[int] = IntParam(
        "min_support_count",
        "Minimal support count. MIN_ITEM_COUNT has no "
        + "effect when less than or equal to 0, The default value is -1.",
        -1)

    MAX_PATTERN_LENGTH: Param[int] = FloatParam(
        "max_pattern_length",
        "Max frequent pattern length.",
        10,
        ParamValidators.gt(0))

    def __init__(self, java_params):
        super(_FPGrowthParams, self).__init__(java_params)

    def set_items_col(self, value: str):
        return typing.cast(_FPGrowthParams, self.set(self.ITEMS_COL, value))

    def get_items_col(self) -> str:
        return self.get(self.ITEMS_COL)

    def set_field_delimiter(self, value: str):
        return typing.cast(_FPGrowthParams, self.set(self.FIELD_DELIMITER, value))

    def get_field_delimiter(self) -> str:
        return self.get(self.FIELD_DELIMITER)

    def set_min_lift(self, value: float):
        return typing.cast(_FPGrowthParams, self.set(self.MIN_LIFT, value))

    def get_min_lift(self) -> float:
        return self.get(self.MIN_LIFT)

    def set_min_confidence(self, value: float):
        return typing.cast(_FPGrowthParams, self.set(self.MIN_CONFIDENCE, value))

    def get_min_confidence(self) -> float:
        return self.get(self.MIN_CONFIDENCE)

    def set_min_support(self, value: float):
        return typing.cast(_FPGrowthParams, self.set(self.MIN_SUPPORT, value))

    def get_min_support(self) -> float:
        return self.get(self.MIN_SUPPORT)

    def set_min_support_count(self, value: int):
        return typing.cast(_FPGrowthParams, self.set(self.MIN_SUPPORT_COUNT, value))

    def get_min_support_count(self) -> int:
        return self.get(self.MIN_SUPPORT_COUNT)

    def set_max_pattern_length(self, value: int):
        return typing.cast(_FPGrowthParams, self.set(self.MAX_PATTERN_LENGTH, value))

    def get_max_pattern_length(self) -> int:
        return self.get(self.MAX_PATTERN_LENGTH)

    @property
    def items_col(self) -> str:
        return self.get_items_col()

    @property
    def field_delimiter(self) -> str:
        return self.get_field_delimiter()

    @property
    def min_lift(self) -> float:
        return self.get_min_lift()

    @property
    def min_confidence(self) -> float:
        return self.get_min_confidence()

    @property
    def min_support(self) -> float:
        return self.get_min_support()

    @property
    def min_support_count(self) -> int:
        return self.get_min_support_count()

    @property
    def max_pattern_length(self) -> int:
        return self.get_max_pattern_length()


class FPGrowth(JavaRecommendationAlgoOperator, _FPGrowthParams):
    """
    An implementation of parallel FP-growth algorithm to mine frequent itemset.

    <p>For detail descriptions, please refer to: <a
    href="http://dx.doi.org/10.1145/335191.335372">Han et al., Mining frequent patterns without
    candidate generation</a>. <a href="https://doi.org/10.1145/1454008.1454027">Li et al., PFP:
    Parallel FP-growth for query recommendation</a>
    """

    def __init__(self, java_algo_operator=None):
        super(FPGrowth, self).__init__(java_algo_operator)

    @classmethod
    def _java_algo_operator_package_name(cls) -> str:
        return "fpgrowth"

    @classmethod
    def _java_algo_operator_class_name(cls) -> str:
        return "FPGrowth"
