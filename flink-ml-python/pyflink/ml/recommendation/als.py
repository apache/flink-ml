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
from pyflink.ml.param import Param, StringParam, IntParam, FloatParam, BooleanParam, ParamValidators
from pyflink.ml.feature.common import JavaFeatureModel, JavaFeatureEstimator
from pyflink.ml.common.param import HasPredictionCol


class _AlsParams(
    JavaWithParams,
    HasPredictionCol
):
    """
    Params for :class:`Als`.
    """
    USER_COL: Param[str] = StringParam(
        "user_col",
        "User column name.",
        "user",
        ParamValidators.not_null())

    ITEM_COL: Param[str] = StringParam(
        "item_col",
        "Item column name.",
        "item",
        ParamValidators.not_null())

    RATING_COL: Param[str] = StringParam(
        "rating_col",
        "Rating column name.",
        "rating",
        ParamValidators.not_null())

    ALPHA: Param[float] = FloatParam(
        "alpha",
        "Alpha for implicit preference.",
        1.0,
        ParamValidators.gt_eq(0))

    REG_PARAM: Param[float] = FloatParam(
        "reg_param",
        "Regularization parameter.",
        0.1,
        ParamValidators.gt_eq(0))

    IMPLICIT_PREFS: Param[bool] = BooleanParam(
        "implicit_refs",
        "Whether to use implicit preference.",
        False)

    NON_NEGATIVE: Param[bool] = BooleanParam(
        "non_negative",
        "Whether to use non negative constraint for least squares.",
        False)

    RANK: Param[int] = IntParam(
        "rank",
        "Rank of the factorization.",
        10,
        ParamValidators.gt(0))

    MAX_ITER: Param[int] = IntParam(
        "max_iter",
        "Maximum number of iterations.",
        10,
        ParamValidators.gt(0))

    def __init__(self, java_params):
        super(_AlsParams, self).__init__(java_params)

    def set_user_col(self, value: str):
        return typing.cast(_AlsParams, self.set(self.USER_COL, value))

    def get_user_col(self) -> str:
        return self.get(self.USER_COL)

    def set_item_col(self, value: str):
        return typing.cast(_AlsParams, self.set(self.ITEM_COL, value))

    def get_item_col(self) -> str:
        return self.get(self.ITEM_COL)

    def set_rating_col(self, value: str):
        return typing.cast(_AlsParams, self.set(self.RATING_COL, value))

    def get_rating_col(self) -> str:
        return self.get(self.RATING_COL)

    def set_alpha(self, value: float):
        return typing.cast(_AlsParams, self.set(self.ALPHA, value))

    def get_alpha(self) -> float:
        return self.get(self.ALPHA)

    def set_reg_param(self, value: float):
        return typing.cast(_AlsParams, self.set(self.REG_PARAM, value))

    def get_reg_param(self) -> float:
        return self.get(self.REG_PARAM)

    def set_implicit_refs(self, value: bool):
        return typing.cast(_AlsParams, self.set(self.IMPLICIT_PREFS, value))

    def get_implicit_refs(self) -> bool:
        return self.get(self.NON_NEGATIVE)

    def set_non_negative(self, value: bool):
        return typing.cast(_AlsParams, self.set(self.IMPLICIT_PREFS, value))

    def get_non_negative(self) -> bool:
        return self.get(self.NON_NEGATIVE)

    def set_rank(self, value: int):
        return typing.cast(_AlsParams, self.set(self.RANK, value))

    def get_rank(self) -> int:
        return self.get(self.RANK)

    def set_max_iter(self, value: int):
        return typing.cast(_AlsParams, self.set(self.MAX_ITER, value))

    def get_max_iter(self) -> int:
        return self.get(self.MAX_ITER)


class AlsModel(JavaFeatureModel, _AlsParams):
    """
     A Model which transforms data using the model data computed by :class:`Als`.
    """

    def __init__(self, java_model=None):
        super(AlsModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "als"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "AlsModel"


class Als(JavaFeatureEstimator, _AlsParams):
    """
    An Estimator which implements the Als algorithm. ALS tries to decompose a matrix
    R as R = X * Yt. Here X and Y are called factor matrices. Matrix R is usually a
    sparse matrix representing ratings given from users to items. ALS tries to
    find X and Y that minimize || R - X * Yt ||^2. This is done by iterations. At each
    step, X is fixed and Y is solved, then Y is fixed and X is solved.
    """

    def __init__(self):
        super(Als, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> AlsModel:
        return AlsModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "als"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "Als"
