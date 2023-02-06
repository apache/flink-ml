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

from pyflink.ml.param import FloatParam, StringParam, ParamValidators
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureModel, JavaFeatureEstimator
from pyflink.ml.common.param import HasInputCols, HasOutputCols, HasRelativeError


class _ImputerModelParams(
    JavaWithParams,
    HasInputCols,
    HasOutputCols,
    HasRelativeError
):
    """
    Params for :class:`ImputerModel`.
    """
    MISSING_VALUE: FloatParam = FloatParam(
        "missing_value",
        "The placeholder for the missing values. All occurrences of missing value will be imputed.",
        float("NaN")
    )

    def __init__(self, java_params):
        super(_ImputerModelParams, self).__init__(java_params)

    def set_missing_value(self, value: float):
        return typing.cast(_ImputerModelParams, self.set(self.MISSING_VALUE, value))

    def get_missing_value(self):
        return self.get(self.MISSING_VALUE)

    @property
    def missing_value(self):
        return self.get_missing_value()


class _ImputerParams(_ImputerModelParams):
    """
    Params for :class:`Imputer`.
    """

    """
    Supported options of the imputation strategy.
    <ul>
        <li>mean: replace missing values using the mean along each column.
        <li>median: replace missing values using the median along each column.
        <li>most_frequent: replace missing using the most frequent value along each column.
            If there is more than one such value, only the smallest is returned.
    </ul>
    """
    STRATEGY: StringParam = StringParam(
        "strategy",
        "The imputation strategy.",
        'mean',
        ParamValidators.in_array(['mean', 'median', 'most_frequent']))

    def __init__(self, java_params):
        super(_ImputerParams, self).__init__(java_params)

    def set_strategy(self, value: str):
        return typing.cast(_ImputerParams, self.set(self.STRATEGY, value))

    def get_strategy(self) -> str:
        return self.get(self.STRATEGY)

    @property
    def strategy(self):
        return self.get_strategy()


class ImputerModel(JavaFeatureModel, _ImputerModelParams):
    """
    A Model which replaces the missing values using the model data computed by Imputer.
    """

    def __init__(self, java_model=None):
        super(ImputerModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "imputer"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "ImputerModel"


class Imputer(JavaFeatureEstimator, _ImputerParams):
    """
    The imputer for completing missing values of the input columns.
    Missing values can be imputed using the statistics (mean, median or most frequent) of each
    column in which the missing values are located. The input columns should be of numeric type.

    Note that the mean/median/most_frequent value is computed after filtering out missing values.
    All null values in the input columns are treated as missing, and so are also imputed.
    """

    def __init__(self):
        super(Imputer, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> ImputerModel:
        return ImputerModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "imputer"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "Imputer"
