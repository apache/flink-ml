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

from pyflink.ml.param import Param, StringParam, ParamValidators
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureModel, JavaFeatureEstimator
from pyflink.ml.common.param import HasInputCols, HasOutputCols, HasHandleInvalid


class _IndexToStringModelParams(
    JavaWithParams,
    HasInputCols,
    HasOutputCols
):
    """
    Params for :class:`IndexToStringModel`.
    """

    def __init__(self, java_params):
        super(_IndexToStringModelParams, self).__init__(java_params)


class _StringIndexerModelParams(
    JavaWithParams,
    HasInputCols,
    HasOutputCols,
    HasHandleInvalid
):
    """
    Params for :class:`StringIndexerModel`.
    """

    def __init__(self, java_params):
        super(_StringIndexerModelParams, self).__init__(java_params)


class _StringIndexerParams(_StringIndexerModelParams):
    """
    Params for :class:`StringIndexer`.
    """

    STRING_ORDER_TYPE: Param[str] = StringParam(
        "string_order_type",
        "How to order strings of each column.",
        "arbitrary",
        ParamValidators.in_array(
            ['arbitrary', 'frequencyDesc', 'frequencyAsc', 'alphabetDesc', 'alphabetAsc']))

    def __init__(self, java_params):
        super(_StringIndexerParams, self).__init__(java_params)

    def set_string_order_type(self, value: str):
        return typing.cast(_StringIndexerParams, self.set(self.STRING_ORDER_TYPE, value))

    def get_string_order_type(self) -> str:
        return self.get(self.STRING_ORDER_TYPE)

    @property
    def string_order_type(self):
        return self.get_string_order_type()


class IndexToStringModel(JavaFeatureModel, _IndexToStringModelParams):
    """
    A Model which transforms input index column(s) to string column(s) using the model data computed
    by :class:StringIndexer. It is a reverse operation of :class:StringIndexerModel.
    """

    def __init__(self, java_model=None):
        super(IndexToStringModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "stringindexer"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "IndexToStringModel"


class StringIndexerModel(JavaFeatureModel, _StringIndexerModelParams):
    """
    A Model which transforms input string/numeric column(s) to integer column(s) using the model
    data computed by :class:StringIndexer.

    The `keep` option of {@link HasHandleInvalid} means that we put the invalid entries in a
    special bucket, whose index is the number of distinct values in this column.
    """

    def __init__(self, java_model=None):
        super(StringIndexerModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "stringindexer"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "StringIndexerModel"


class StringIndexer(JavaFeatureEstimator, _StringIndexerParams):
    """
    An Estimator which implements the string indexing algorithm.

    A string indexer maps one or more columns (string/numerical value) of the input to one or more
    indexed output columns (integer value). The output indices of two data points are the same iff
    their corresponding input columns are the same. The indices are in [0,
    numDistinctValuesInThisColumn].

    The input columns are cast to string if they are numeric values. By default, the output model
    is arbitrarily ordered. Users can control this by setting {@link
    StringIndexerParams#STRING_ORDER_TYPE}.

    The `keep` option of {@link HasHandleInvalid} means that we put the invalid entries in a
    special bucket, whose index is the number of distinct values in this column.
    """

    def __init__(self):
        super(StringIndexer, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> StringIndexerModel:
        return StringIndexerModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "stringindexer"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "StringIndexer"
