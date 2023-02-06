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

from pyflink.ml.param import Param, BooleanParam
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureEstimator, JavaFeatureModel
from pyflink.ml.common.param import (HasInputCols, HasOutputCols, HasHandleInvalid)


class _OneHotEncoderParams(
    JavaWithParams,
    HasInputCols,
    HasOutputCols,
    HasHandleInvalid
):
    """
    Params for :class:`OneHotEncoder`.
    """

    DROP_LAST: Param[bool] = BooleanParam(
        "drop_last",
        "Whether to drop the last category.",
        True)

    def __init__(self, java_params):
        super(_OneHotEncoderParams, self).__init__(java_params)

    def set_drop_last(self, value: bool):
        return typing.cast(_OneHotEncoderParams, self.set(self.DROP_LAST, value))

    def get_drop_last(self) -> bool:
        return self.get(self.DROP_LAST)

    @property
    def drop_last(self):
        return self.get_drop_last()


class OneHotEncoderModel(JavaFeatureModel, _OneHotEncoderParams):
    """
    A Model which encodes data into one-hot format using the model data computed by :class:
    OneHotEncoder}.

    The `keep` and `skip` option of {@link HasHandleInvalid} is not supported in
    :class:_OneHotEncoderParams.
    """

    def __init__(self, java_model=None):
        super(OneHotEncoderModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "onehotencoder"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "OneHotEncoderModel"


class OneHotEncoder(JavaFeatureEstimator, _OneHotEncoderParams):
    """
    An Estimator which implements the one-hot encoding algorithm.

    Data of selected input columns should be indexed numbers in order for OneHotEncoder to
    function correctly.

    The `keep` and `skip` option of :class:HasHandleInvalid is not supported in
    :class:OneHotEncoderParams.

    See https://en.wikipedia.org/wiki/One-hot.
    """

    def __init__(self):
        super(OneHotEncoder, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> OneHotEncoderModel:
        return OneHotEncoderModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "onehotencoder"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "OneHotEncoder"
