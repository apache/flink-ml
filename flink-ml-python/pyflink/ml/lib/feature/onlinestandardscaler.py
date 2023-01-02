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

from pyflink.ml.lib.feature.common import JavaFeatureEstimator, JavaFeatureModel
from pyflink.ml.lib.param import HasModelVersionCol, \
    HasMaxAllowedModelDelayMs, HasWindows
from pyflink.ml.lib.feature.standardscaler import _StandardScalerParams


class _OnlineStandardScalerModelParams(
    _StandardScalerParams,
    HasModelVersionCol,
    HasMaxAllowedModelDelayMs
):
    """
    Params for :class:`OnlineStandardScalerModel`.
    """

    def __init__(self, java_params):
        super(_OnlineStandardScalerModelParams, self).__init__(java_params)


class _OnlineStandardScalerParams(
    _OnlineStandardScalerModelParams,
    HasWindows
):
    """
    Params for :class:`OnlineStandardScaler`.
    """

    def __init__(self, java_params):
        super(_OnlineStandardScalerParams, self).__init__(java_params)


class OnlineStandardScalerModel(JavaFeatureModel, _OnlineStandardScalerModelParams):
    """
    A Model which transforms data using the model data computed by
    :class:`OnlineStandardScaler`.
    """

    def __init__(self, java_model=None):
        super(OnlineStandardScalerModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "standardscaler"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "OnlineStandardScalerModel"


class OnlineStandardScaler(JavaFeatureEstimator, _OnlineStandardScalerParams):
    """
    An Estimator which implements the online standard scaling algorithm, which is the
    online version of {@link StandardScaler}.

    OnlineStandardScaler splits the input data by the user-specified window strategy
    (i.e., {@link org.apache.flink.ml.common.param.HasWindows}). For each window, it
    computes the mean and standard deviation using the data seen so far (i.e., not only
    the data in the current window, but also the history data). The model data generated
    by OnlineStandardScaler is a model stream. There is one model data for each window.

    <p>During the inference phase (i.e., using {@link OnlineStandardScalerModel} for
    prediction), users could output the model version that is used for predicting each
    data point. Moreover,

    <ul>
        <li>When the train data and test data both contain event time, users could
        specify the maximum difference between the timestamps of the input and model data
        ({@link org.apache.flink.ml.common.param.HasMaxAllowedModelDelayMs}), which
        enforces to use a relatively fresh model for prediction.
        <li>Otherwise, the prediction process always uses the current model data for prediction.
    </ul>
    """

    def __init__(self):
        super(OnlineStandardScaler, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> OnlineStandardScalerModel:
        return OnlineStandardScalerModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "standardscaler"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "OnlineStandardScaler"
