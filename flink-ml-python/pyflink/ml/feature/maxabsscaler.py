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
from pyflink.ml.feature.common import JavaFeatureModel, JavaFeatureEstimator
from pyflink.ml.common.param import HasInputCol, HasOutputCol


class _MaxAbsScalerParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):

    def __init__(self, java_params):
        super(_MaxAbsScalerParams, self).__init__(java_params)


class MaxAbsScalerModel(JavaFeatureModel, _MaxAbsScalerParams):
    """
     A Model which transforms data using the model data computed by :class:`MaxAbsScaler`.
    """

    def __init__(self, java_model=None):
        super(MaxAbsScalerModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "maxabsscaler"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "MaxAbsScalerModel"


class MaxAbsScaler(JavaFeatureEstimator, _MaxAbsScalerParams):
    """
    An Estimator which implements the MaxAbsScaler algorithm. This algorithm rescales feature
    values to the range [-1, 1] by dividing through the largest maximum absolute value in each
    feature. It does not shift/center the data and thus does not destroy any sparsity.
    """

    def __init__(self):
        super(MaxAbsScaler, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> MaxAbsScalerModel:
        return MaxAbsScalerModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "maxabsscaler"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "MaxAbsScaler"
