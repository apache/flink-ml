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

from pyflink.ml.core.param import Param, StringParam, ParamValidators, IntParam, FloatParam
from pyflink.ml.core.wrapper import JavaWithParams
from pyflink.ml.lib.feature.common import JavaFeatureEstimator, JavaFeatureModel
from pyflink.ml.lib.param import HasOutputCol, HasFeaturesCol, HasLabelCol


class _ChiSqSelectorModelParams(
    JavaWithParams,
    HasFeaturesCol,
    HasOutputCol
):
    """
    Params for :class:`ChiSqSelectorModel`.
    """

    def __init__(self, java_params):
        super(_ChiSqSelectorModelParams, self).__init__(java_params)


class _ChiSqSelectorParams(
    _ChiSqSelectorModelParams,
    HasLabelCol
):
    """
    Params for :class:`ChiSqSelector`.
    """

    SELECTOR_TYPE: Param[str] = StringParam(
        "selector_type",
        "The selector type. Supported options: numTopFeatures, percentile, fpr, fdr, fwe.",
        "numTopFeatures",
        ParamValidators.in_array(['numTopFeatures', 'percentile', 'fpr', 'fdr', 'fwe']))

    NUM_TOP_FEATURES: Param[int] = IntParam(
        "num_top_features",
        "Number of features that selector will select, ordered by ascending p-value. If the"
        " number of features is < numTopFeatures, then this will select all features.",
        50,
        ParamValidators.gt_eq(1))

    PERCENTILE: Param[float] = FloatParam(
        "percentile",
        "Percentile of features that selector will select, ordered by ascending p-value.",
        0.1,
        ParamValidators.in_range(0, 1))

    FPR: Param[float] = FloatParam(
        "fpr",
        "The highest p-value for features to be kept.",
        0.05,
        ParamValidators.in_range(0, 1))

    FDR: Param[float] = FloatParam(
        "fdr",
        "The upper bound of the expected false discovery rate.",
        0.05,
        ParamValidators.in_range(0, 1))

    FWE: Param[float] = FloatParam(
        "fwe",
        "The upper bound of the expected family-wise error rate.",
        0.05,
        ParamValidators.in_range(0, 1))

    def __init__(self, java_params):
        super(_ChiSqSelectorParams, self).__init__(java_params)

    def set_selector_type(self, value: str):
        return typing.cast(_ChiSqSelectorParams, self.set(self.SELECTOR_TYPE, value))

    def set_num_top_features(self, value: int):
        return typing.cast(_ChiSqSelectorParams, self.set(self.NUM_TOP_FEATURES, value))

    def set_percentile(self, value: float):
        return typing.cast(_ChiSqSelectorParams, self.set(self.PERCENTILE, value))

    def set_fpr(self, value: float):
        return typing.cast(_ChiSqSelectorParams, self.set(self.FPR, value))

    def set_fdr(self, value: float):
        return typing.cast(_ChiSqSelectorParams, self.set(self.FDR, value))

    def set_fwe(self, value: float):
        return typing.cast(_ChiSqSelectorParams, self.set(self.FWE, value))

    def get_selector_type(self) -> bool:
        return self.get(self.SELECTOR_TYPE)

    def get_num_top_features(self) -> bool:
        return self.get(self.NUM_TOP_FEATURES)

    def get_percentile(self) -> bool:
        return self.get(self.PERCENTILE)

    def get_fpr(self) -> bool:
        return self.get(self.FPR)

    def get_fdr(self) -> bool:
        return self.get(self.FDR)

    def get_fwe(self) -> bool:
        return self.get(self.FWE)

    @property
    def selector_type(self):
        return self.get_selector_type()

    @property
    def num_top_features(self):
        return self.get_num_top_features()

    @property
    def percentile(self):
        return self.get_percentile()

    @property
    def fpr(self):
        return self.get_fpr()

    @property
    def fdr(self):
        return self.get_fdr()

    @property
    def fwe(self):
        return self.get_fwe()


class ChiSqSelectorModel(JavaFeatureModel, _ChiSqSelectorParams):
    """
    A Model which selects features using the model data computed by :class:`ChiSqSelector`.
    """

    def __init__(self, java_model=None):
        super(ChiSqSelectorModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "chisqselector"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "ChiSqSelectorModel"


class ChiSqSelector(JavaFeatureEstimator, _ChiSqSelectorParams):
    """
    ChiSqSelector is an algorithm that selects categorical features to use for predicting a
    categorical label.

    The selector supports different selection methods as follows.

    <ul>
      <li>`numTopFeatures` chooses a fixed number of top features according to a chi-squared test.
      <li>`percentile` is similar but chooses a fraction of all features instead of a fixed number.
      <li>`fpr` chooses all features whose p-value are below a threshold, thus controlling the false
          positive rate of selection.
      <li>`fdr` uses the [Benjamini-Hochberg procedure]
          (https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini.E2.80.93Hochberg_procedure)
          to choose all features whose false discovery rate is below a threshold.
      <li>`fwe` chooses all features whose p-values are below a threshold. The threshold is scaled
          by 1/numFeatures, thus controlling the family-wise error rate of selection.
    </ul>

    By default, the selection method is `numTopFeatures`, with the default number of top features
    set to 50.
    """

    def __init__(self):
        super(ChiSqSelector, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> ChiSqSelectorModel:
        return ChiSqSelectorModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "chisqselector"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "ChiSqSelector"
