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
#  limitations under the License.
import typing
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.param import StringParam, FloatParam
from pyflink.ml.common.param import HasFeaturesCol, HasLabelCol, HasOutputCol
from pyflink.ml.feature.common import JavaFeatureModel, JavaFeatureEstimator


class _UnivariateFeatureSelectorModelParams(
    JavaWithParams,
    HasFeaturesCol,
    HasOutputCol
):
    """
    Params for :class `UnivariateFeatureSelectorModel`.
    """
    def __init__(self, java_params):
        super(_UnivariateFeatureSelectorModelParams, self).__init__(java_params)


class _UnivariateFeatureSelectorParams(HasLabelCol, _UnivariateFeatureSelectorModelParams):
    """
    Params for :class `UnivariateFeatureSelector`.
    """

    """
    Supported options of the feature type.

    <ul>
        <li>categorical: the features are categorical data.
        <li>continuous: the features are continuous data.
    </ul>
    """
    FEATURE_TYPE: StringParam = StringParam(
        "feature_type",
        "The feature type.",
        None)

    """
    Supported options of the label type.

    <ul>
        <li>categorical: the label is categorical data.
        <li>continuous: the label is continuous data.
    </ul>
    """
    LABEL_TYPE: StringParam = StringParam(
        "label_type",
        "The label type.",
        None)

    """
    Supported options of the feature selection mode.

    <ul>
        <li>numTopFeatures: chooses a fixed number of top features according to a hypothesis.
        <li>percentile: similar to numTopFeatures but chooses a fraction of all features
            instead of a fixed number.
        <li>fpr: chooses all features whose p-value are below a threshold, thus controlling the
            false positive rate of selection.
        <li>fdr: uses the <ahref="https://en.wikipedia.org/wiki/False_discovery_rate#
            Benjamini.E2.80.93Hochberg_procedure">Benjamini-Hochberg procedure</a> to choose
            all features whose false discovery rate is below a threshold.
        <li>fwe: chooses all features whose p-values are below a threshold. The threshold is
            scaled by 1/numFeatures, thus controlling the family-wise error rate of selection.
    </ul>
    """
    SELECTION_MODE: StringParam = StringParam(
        "selection_mode",
        "The feature selection mode.",
        "numTopFeatures")

    SELECTION_THRESHOLD: FloatParam = FloatParam(
        "selection_threshold",
        "The upper bound of the features that selector will select. If not set, it will be "
        "replaced with a meaningful value according to different selection modes at runtime. "
        "When the mode is numTopFeatures, it will be replaced with 50; when the mode is "
        "percentile, it will be replaced with 0.1; otherwise, it will be replaced with 0.05.",
        None)

    def __init__(self, java_params):
        super(_UnivariateFeatureSelectorParams, self).__init__(java_params)

    def set_feature_type(self, value: str):
        return typing.cast(_UnivariateFeatureSelectorParams, self.set(self.FEATURE_TYPE, value))

    def get_feature_type(self) -> str:
        return self.get(self.FEATURE_TYPE)

    def set_label_type(self, value: str):
        return typing.cast(_UnivariateFeatureSelectorParams, self.set(self.LABEL_TYPE, value))

    def get_label_type(self) -> str:
        return self.get(self.LABEL_TYPE)

    def set_selection_mode(self, value: str):
        return typing.cast(_UnivariateFeatureSelectorParams, self.set(self.SELECTION_MODE, value))

    def get_selection_mode(self) -> str:
        return self.get(self.SELECTION_MODE)

    def set_selection_threshold(self, value: float):
        return typing.cast(_UnivariateFeatureSelectorParams,
                           self.set(self.SELECTION_THRESHOLD, float(value)))

    def get_selection_threshold(self) -> float:
        return self.get(self.SELECTION_THRESHOLD)

    @property
    def feature_type(self):
        return self.get_feature_type()

    @property
    def label_type(self):
        return self.get_label_type()

    @property
    def selection_mode(self):
        return self.get_selection_mode()

    @property
    def selection_threshold(self):
        return self.get_selection_threshold()


class UnivariateFeatureSelectorModel(JavaFeatureModel, _UnivariateFeatureSelectorModelParams):
    """
    A Model which transforms data using the model data computed
    by :class:`UnivariateFeatureSelector`.
    """

    def __init__(self, java_model=None):
        super(UnivariateFeatureSelectorModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "univariatefeatureselector"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "UnivariateFeatureSelectorModel"


class UnivariateFeatureSelector(JavaFeatureEstimator, _UnivariateFeatureSelectorParams):
    """
    An Estimator which selects features based on univariate statistical tests against labels.

    Currently, Flink supports three Univariate Feature Selectors: chi-squared, ANOVA F-test and
    F-value. User can choose Univariate Feature Selector by setting `featureType` and `labelType`,
    and Flink will pick the score function based on the specified `featureType` and `labelType`.

    The following combination of `featureType` and `labelType` are supported:

    <ul>
        <li>`featureType` `categorical` and `labelType` `categorical`: Flink uses chi-squared,
            i.e. chi2 in sklearn.
        <li>`featureType` `continuous` and `labelType` `categorical`: Flink uses ANOVA F-test,
            i.e. f_classif in sklearn.
        <li>`featureType` `continuous` and `labelType` `continuous`: Flink uses F-value,
            i.e. f_regression in sklearn.
    </ul>

    The `UnivariateFeatureSelector` supports different selection modes:

    <ul>
        <li>numTopFeatures: chooses a fixed number of top features according to a hypothesis.
        <li>percentile: similar to numTopFeatures but chooses a fraction of all features
            instead of a fixed number.
        <li>fpr: chooses all features whose p-value are below a threshold, thus controlling
            the false positive rate of selection.
        <li>fdr: uses the <ahref="https://en.wikipedia.org/wiki/False_discovery_rate#
            Benjamini.E2.80.93Hochberg_procedure">Benjamini-Hochberg procedure</a> to choose
            all features whose false discovery rate is below a threshold.
        <li>fwe: chooses all features whose p-values are below a threshold. The threshold is
            scaled by 1/numFeatures, thus controlling the family-wise error rate of selection.
    </ul>

    By default, the selection mode is `numTopFeatures`.
    """

    def __init__(self):
        super(UnivariateFeatureSelector, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> UnivariateFeatureSelectorModel:
        return UnivariateFeatureSelectorModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "univariatefeatureselector"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "UnivariateFeatureSelector"
