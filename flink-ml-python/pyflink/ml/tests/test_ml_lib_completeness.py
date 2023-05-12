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
import glob
import importlib
import os
import pkgutil
import unittest
import inspect
from abc import abstractmethod
from typing import List

from pyflink.java_gateway import get_gateway
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class CompletenessTest(object):
    def test_completeness(self):
        self.assertEqual(
            self._java_stages,
            self._python_stages,
            'java algorithms {0} does not match aligned python algorithms {1}'.format(
                self._java_stages,
                self._python_stages))


class MLLibTest(PyFlinkMLTestCase):
    def setUp(self) -> None:
        super(MLLibTest, self).setUp()
        self._java_stages = self._load_java_stages()
        self._python_stages = self._load_python_stages()

    def _load_java_stages(self):
        this_directory = os.path.abspath(os.path.dirname(__file__))
        FLINK_ML_LIB_SOURCE_PATH = os.path.abspath(os.path.join(
            this_directory, "../../../../flink-ml-lib"))

        paths = glob.glob(os.path.join(
            FLINK_ML_LIB_SOURCE_PATH, "target", "flink-ml-lib-*.jar"))
        paths = [path for path in paths if "test" not in path]
        if len(paths) != 1:
            raise Exception("The number of matched paths " + str(paths) + " is unexpected.")
        ml_lib_jar = paths[0]

        StageAnalyzer = get_gateway().jvm.org.apache.flink.ml.util.StageAnalyzer
        module_path = 'org.apache.flink.ml.{0}'.format(self.module_name())
        excluded_stages = list(map(lambda x: f'{module_path}.{x}', self.exclude_java_stage()))
        return sorted([stage for stage in StageAnalyzer.analyzeLibJars(ml_lib_jar)
                       if module_path in stage
                       and stage not in excluded_stages])

    def _load_python_stages(self):
        modules = [importlib.import_module('.'.join([self.module().__name__, file_name]))
                   for _, file_name, _ in pkgutil.walk_packages(self.module().__path__)
                   if 'tests' not in file_name and 'common' not in file_name]
        return sorted([stage._java_stage_path() for name, stage in
                       sum([self._load_stages_from_module(module) for module in modules],
                           [])])

    @classmethod
    def _load_stages_from_module(cls, module):
        return [(name, obj) for name, obj in module.__dict__.items()
                if hasattr(obj, '_java_stage_path') and not inspect.isabstract(obj)
                and name not in (
                    'JavaClassificationEstimator', 'JavaClassificationModel',
                    'JavaClusteringEstimator', 'JavaClusteringModel',
                    'JavaClusteringAlgoOperator', 'JavaEvaluationAlgoOperator',
                    'JavaFeatureTransformer', 'JavaFeatureEstimator',
                    'JavaFeatureModel', 'JavaRegressionEstimator',
                    'JavaRegressionModel', 'JavaStatsAlgoOperator')]

    @abstractmethod
    def module_name(self):
        pass

    @abstractmethod
    def module(self):
        pass

    def exclude_java_stage(self):
        return []


class ClassificationCompletenessTest(CompletenessTest, MLLibTest):
    def module_name(self):
        return 'classification'

    def module(self):
        from pyflink.ml import classification
        return classification

    def exclude_java_stage(self) -> List[str]:
        # TODO: Add python support for LogisticRegressionWithFtrl.
        return [
            "logisticregression.LogisticRegressionWithFtrl",
        ]


class ClusteringCompletenessTest(CompletenessTest, MLLibTest):

    def module_name(self):
        return "clustering"

    def module(self):
        from pyflink.ml import clustering
        return clustering


class EvaluationCompletenessTest(CompletenessTest, MLLibTest):

    def module_name(self):
        return "evaluation"

    def module(self):
        from pyflink.ml import evaluation
        return evaluation


class FeatureCompletenessTest(CompletenessTest, MLLibTest):

    def module_name(self):
        return "feature"

    def module(self):
        from pyflink.ml import feature
        return feature


class RegressionCompletenessTest(CompletenessTest, MLLibTest):

    def module_name(self):
        return "regression"

    def module(self):
        from pyflink.ml import regression
        return regression


class StatsCompletenessTest(CompletenessTest, MLLibTest):

    def module_name(self):
        return "stats"

    def module(self):
        from pyflink.ml import stats
        return stats

    def exclude_java_stage(self) -> List[str]:
        return [
            "anovatest.ANOVATest",
            "fvaluetest.FValueTest",
        ]


if __name__ == "__main__":
    try:
        import xmlrunner

        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
