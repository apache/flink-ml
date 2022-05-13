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
from abc import abstractmethod

from pyflink.ml.tests.test_utils import PyFlinkMLTestCase

from pyflink.java_gateway import get_gateway


class CompletenessTest(object):
    def test_completeness(self):
        self.assertEqual(
            self._java_transformers,
            self._python_transformers,
            'java algorithms {0} does not match aligned python algorithms {1}'.format(
                self._java_transformers,
                self._python_transformers))


class MLLibTest(PyFlinkMLTestCase):
    def setUp(self) -> None:
        super(MLLibTest, self).setUp()
        self._java_transformers = self._load_java_transformers()
        self._python_transformers = self._load_python_transformers()

    def _load_java_transformers(self):
        this_directory = os.path.abspath(os.path.dirname(__file__))
        FLINK_ML_LIB_SOURCE_PATH = os.path.abspath(os.path.join(
            this_directory, "../../../../../flink-ml-lib"))

        ml_lib_jar = glob.glob(os.path.join(
            FLINK_ML_LIB_SOURCE_PATH, "target", "flink-ml-lib_*-SNAPSHOT.jar"))[0]

        TransformerAnalyzer = get_gateway().jvm.org.apache.flink.ml.util.TransformerAnalyzer
        return sorted([transformer for transformer in TransformerAnalyzer.analyzeLibJars(ml_lib_jar)
                       if 'org.apache.flink.ml.{0}.'.format(self.module_name()) in transformer])

    def _load_python_transformers(self):
        modules = [importlib.import_module('.'.join([self.module().__name__, file_name]))
                   for _, file_name, _ in pkgutil.walk_packages(self.module().__path__)
                   if 'tests' not in file_name and 'common' not in file_name]
        return sorted([transformer._java_stage_path() for name, transformer in
                       sum([self._load_transformers_from_module(module) for module in modules],
                           [])])

    @classmethod
    def _load_transformers_from_module(cls, module):
        return [(name, obj) for name, obj in module.__dict__.items()
                if hasattr(obj, 'transform') and name not in (
                    'JavaClassificationModel', 'JavaClusteringModel', 'JavaFeatureTransformer',
                    'JavaFeatureModel')]

    @abstractmethod
    def module_name(self):
        pass

    @abstractmethod
    def module(self):
        pass


class ClassificationCompletenessTest(CompletenessTest, MLLibTest):
    def module_name(self):
        return 'classification'

    def module(self):
        from pyflink.ml.lib import classification
        return classification


class ClusteringCompletenessTest(CompletenessTest, MLLibTest):

    def module_name(self):
        return "clustering"

    def module(self):
        from pyflink.ml.lib import clustering
        return clustering


class FeatureCompletenessTest(CompletenessTest, MLLibTest):

    def module_name(self):
        return "feature"

    def module(self):
        from pyflink.ml.lib import feature
        return feature


if __name__ == "__main__":
    try:
        import xmlrunner

        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
