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

import importlib
import pkgutil

from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class ExamplesTest(PyFlinkMLTestCase):
    def test_examples(self):
        self.execute_all_in_module('pyflink.examples.ml.classification')
        self.execute_all_in_module('pyflink.examples.ml.clustering')
        self.execute_all_in_module('pyflink.examples.ml.evaluation')
        self.execute_all_in_module('pyflink.examples.ml.feature')
        self.execute_all_in_module('pyflink.examples.ml.regression')

    def execute_all_in_module(self, module):
        module = importlib.import_module(module)
        for importer, sub_modname, ispkg in pkgutil.iter_modules(module.__path__):
            # importing an example module means executing the example.
            importlib.import_module(module.__name__ + "." + sub_modname)
