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
import os
import sys
from pathlib import Path

# Because the project and the dependent `pyflink` project have the same directory structure,
# we need to manually add `flink-ml-python` path to `sys.path` in the test of this project to change
# the order of package search.
flink_ml_python_dir = Path(__file__).parents[5]
sys.path.append(str(flink_ml_python_dir))

import pyflink

pyflink.__path__.insert(0, os.path.join(flink_ml_python_dir, 'pyflink'))
