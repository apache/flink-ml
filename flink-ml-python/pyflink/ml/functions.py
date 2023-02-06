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
from pyflink.ml.wrapper import call_java_table_function
from pyflink.table import Expression


def vector_to_array(*args) -> Expression:
    """
    Converts a column of :class:`Vector`s into a column of double arrays.
    """
    return call_java_table_function('org.apache.flink.ml.Functions.vectorToArray', *args)


def array_to_vector(*args) -> Expression:
    """
    Converts a column of arrays of numeric type into a column of
    :class:`DenseVector` instances.
    """
    return call_java_table_function('org.apache.flink.ml.Functions.arrayToVector', *args)
