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

from pyflink.ml.param import Param, StringParam
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureTransformer


class _SQLTransformerParams(
    JavaWithParams
):
    """
    Params for :class:`SQLTransformer`.
    """

    STATEMENT: Param[str] = StringParam(
        "statement",
        "SQL statement.",
        None
    )

    def __init__(self, java_params):
        super(_SQLTransformerParams, self).__init__(java_params)

    def set_statement(self, value: str):
        return typing.cast(_SQLTransformerParams, self.set(self.STATEMENT, value))

    def get_statement(self) -> str:
        return self.get(self.STATEMENT)

    @property
    def statement(self) -> str:
        return self.get_statement()


class SQLTransformer(JavaFeatureTransformer, _SQLTransformerParams):
    """
    SQLTransformer implements the transformations that are defined by SQL statement.

    Currently we only support SQL syntax like `SELECT ... FROM __THIS__ ...` where `__THIS__`
    represents the input table and cannot be modified.

    The select clause specifies the fields, constants, and expressions to display in the output.
    Except the cases described in the note section below, it can be any select clause that Flink SQL
    supports. Users can also use Flink SQL built-in function and UDFs to operate on these selected
    columns.

    For example, SQLTransformer supports statements like:

    - `SELECT a, a + b AS a_b FROM __THIS__`
    - `SELECT a, SQRT(b) AS b_sqrt FROM __THIS__ where a > 5`
    - `SELECT a, b, SUM(c) AS c_sum FROM __THIS__ GROUP BY a, b`

    Note: This operator only generates append-only/insert-only table as its output. If the output
    table could possibly contain retract messages(e.g. perform `SELECT ... FROM __THIS__ GROUP BY
    ...` operation on a table in streaming mode), this operator would aggregate all changelogs and
    only output the final state.
    """

    def __init__(self, java_model=None):
        super(SQLTransformer, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "sqltransformer"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "SQLTransformer"
