---
title: "SQLTransformer"
weight: 1
type: docs
aliases:
- /operators/feature/sqltransformer.html
---

<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

## SQLTransformer

SQLTransformer implements the transformations that are defined by SQL statement.

Currently we only support SQL syntax like `SELECT ... FROM __THIS__ ...` where
`__THIS__` represents the input table and cannot be modified.

The select clause specifies the fields, constants, and expressions to display in
the output. Except the cases described in the note section below, it can be any
select clause that Flink SQL supports. Users can also use Flink SQL built-in
function and UDFs to operate on these selected columns.

For example, SQLTransformer supports statements like:

- `SELECT a, a + b AS a_b FROM __THIS__`
- `SELECT a, SQRT(b) AS b_sqrt FROM __THIS__ where a > 5`
- `SELECT a, b, SUM(c) AS c_sum FROM __THIS__ GROUP BY a, b`

Note: This operator only generates append-only/insert-only table as its output.
If the output table could possibly contain retract messages(e.g. perform `SELECT
... FROM __THIS__ GROUP BY ...` operation on a table in streaming mode), this
operator would aggregate all changelogs and only output the final state.

### Parameters

| Key       | Default | Type   | Required | Description    |
|:----------|:--------|:-------|:---------|:---------------|
| statement | `null`  | String | yes      | SQL statement. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.feature.sqltransformer.SQLTransformer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import java.util.Arrays;

/** Simple program that creates a SQLTransformer instance and uses it for feature engineering. */
public class SQLTransformerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromCollection(
                        Arrays.asList(Row.of(0, 1.0, 3.0), Row.of(2, 2.0, 5.0)),
                        new RowTypeInfo(Types.INT, Types.DOUBLE, Types.DOUBLE));
        Table inputTable = tEnv.fromDataStream(inputStream).as("id", "v1", "v2");

        // Creates a SQLTransformer object and initializes its parameters.
        SQLTransformer sqlTransformer =
                new SQLTransformer()
                        .setStatement("SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__");

        // Uses the SQLTransformer object for feature transformations.
        Table outputTable = sqlTransformer.transform(inputTable)[0];

        // Extracts and displays the results.
        outputTable.execute().print();
    }
}
```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates a SQLTransformer instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.lib.feature.sqltransformer import SQLTransformer
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()

t_env = StreamTableEnvironment.create(env)

# Generates input data.
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (0, 1.0, 3.0),
        (2, 2.0, 5.0),
    ],
        type_info=Types.ROW_NAMED(
            ['id', 'v1', 'v2'],
            [Types.INT(), Types.DOUBLE(), Types.DOUBLE()])))

# Creates a SQLTransformer object and initializes its parameters.
sql_transformer = SQLTransformer() \
    .set_statement('SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__')

# Uses the SQLTransformer object for feature transformations.
output_table = sql_transformer.transform(input_data_table)[0]

# Extracts and displays the results.
output_table.execute().print()
```

{{< /tab>}}

{{< /tabs>}}
