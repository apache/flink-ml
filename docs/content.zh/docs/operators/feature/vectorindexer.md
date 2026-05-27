---
title: "VectorIndexer"
weight: 1
type: docs
aliases:
- /operators/feature/vectorindexer.html
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

## VectorIndexer

VectorIndexer is an algorithm that implements the vector 
indexing algorithm. A vector indexer maps each column of
the input vector into a continuous/categorical feature.
Whether one feature is transformed into a continuous or
categorical feature depends on the number of distinct 
values in this column. If the number of distinct values
in one column is greater than a specified parameter 
(i.e., maxCategories), the corresponding output column
is unchanged. Otherwise, it is transformed into a 
categorical value. For categorical outputs, the indices
are in [0, numDistinctValuesInThisColumn].

The output model is organized in ascending order except
that 0.0 is always mapped to 0 (for sparsity).

### Input Columns

| Param name | Type   | Default   | Description            |
|:-----------|:-------|:----------|:-----------------------|
| inputCol   | Vector | `"input"` | Vectors to be indexed. |

### Output Columns

| Param name | Type   | Default    | Description      |
|:-----------|:-------|:-----------|:-----------------|
| outputCol  | Vector | `"output"` | Indexed vectors. |

### Parameters

Below are the parameters required by `VectorIndexerModel`.

| Key           | Default    | Type   | Required | Description                                                                      |
|:--------------|:-----------|:-------|:---------|:---------------------------------------------------------------------------------|
| inputCol      | `"input"`  | String | no       | Input column name.                                                               |
| outputCol     | `"output"` | String | no       | Output column name.                                                              |
| handleInvalid | `"error"`  | String | no       | Strategy to handle invalid entries. Supported values: `'error', 'skip', 'keep'`. |

`VectorIndexer` needs parameters above and also below.

| Key           | Default | Type    | Required | Description                                                                                                                                                     |
|:--------------|:--------|:--------|:---------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| maxCategories | `20`    | Integer | no       | Threshold for the number of values a categorical feature can take (>= 2). If a feature is found to have > maxCategories values, then it is declared continuous. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.feature.vectorindexer.VectorIndexer;
import org.apache.flink.ml.feature.vectorindexer.VectorIndexerModel;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.List;

/** Simple program that creates a VectorIndexer instance and uses it for feature engineering. */
public class VectorIndexerExample {
	public static void main(String[] args) {
		StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

		// Generates input data.
		List<Row> trainInput =
			Arrays.asList(
				Row.of(Vectors.dense(1, 1)),
				Row.of(Vectors.dense(2, -1)),
				Row.of(Vectors.dense(3, 1)),
				Row.of(Vectors.dense(4, 0)),
				Row.of(Vectors.dense(5, 0)));

		List<Row> predictInput =
			Arrays.asList(
				Row.of(Vectors.dense(0, 2)),
				Row.of(Vectors.dense(0, 0)),
				Row.of(Vectors.dense(0, -1)));

		Table trainTable = tEnv.fromDataStream(env.fromCollection(trainInput)).as("input");
		Table predictTable = tEnv.fromDataStream(env.fromCollection(predictInput)).as("input");

		// Creates a VectorIndexer object and initializes its parameters.
		VectorIndexer vectorIndexer =
			new VectorIndexer()
				.setInputCol("input")
				.setOutputCol("output")
				.setHandleInvalid(HasHandleInvalid.KEEP_INVALID)
				.setMaxCategories(3);

		// Trains the VectorIndexer Model.
		VectorIndexerModel model = vectorIndexer.fit(trainTable);

		// Uses the VectorIndexer Model for predictions.
		Table outputTable = model.transform(predictTable)[0];

		// Extracts and displays the results.
		for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
			Row row = it.next();
			System.out.printf(
				"Input Value: %s \tOutput Value: %s\n",
				row.getField(vectorIndexer.getInputCol()),
				row.getField(vectorIndexer.getOutputCol()));
		}
	}
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that trains a VectorIndexer model and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.vectorindexer import VectorIndexer
from pyflink.table import StreamTableEnvironment

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input training and prediction data.
train_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(1, 1),),
        (Vectors.dense(2, -1),),
        (Vectors.dense(3, 1),),
        (Vectors.dense(4, 0),),
        (Vectors.dense(5, 0),)
    ],
        type_info=Types.ROW_NAMED(
            ['input', ],
            [DenseVectorTypeInfo(), ])))

predict_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(0, 2),),
        (Vectors.dense(0, 0),),
        (Vectors.dense(0, -1),),
    ],
        type_info=Types.ROW_NAMED(
            ['input', ],
            [DenseVectorTypeInfo(), ])))

# Creates a VectorIndexer object and initializes its parameters.
vector_indexer = VectorIndexer() \
    .set_input_col('input') \
    .set_output_col('output') \
    .set_handle_invalid('keep') \
    .set_max_categories(3)

# Trains the VectorIndexer Model.
model = vector_indexer.fit(train_table)

# Uses the VectorIndexer Model for predictions.
output = model.transform(predict_table)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    print('Input Value: ' + str(result[field_names.index(vector_indexer.get_input_col())])
          + '\tOutput Value: ' + str(result[field_names.index(vector_indexer.get_output_col())]))

```

{{< /tab>}}

{{< /tabs>}}
