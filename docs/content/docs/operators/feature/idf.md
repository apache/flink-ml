---
title: "IDF"
weight: 1
type: docs
aliases:
- /operators/feature/IDF.html
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

## IDF

IDF computes the inverse document frequency (IDF) for the
input documents. IDF is computed following 
`idf = log((m + 1) / (d(t) + 1))`, where `m` is the total
number of documents and `d(t)` is the number of documents
that contains `t`.

IDFModel further uses the computed inverse document frequency
to compute [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

### Input Columns

| Param name | Type   | Default   | Description      |
|:-----------|:-------|:----------|:-----------------|
| inputCol   | Vector | `"input"` | Input documents. |

### Output Columns

| Param name | Type   | Default    | Description                 |
|:-----------|:-------|:-----------|:----------------------------|
| outputCol  | Vector | `"output"` | Tf-idf values of the input. |

### Parameters

Below are the parameters required by `IDFModel`.

| Key       | Default    | Type   | Required | Description         |
|:----------|:-----------|:-------|:---------|:--------------------|
| inputCol  | `"input"`  | String | no       | Input column name.  |
| outputCol | `"output"` | String | no       | Output column name. |


`IDF` needs parameters above and also below.

| Key        | Default    | Type    | Required | Description                                                          |
|:-----------|:-----------|:--------|:---------|:---------------------------------------------------------------------|
| minDocFreq | `0`        | Integer | no       | Minimum number of documents that a term should appear for filtering. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.idf.IDF;
import org.apache.flink.ml.feature.idf.IDFModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains an IDF model and uses it for feature engineering. */
public class IDFExample {
	public static void main(String[] args) {
		StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

		// Generates input data.
		DataStream<Row> inputStream =
			env.fromElements(
				Row.of(Vectors.dense(0, 1, 0, 2)),
				Row.of(Vectors.dense(0, 1, 2, 3)),
				Row.of(Vectors.dense(0, 1, 0, 0)));

		Table inputTable = tEnv.fromDataStream(inputStream).as("input");

		// Creates an IDF object and initializes its parameters.
		IDF idf = new IDF().setMinDocFreq(2);

		// Trains the IDF Model.
		IDFModel model = idf.fit(inputTable);

		// Uses the IDF Model for predictions.
		Table outputTable = model.transform(inputTable)[0];

		// Extracts and displays the results.
		for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
			Row row = it.next();
			DenseVector inputValue = (DenseVector) row.getField(idf.getInputCol());
			DenseVector outputValue = (DenseVector) row.getField(idf.getOutputCol());
			System.out.printf("Input Value: %s\tOutput Value: %s\n", inputValue, outputValue);
		}
	}
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that trains an IDF model and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.idf import IDF
from pyflink.table import StreamTableEnvironment

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input for training and prediction.
input_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(0, 1, 0, 2),),
        (Vectors.dense(0, 1, 2, 3),),
        (Vectors.dense(0, 1, 0, 0),),
    ],
        type_info=Types.ROW_NAMED(
            ['input', ],
            [DenseVectorTypeInfo(), ])))

# Creates an IDF object and initializes its parameters.
idf = IDF().set_min_doc_freq(2)

# Trains the IDF Model.
model = idf.fit(input_table)

# Uses the IDF Model for predictions.
output = model.transform(input_table)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_index = field_names.index(idf.get_input_col())
    output_index = field_names.index(idf.get_output_col())
    print('Input Value: ' + str(result[input_index]) +
          '\tOutput Value: ' + str(result[output_index]))
```

{{< /tab>}}

{{< /tabs>}}
