---
title: "FeatureHasher"
weight: 1
type: docs
aliases:
- /operators/feature/featurehasher.html
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

## FeatureHasher

FeatureHasher transforms a set of categorical or numerical features into a sparse vector of
a specified dimension. The rules of hashing categorical columns and numerical columns are as
follows:

<ul>
<li>For numerical columns, the index of this feature in the output vector is the hash value of
      the column name and its correponding value is the same as the input.
<li>For categorical columns, the index of this feature in the output vector is the hash value
      of the string "column_name=value" and the corresponding value is 1.0.
</ul>

<p>If multiple features are projected into the same column, the output values are accumulated.
For the hashing trick, see https://en.wikipedia.org/wiki/Feature_hashing for details.

### Input Columns

| Param name | Type                  | Default | Description           |
|:-----------|:----------------------|:--------|:----------------------|
| inputCols  | Number/String/Boolean | `null`  | Columns to be hashed. |

### Output Columns

| Param name | Type   | Default    | Description    |
|:-----------|:-------|:-----------|:---------------|
| outputCol  | Vector | `"output"` | Output vector. |

### Parameters

| Key             | Default    | Type      | Required | Description               |
|-----------------|------------|-----------|----------|---------------------------|
| inputCols       | `null`     | String[]  | yes      | Input column names.       |
| outputCol       | `"output"` | String    | no       | Output column name.       |
| categoricalCols | `[]`       | String[]  | no       | Categorical column names. |
| numFeatures     | `262144`   | Integer   | no       | The number of features.   |
### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.featurehasher.FeatureHasher;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that creates a FeatureHasher instance and uses it for feature engineering. */
public class FeatureHasherExample {
    public static void main(String[] args) {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> dataStream =
                env.fromCollection(
                        Arrays.asList(Row.of(0, "a", 1.0, true), Row.of(1, "c", 1.0, false)));
        Table inputDataTable = tEnv.fromDataStream(dataStream).as("id", "f0", "f1", "f2");

        // Creates a FeatureHasher object and initializes its parameters.
        FeatureHasher featureHash =
                new FeatureHasher()
                        .setInputCols("f0", "f1", "f2")
                        .setCategoricalCols("f0", "f2")
                        .setOutputCol("vec")
                        .setNumFeatures(1000);

        // Uses the FeatureHasher object for feature transformations.
        Table outputTable = featureHash.transform(inputDataTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            Object[] inputValues = new Object[featureHash.getInputCols().length];
            for (int i = 0; i < inputValues.length; i++) {
                inputValues[i] = row.getField(featureHash.getInputCols()[i]);
            }
            Vector outputValue = (Vector) row.getField(featureHash.getOutputCol());

            System.out.printf(
                    "Input Values: %s \tOutput Value: %s\n",
                    Arrays.toString(inputValues), outputValue);
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates a FeatureHasher instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.featurehasher import FeatureHasher
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (0, 'a', 1.0, True),
        (1, 'c', 1.0, False),
    ],
        type_info=Types.ROW_NAMED(
            ['id', 'f0', 'f1', 'f2'],
            [Types.INT(), Types.STRING(), Types.DOUBLE(), Types.BOOLEAN()])))

# create a feature hasher object and initialize its parameters
feature_hasher = FeatureHasher() \
    .set_input_cols('f0', 'f1', 'f2') \
    .set_categorical_cols('f0', 'f2') \
    .set_output_col('vec') \
    .set_num_features(1000)

# use the feature hasher for feature engineering
output = feature_hasher.transform(input_data_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
input_values = [None for _ in feature_hasher.get_input_cols()]
for result in t_env.to_data_stream(output).execute_and_collect():
    for i in range(len(feature_hasher.get_input_cols())):
        input_values[i] = result[field_names.index(feature_hasher.get_input_cols()[i])]
    output_value = result[field_names.index(feature_hasher.get_output_col())]
    print('Input Values: ' + str(input_values) + '\tOutput Value: ' + str(output_value))

```

{{< /tab>}}

{{< /tabs>}}
