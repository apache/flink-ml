---
title: "Bucketizer"
weight: 1
type: docs
aliases:
- /operators/feature/bucketizer.html
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

## Bucketizer

Bucketizer is an algorithm that maps multiple columns of continuous features to
multiple columns of discrete features, i.e., buckets indices. The indices are in
[0, numSplitsInThisColumn - 1].
### Input Columns

| Param name | Type   | Default | Description                          |
| :--------- | :----- | :------ | :----------------------------------- |
| inputCols  | Number | `null`  | Continuous features to be bucketized |

### Output Columns

| Param name | Type   | Default | Description                  |
| :--------- | :----- | :------ | :--------------------------- |
| outputCols | Double | `null`  | Discrete bucketized features |

### Parameters

| Key           | Default                          | Type        | Required | Description                                                  |
| ------------- | -------------------------------- | ----------- | -------- | ------------------------------------------------------------ |
| inputCols     | `null`                           | String      | yes      | Input column names.                                          |
| outputCols    | `null`                           | String      | yes      | Output column names.                                         |
| handleInvalid | `HasHandleInvalid.ERROR_INVALID` | String      | No       | Strategy to handle invalid entries.                          |
| splitsArray   | `null`                           | Double\[][] | yes      | Array of split points for mapping continuous features into buckets. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.feature.bucketizer.Bucketizer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that creates a Bucketizer instance and uses it for feature engineering. */
public class BucketizerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream = env.fromElements(Row.of(-0.5, 0.0, 1.0, 0.0));
        Table inputTable = tEnv.fromDataStream(inputStream).as("f1", "f2", "f3", "f4");

        // Creates a Bucketizer object and initializes its parameters.
        Double[][] splitsArray =
                new Double[][] {
                    new Double[] {-0.5, 0.0, 0.5},
                    new Double[] {-1.0, 0.0, 2.0},
                    new Double[] {Double.NEGATIVE_INFINITY, 10.0, Double.POSITIVE_INFINITY},
                    new Double[] {Double.NEGATIVE_INFINITY, 1.5, Double.POSITIVE_INFINITY}
                };
        Bucketizer bucketizer =
                new Bucketizer()
                        .setInputCols("f1", "f2", "f3", "f4")
                        .setOutputCols("o1", "o2", "o3", "o4")
                        .setSplitsArray(splitsArray)
                        .setHandleInvalid(HasHandleInvalid.SKIP_INVALID);

        // Uses the Bucketizer object for feature transformations.
        Table outputTable = bucketizer.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            double[] inputValues = new double[bucketizer.getInputCols().length];
            double[] outputValues = new double[bucketizer.getInputCols().length];
            for (int i = 0; i < inputValues.length; i++) {
                inputValues[i] = (double) row.getField(bucketizer.getInputCols()[i]);
                outputValues[i] = (double) row.getField(bucketizer.getOutputCols()[i]);
            }

            System.out.printf(
                    "Input Values: %s\tOutput Values: %s\n",
                    Arrays.toString(inputValues), Arrays.toString(outputValues));
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates a Bucketizer instance and uses it for feature
# engineering.
#
# Before executing this program, please make sure you have followed Flink ML's
# quick start guideline to set up Flink ML and Flink environment. The guideline
# can be found at
#
# https://nightlies.apache.org/flink/flink-ml-docs-master/docs/try-flink-ml/quick-start/

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.lib.feature.bucketizer import Bucketizer
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data = t_env.from_data_stream(
    env.from_collection([
        (-0.5, 0.0, 1.0, 0.0),
    ],
        type_info=Types.ROW_NAMED(
            ['f1', 'f2', 'f3', 'f4'],
            [Types.DOUBLE(), Types.DOUBLE(), Types.DOUBLE(), Types.DOUBLE()])
    ))

# create a bucketizer object and initialize its parameters
splits_array = [
    [-0.5, 0.0, 0.5],
    [-1.0, 0.0, 2.0],
    [float('-inf'), 10.0, float('inf')],
    [float('-inf'), 1.5, float('inf')],
]

bucketizer = Bucketizer() \
    .set_input_cols('f1', 'f2', 'f3', 'f4') \
    .set_output_cols('o1', 'o2', 'o3', 'o4') \
    .set_splits_array(splits_array)

# use the bucketizer model for feature engineering
output = bucketizer.transform(input_data)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
input_values = [None for _ in bucketizer.get_input_cols()]
output_values = [None for _ in bucketizer.get_input_cols()]
for result in t_env.to_data_stream(output).execute_and_collect():
    for i in range(len(bucketizer.get_input_cols())):
        input_values[i] = result[field_names.index(bucketizer.get_input_cols()[i])]
        output_values[i] = result[field_names.index(bucketizer.get_output_cols()[i])]
    print('Input Values: ' + str(input_values) + '\tOutput Values: ' + str(output_values))

```

{{< /tab>}}

{{< /tabs>}}
