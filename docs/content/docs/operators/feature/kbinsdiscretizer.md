---
title: "KBinsDiscretizer"
weight: 1
type: docs
aliases:
- /operators/feature/kbinsdiscretizer.html
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

## KBinsDiscretizer

KBinsDiscretizer is an algorithm that implements discretization (also known as
quantization or binning) to transform continuous features into discrete ones.
The output values are in [0, numBins).

### Input Columns

| Param name | Type        | Default   | Description                |
|:-----------|:------------|:----------|:---------------------------|
| inputCol   | DenseVector | `"input"` | Vectors to be discretized. |

### Output Columns

| Param name | Type        | Default    | Description          |
|:-----------|:------------|:-----------|:---------------------|
| outputCol  | DenseVector | `"output"` | Discretized vectors. |

### Parameters

Below are the parameters required by `KBinsDiscretizerModel`.

| Key       | Default    | Type   | Required | Description         |
|:----------|:-----------|:-------|:---------|:--------------------|
| inputCol  | `"input"`  | String | no       | Input column name.  |
| outputCol | `"output"` | String | no       | Output column name. |

`KBinsDiscretizer` needs parameters above and also below.

| Key        | Default      | Type    | Required | Description                                                                                      | 
|:-----------|:-------------|:--------|:---------|:-------------------------------------------------------------------------------------------------|
| strategy   | `"quantile"` | String  | no       | Strategy used to define the width of the bin. Supported values: 'uniform', 'quantile', 'kmeans'. |
| numBins    | `5`          | Integer | no       | Number of bins to produce.                                                                       |
| subSamples | `200000`     | Integer | no       | Maximum number of samples used to fit the model.                                                 |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizer;
import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizerModel;
import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizerParams;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a KBinsDiscretizer model and uses it for feature engineering. */
public class KBinsDiscretizerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(Vectors.dense(1, 10, 0)),
                        Row.of(Vectors.dense(1, 10, 0)),
                        Row.of(Vectors.dense(1, 10, 0)),
                        Row.of(Vectors.dense(4, 10, 0)),
                        Row.of(Vectors.dense(5, 10, 0)),
                        Row.of(Vectors.dense(6, 10, 0)),
                        Row.of(Vectors.dense(7, 10, 0)),
                        Row.of(Vectors.dense(10, 10, 0)),
                        Row.of(Vectors.dense(13, 10, 3)));
        Table inputTable = tEnv.fromDataStream(inputStream).as("input");

        // Creates a KBinsDiscretizer object and initializes its parameters.
        KBinsDiscretizer kBinsDiscretizer =
                new KBinsDiscretizer().setNumBins(3).setStrategy(KBinsDiscretizerParams.UNIFORM);

        // Trains the KBinsDiscretizer Model.
        KBinsDiscretizerModel model = kBinsDiscretizer.fit(inputTable);

        // Uses the KBinsDiscretizer Model for predictions.
        Table outputTable = model.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector inputValue = (DenseVector) row.getField(kBinsDiscretizer.getInputCol());
            DenseVector outputValue = (DenseVector) row.getField(kBinsDiscretizer.getOutputCol());
            System.out.printf("Input Value: %s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that trains a KBinsDiscretizer model and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.lib.feature.kbinsdiscretizer import KBinsDiscretizer
from pyflink.table import StreamTableEnvironment

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input for training and prediction.
input_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(1, 10, 0),),
        (Vectors.dense(1, 10, 0),),
        (Vectors.dense(1, 10, 0),),
        (Vectors.dense(4, 10, 0),),
        (Vectors.dense(5, 10, 0),),
        (Vectors.dense(6, 10, 0),),
        (Vectors.dense(7, 10, 0),),
        (Vectors.dense(10, 10, 0),),
        (Vectors.dense(13, 10, 0),),
    ],
        type_info=Types.ROW_NAMED(
            ['input', ],
            [DenseVectorTypeInfo(), ])))

# Creates a KBinsDiscretizer object and initializes its parameters.
k_bins_discretizer = KBinsDiscretizer() \
    .set_input_col('input') \
    .set_output_col('output') \
    .set_num_bins(3) \
    .set_strategy('uniform')

# Trains the KBinsDiscretizer Model.
model = k_bins_discretizer.fit(input_table)

# Uses the KBinsDiscretizer Model for predictions.
output = model.transform(input_table)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    print('Input Value: ' + str(result[field_names.index(k_bins_discretizer.get_input_col())])
          + '\tOutput Value: ' +
          str(result[field_names.index(k_bins_discretizer.get_output_col())]))

```

{{< /tab>}}

{{< /tabs>}}
