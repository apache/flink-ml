---
title: "Robust Scaler"
weight: 1
type: docs
aliases:
- /operators/feature/robustscaler.html
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

## Robust Scaler

Robust Scaler is an algorithm that scales features using statistics that are
robust to outliers.

This Scaler removes the median and scales the data according to the quantile
range (defaults to IQR: Interquartile Range). The IQR is the range between 
the 1st quartile (25th quantile) and the 3rd quartile (75th quantile) but can
be configured.

Centering and scaling happen independently on each feature by computing the 
relevant statistics on the samples in the training set. Median and quantile 
range are then stored to be used on later data using the transform method.

Standardization of a dataset is a common requirement for many machine learning
estimators. Typically this is done by removing the mean and scaling to unit 
variance. However, outliers can often influence the sample mean / variance 
in a negative way. In such cases, the median and the interquartile range 
often give better results.

Note that NaN values are ignored in the computation of medians and ranges.

### Input Columns

| Param name | Type   | Default   | Description            |
|:-----------|:-------|:----------|:-----------------------|
| inputCol   | Vector | `"input"` | Features to be scaled. |

### Output Columns

| Param name | Type   | Default    | Description      |
|:-----------|:-------|:-----------|:-----------------|
| outputCol  | Vector | `"output"` | Scaled features. |

### Parameters

Below are the parameters required by `RobustScalerModel`.

| Key           | Default    | Type        | Required | Description                                                           |
|---------------|------------|-------------|----------|-----------------------------------------------------------------------|
| inputCol      | `"input"`  | String      | no       | Input column name.                                                    |
| outputCol     | `"output"` | String      | no       | Output column name.                                                   |
| withCentering | `false`    | Boolean     | no       | Whether to center the data with median before scaling.                |
| withScaling   | `true`     | Boolean     | no       | Whether to scale the data to quantile range.                          |

`RobustScaler` needs parameters above and also below.

| Key           | Default      | Type        | Required | Description                                                           |
|---------------|--------------|-------------|----------|-----------------------------------------------------------------------|
| lower         | `0.25`       | Double      | no       | Lower quantile to calculate quantile range.                           |
| upper         | `0.75`       | Double      | no       | Upper quantile to calculate quantile range.                           |
| relativeError | `0.001`      | Double      | no       | The relative target precision for the approximate quantile algorithm. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.robustscaler.RobustScaler;
import org.apache.flink.ml.feature.robustscaler.RobustScalerModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a {@link RobustScaler} model and uses it for feature selection. */
public class RobustScalerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(
                        Row.of(1, Vectors.dense(0.0, 0.0)),
                        Row.of(2, Vectors.dense(1.0, -1.0)),
                        Row.of(3, Vectors.dense(2.0, -2.0)),
                        Row.of(4, Vectors.dense(3.0, -3.0)),
                        Row.of(5, Vectors.dense(4.0, -4.0)),
                        Row.of(6, Vectors.dense(5.0, -5.0)),
                        Row.of(7, Vectors.dense(6.0, -6.0)),
                        Row.of(8, Vectors.dense(7.0, -7.0)),
                        Row.of(9, Vectors.dense(8.0, -8.0)));
        Table trainTable = tEnv.fromDataStream(trainStream).as("id", "input");

        // Creates a RobustScaler object and initializes its parameters.
        RobustScaler robustScaler =
                new RobustScaler()
                        .setLower(0.25)
                        .setUpper(0.75)
                        .setRelativeError(0.001)
                        .setWithScaling(true)
                        .setWithCentering(true);

        // Trains the RobustScaler model.
        RobustScalerModel model = robustScaler.fit(trainTable);

        // Uses the RobustScaler model for predictions.
        Table outputTable = model.transform(trainTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector inputValue = (DenseVector) row.getField(robustScaler.getInputCol());
            DenseVector outputValue = (DenseVector) row.getField(robustScaler.getOutputCol());
            System.out.printf("Input Value: %-15s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}
```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates a RobustScaler instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo

from pyflink.ml.feature.robustscaler import RobustScaler

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input training and prediction data.
train_data = t_env.from_data_stream(
    env.from_collection([
        (1, Vectors.dense(0.0, 0.0),),
        (2, Vectors.dense(1.0, -1.0),),
        (3, Vectors.dense(2.0, -2.0),),
        (4, Vectors.dense(3.0, -3.0),),
        (5, Vectors.dense(4.0, -4.0),),
        (6, Vectors.dense(5.0, -5.0),),
        (7, Vectors.dense(6.0, -6.0),),
        (8, Vectors.dense(7.0, -7.0),),
        (9, Vectors.dense(8.0, -8.0),),
    ],
        type_info=Types.ROW_NAMED(
            ['id', 'input'],
            [Types.INT(), DenseVectorTypeInfo()])
    ))

# Creates an RobustScaler object and initializes its parameters.
robust_scaler = RobustScaler()\
    .set_lower(0.25)\
    .set_upper(0.75)\
    .set_relative_error(0.001)\
    .set_with_scaling(True)\
    .set_with_centering(True)

# Trains the RobustScaler Model.
model = robust_scaler.fit(train_data)

# Uses the RobustScaler Model for predictions.
output = model.transform(train_data)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_index = field_names.index(robust_scaler.get_input_col())
    output_index = field_names.index(robust_scaler.get_output_col())
    print('Input Value: ' + str(result[input_index]) +
          '\tOutput Value: ' + str(result[output_index]))

```

{{< /tab>}}

{{< /tabs>}}
