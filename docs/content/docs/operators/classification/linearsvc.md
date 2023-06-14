---
title: "Linear SVC"
type: docs
aliases:
- /operators/classification/linearsvc.html
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

## Linear Support Vector Machine

Linear Support Vector Machine (Linear SVC) is an algorithm that attempts to find
a hyperplane to maximize the distance between classified samples.

### Input Columns

| Param name  | Type    | Default      | Description       |
| :---------- | :------ | :----------- |:------------------|
| featuresCol | Vector  | `"features"` | Feature vector.   |
| labelCol    | Integer | `"label"`    | Label to predict. |
| weightCol   | Double  | `"weight"`   | Weight of sample. |

### Output Columns

| Param name       | Type    | Default           | Description                              |
| :--------------- | :------ | :---------------- |:-----------------------------------------|
| predictionCol    | Integer | `"prediction"`    | Label of the max probability.            |
| rawPredictionCol | Vector  | `"rawPrediction"` | Vector of the probability of each label. |

### Parameters

Below are the parameters required by `LinearSVCModel`.

| Key              | Default           | Type   | Required | Description                                                             |
|------------------|-------------------|--------|----------|-------------------------------------------------------------------------|
| featuresCol      | `"features"`      | String | no       | Features column name.                                                   |
| predictionCol    | `"prediction"`    | String | no       | Prediction column name.                                                 |
| rawPredictionCol | `"rawPrediction"` | String | no       | Raw prediction column name.                                             |
| threshold        | `0.0`             | Double | no       | Threshold in binary classification prediction applied to rawPrediction. |

`LinearSVC` needs parameters above and also below.

| Key             | Default   | Type    | Required | Description                                     |
| --------------- | --------- | ------- | -------- | ----------------------------------------------- |
| labelCol        | `"label"` | String  | no       | Label column name.                              |
| weightCol       | `null`    | String  | no       | Weight column name.                             |
| maxIter         | `20`      | Integer | no       | Maximum number of iterations.                   |
| reg             | `0.`      | Double  | no       | Regularization parameter.                       |
| elasticNet      | `0.`      | Double  | no       | ElasticNet parameter.                           |
| learningRate    | `0.1`     | Double  | no       | Learning rate of optimization method.           |
| globalBatchSize | `32`      | Integer | no       | Global batch size of training algorithms.       |
| tol             | `1e-6`    | Double  | no       | Convergence tolerance for iterative algorithms. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.classification.linearsvc.LinearSVC;
import org.apache.flink.ml.classification.linearsvc.LinearSVCModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a LinearSVC model and uses it for classification. */
public class LinearSVCExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(Vectors.dense(1, 2, 3, 4), 0., 1.),
                        Row.of(Vectors.dense(2, 2, 3, 4), 0., 2.),
                        Row.of(Vectors.dense(3, 2, 3, 4), 0., 3.),
                        Row.of(Vectors.dense(4, 2, 3, 4), 0., 4.),
                        Row.of(Vectors.dense(5, 2, 3, 4), 0., 5.),
                        Row.of(Vectors.dense(11, 2, 3, 4), 1., 1.),
                        Row.of(Vectors.dense(12, 2, 3, 4), 1., 2.),
                        Row.of(Vectors.dense(13, 2, 3, 4), 1., 3.),
                        Row.of(Vectors.dense(14, 2, 3, 4), 1., 4.),
                        Row.of(Vectors.dense(15, 2, 3, 4), 1., 5.));
        Table inputTable = tEnv.fromDataStream(inputStream).as("features", "label", "weight");

        // Creates a LinearSVC object and initializes its parameters.
        LinearSVC linearSVC = new LinearSVC().setWeightCol("weight");

        // Trains the LinearSVC Model.
        LinearSVCModel linearSVCModel = linearSVC.fit(inputTable);

        // Uses the LinearSVC Model for predictions.
        Table outputTable = linearSVCModel.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector features = (DenseVector) row.getField(linearSVC.getFeaturesCol());
            double expectedResult = (Double) row.getField(linearSVC.getLabelCol());
            double predictionResult = (Double) row.getField(linearSVC.getPredictionCol());
            DenseVector rawPredictionResult =
                    (DenseVector) row.getField(linearSVC.getRawPredictionCol());
            System.out.printf(
                    "Features: %-25s \tExpected Result: %s \tPrediction Result: %s \tRaw Prediction Result: %s\n",
                    features, expectedResult, predictionResult, rawPredictionResult);
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that trains a LinearSVC model and uses it for classification.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.classification.linearsvc import LinearSVC
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense([1, 2, 3, 4]), 0., 1.),
        (Vectors.dense([2, 2, 3, 4]), 0., 2.),
        (Vectors.dense([3, 2, 3, 4]), 0., 3.),
        (Vectors.dense([4, 2, 3, 4]), 0., 4.),
        (Vectors.dense([5, 2, 3, 4]), 0., 5.),
        (Vectors.dense([11, 2, 3, 4]), 1., 1.),
        (Vectors.dense([12, 2, 3, 4]), 1., 2.),
        (Vectors.dense([13, 2, 3, 4]), 1., 3.),
        (Vectors.dense([14, 2, 3, 4]), 1., 4.),
        (Vectors.dense([15, 2, 3, 4]), 1., 5.),
    ],
        type_info=Types.ROW_NAMED(
            ['features', 'label', 'weight'],
            [DenseVectorTypeInfo(), Types.DOUBLE(), Types.DOUBLE()])
    ))

# create a linear svc object and initialize its parameters
linear_svc = LinearSVC().set_weight_col('weight')

# train the linear svc model
model = linear_svc.fit(input_table)

# use the linear svc model for predictions
output = model.transform(input_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    features = result[field_names.index(linear_svc.get_features_col())]
    expected_result = result[field_names.index(linear_svc.get_label_col())]
    prediction_result = result[field_names.index(linear_svc.get_prediction_col())]
    raw_prediction_result = result[field_names.index(linear_svc.get_raw_prediction_col())]
    print('Features: ' + str(features) + ' \tExpected Result: ' + str(expected_result)
          + ' \tPrediction Result: ' + str(prediction_result)
          + ' \tRaw Prediction Result: ' + str(raw_prediction_result))
```

{{< /tab>}}

{{< /tabs>}}
