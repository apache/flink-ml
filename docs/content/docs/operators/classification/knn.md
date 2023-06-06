---
title: "KNN"
type: docs
aliases:
- /operators/classification/knn.html
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

## KNN

K Nearest Neighbor(KNN) is a classification algorithm. The basic assumption of
KNN is that if most of the nearest K neighbors of the provided sample belong to
the same label, then it is highly probable that the provided sample also belongs
to that label.

### Input Columns

| Param name  | Type    | Default      | Description       |
| :---------- | :------ | :----------- |:------------------|
| featuresCol | Vector  | `"features"` | Feature vector.   |
| labelCol    | Integer | `"label"`    | Label to predict. |

### Output Columns

| Param name    | Type    | Default        | Description      |
| :------------ | :------ | :------------- |:-----------------|
| predictionCol | Integer | `"prediction"` | Predicted label. |

### Parameters

Below are the parameters required by `KnnModel`.

| Key           | Default        | Type    | Required | Description                      |
|---------------| -------------- | ------- | -------- | -------------------------------- |
| k             | `5`            | Integer | no       | The number of nearest neighbors. |
| featuresCol   | `"features"`   | String  | no       | Features column name.            |
| predictionCol | `"prediction"` | String  | no       | Prediction column name.          |

`Knn` needs parameters above and also below.

| Key      | Default   | Type   | Required | Description        |
| -------- | --------- | ------ | -------- | ------------------ |
| labelCol | `"label"` | String | no       | Label column name. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}
```java
import org.apache.flink.ml.classification.knn.Knn;
import org.apache.flink.ml.classification.knn.KnnModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a Knn model and uses it for classification. */
public class KnnExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(
                        Row.of(Vectors.dense(2.0, 3.0), 1.0),
                        Row.of(Vectors.dense(2.1, 3.1), 1.0),
                        Row.of(Vectors.dense(200.1, 300.1), 2.0),
                        Row.of(Vectors.dense(200.2, 300.2), 2.0),
                        Row.of(Vectors.dense(200.3, 300.3), 2.0),
                        Row.of(Vectors.dense(200.4, 300.4), 2.0),
                        Row.of(Vectors.dense(200.4, 300.4), 2.0),
                        Row.of(Vectors.dense(200.6, 300.6), 2.0),
                        Row.of(Vectors.dense(2.1, 3.1), 1.0),
                        Row.of(Vectors.dense(2.1, 3.1), 1.0),
                        Row.of(Vectors.dense(2.1, 3.1), 1.0),
                        Row.of(Vectors.dense(2.1, 3.1), 1.0),
                        Row.of(Vectors.dense(2.3, 3.2), 1.0),
                        Row.of(Vectors.dense(2.3, 3.2), 1.0),
                        Row.of(Vectors.dense(2.8, 3.2), 3.0),
                        Row.of(Vectors.dense(300., 3.2), 4.0),
                        Row.of(Vectors.dense(2.2, 3.2), 1.0),
                        Row.of(Vectors.dense(2.4, 3.2), 5.0),
                        Row.of(Vectors.dense(2.5, 3.2), 5.0),
                        Row.of(Vectors.dense(2.5, 3.2), 5.0),
                        Row.of(Vectors.dense(2.1, 3.1), 1.0));
        Table trainTable = tEnv.fromDataStream(trainStream).as("features", "label");

        DataStream<Row> predictStream =
                env.fromElements(
                        Row.of(Vectors.dense(4.0, 4.1), 5.0), Row.of(Vectors.dense(300, 42), 2.0));
        Table predictTable = tEnv.fromDataStream(predictStream).as("features", "label");

        // Creates a Knn object and initializes its parameters.
        Knn knn = new Knn().setK(4);

        // Trains the Knn Model.
        KnnModel knnModel = knn.fit(trainTable);

        // Uses the Knn Model for predictions.
        Table outputTable = knnModel.transform(predictTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector features = (DenseVector) row.getField(knn.getFeaturesCol());
            double expectedResult = (Double) row.getField(knn.getLabelCol());
            double predictionResult = (Double) row.getField(knn.getPredictionCol());
            System.out.printf(
                    "Features: %-15s \tExpected Result: %s \tPrediction Result: %s\n",
                    features, expectedResult, predictionResult);
        }
    }
}

```
{{< /tab>}}

{{< tab "Python">}}
```python
# Simple program that trains a Knn model and uses it for classification.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.classification.knn import KNN
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input training and prediction data
train_data = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense([2.0, 3.0]), 1.0),
        (Vectors.dense([2.1, 3.1]), 1.0),
        (Vectors.dense([200.1, 300.1]), 2.0),
        (Vectors.dense([200.2, 300.2]), 2.0),
        (Vectors.dense([200.3, 300.3]), 2.0),
        (Vectors.dense([200.4, 300.4]), 2.0),
        (Vectors.dense([200.4, 300.4]), 2.0),
        (Vectors.dense([200.6, 300.6]), 2.0),
        (Vectors.dense([2.1, 3.1]), 1.0),
        (Vectors.dense([2.1, 3.1]), 1.0),
        (Vectors.dense([2.1, 3.1]), 1.0),
        (Vectors.dense([2.1, 3.1]), 1.0),
        (Vectors.dense([2.3, 3.2]), 1.0),
        (Vectors.dense([2.3, 3.2]), 1.0),
        (Vectors.dense([2.8, 3.2]), 3.0),
        (Vectors.dense([300., 3.2]), 4.0),
        (Vectors.dense([2.2, 3.2]), 1.0),
        (Vectors.dense([2.4, 3.2]), 5.0),
        (Vectors.dense([2.5, 3.2]), 5.0),
        (Vectors.dense([2.5, 3.2]), 5.0),
        (Vectors.dense([2.1, 3.1]), 1.0)
    ],
        type_info=Types.ROW_NAMED(
            ['features', 'label'],
            [DenseVectorTypeInfo(), Types.DOUBLE()])))

predict_data = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense([4.0, 4.1]), 5.0),
        (Vectors.dense([300, 42]), 2.0),
    ],
        type_info=Types.ROW_NAMED(
            ['features', 'label'],
            [DenseVectorTypeInfo(), Types.DOUBLE()])))

# create a knn object and initialize its parameters
knn = KNN().set_k(4)

# train the knn model
model = knn.fit(train_data)

# use the knn model for predictions
output = model.transform(predict_data)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    features = result[field_names.index(knn.get_features_col())]
    expected_result = result[field_names.index(knn.get_label_col())]
    actual_result = result[field_names.index(knn.get_prediction_col())]
    print('Features: ' + str(features) + ' \tExpected Result: ' + str(expected_result)
          + ' \tActual Result: ' + str(actual_result))
```
{{< /tab>}}

{{< /tabs>}}

