---
title: "Linear Regression"
type: docs
aliases:
- /operators/regression/linearregression.html
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

## Linear Regression

Linear Regression is a kind of regression analysis by modeling the relationship
between a scalar response and one or more explanatory variables.

### Input Columns

| Param name  | Type    | Default      | Description      |
| :---------- | :------ | :----------- | :--------------- |
| featuresCol | Vector  | `"features"` | Feature vector   |
| labelCol    | Integer | `"label"`    | Label to predict |
| weightCol   | Double  | `"weight"`   | Weight of sample |

### Output Columns

| Param name    | Type    | Default        | Description                  |
| :------------ | :------ | :------------- | :--------------------------- |
| predictionCol | Integer | `"prediction"` | Label of the max probability |

### Parameters

Below are the parameters required by `LinearRegressionModel`.

| Key           | Default        | Type   | Required | Description             |
| ------------- | -------------- | ------ | -------- | ----------------------- |
| featuresCol   | `"features"`   | String | no       | Features column name.   |
| predictionCol | `"prediction"` | String | no       | Prediction column name. |

`LinearRegression` needs parameters above and also below.

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
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.regression.linearregression.LinearRegression;
import org.apache.flink.ml.regression.linearregression.LinearRegressionModel;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a LinearRegression model and uses it for regression. */
public class LinearRegressionExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(Vectors.dense(2, 1), 4.0, 1.0),
                        Row.of(Vectors.dense(3, 2), 7.0, 1.0),
                        Row.of(Vectors.dense(4, 3), 10.0, 1.0),
                        Row.of(Vectors.dense(2, 4), 10.0, 1.0),
                        Row.of(Vectors.dense(2, 2), 6.0, 1.0),
                        Row.of(Vectors.dense(4, 3), 10.0, 1.0),
                        Row.of(Vectors.dense(1, 2), 5.0, 1.0),
                        Row.of(Vectors.dense(5, 3), 11.0, 1.0));
        Table inputTable = tEnv.fromDataStream(inputStream).as("features", "label", "weight");

        // Creates a LinearRegression object and initializes its parameters.
        LinearRegression lr = new LinearRegression().setWeightCol("weight");

        // Trains the LinearRegression Model.
        LinearRegressionModel lrModel = lr.fit(inputTable);

        // Uses the LinearRegression Model for predictions.
        Table outputTable = lrModel.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector features = (DenseVector) row.getField(lr.getFeaturesCol());
            double expectedResult = (Double) row.getField(lr.getLabelCol());
            double predictionResult = (Double) row.getField(lr.getPredictionCol());
            System.out.printf(
                    "Features: %s \tExpected Result: %s \tPrediction Result: %s\n",
                    features, expectedResult, predictionResult);
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that trains a LinearRegression model and uses it for
# regression.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.regression.linearregression import LinearRegression
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(2, 1), 4., 1.),
        (Vectors.dense(3, 2), 7., 1.),
        (Vectors.dense(4, 3), 10., 1.),
        (Vectors.dense(2, 4), 10., 1.),
        (Vectors.dense(2, 2), 6., 1.),
        (Vectors.dense(4, 3), 10., 1.),
        (Vectors.dense(1, 2), 5., 1.),
        (Vectors.dense(5, 3), 11., 1.),
    ],
        type_info=Types.ROW_NAMED(
            ['features', 'label', 'weight'],
            [DenseVectorTypeInfo(), Types.DOUBLE(), Types.DOUBLE()])
    ))

# create a linear regression object and initialize its parameters
linear_regression = LinearRegression().set_weight_col('weight')

# train the linear regression model
model = linear_regression.fit(input_table)

# use the linear regression model for predictions
output = model.transform(input_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    features = result[field_names.index(linear_regression.get_features_col())]
    expected_result = result[field_names.index(linear_regression.get_label_col())]
    prediction_result = result[field_names.index(linear_regression.get_prediction_col())]
    print('Features: ' + str(features) + ' \tExpected Result: ' + str(expected_result)
          + ' \tPrediction Result: ' + str(prediction_result))

```

{{< /tab>}}

{{< /tabs>}}
