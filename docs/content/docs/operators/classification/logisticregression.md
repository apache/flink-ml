---
title: "Logistic Regression"
type: docs
aliases:
- /operators/classification/logisticregression.html
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

## Logistic Regression

Logistic regression is a special case of the Generalized Linear Model. It is
widely used to predict a binary response. 

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

Below are the parameters required by `LogisticRegressionModel`.

| Key              | Default           | Type   | Required | Description                 |
| ---------------- | ----------------- | ------ | -------- | --------------------------- |
| featuresCol      | `"features"`      | String | no       | Features column name.       |
| predictionCol    | `"prediction"`    | String | no       | Prediction column name.     |
| rawPredictionCol | `"rawPrediction"` | String | no       | Raw prediction column name. |

`LogisticRegression` needs parameters above and also below.

| Key             | Default   | Type    | Required | Description                                                               |
|-----------------|-----------|---------|----------|---------------------------------------------------------------------------|
| labelCol        | `"label"` | String  | no       | Label column name.                                                        |
| weightCol       | `null`    | String  | no       | Weight column name.                                                       |
| maxIter         | `20`      | Integer | no       | Maximum number of iterations.                                             |
| reg             | `0.`      | Double  | no       | Regularization parameter.                                                 |
| elasticNet      | `0.`      | Double  | no       | ElasticNet parameter.                                                     |
| learningRate    | `0.1`     | Double  | no       | Learning rate of optimization method.                                     |
| globalBatchSize | `32`      | Integer | no       | Global batch size of training algorithms.                                 |
| tol             | `1e-6`    | Double  | no       | Convergence tolerance for iterative algorithms.                           |
| multiClass      | `"auto"`  | String  | no       | Classification type. Supported values: "auto", "binomial", "multinomial". |

### Examples
{{< tabs examples >}}

{{< tab "Java">}}
```java
import org.apache.flink.ml.classification.logisticregression.LogisticRegression;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a LogisticRegression model and uses it for classification. */
public class LogisticRegressionExample {
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

        // Creates a LogisticRegression object and initializes its parameters.
        LogisticRegression lr = new LogisticRegression().setWeightCol("weight");

        // Trains the LogisticRegression Model.
        LogisticRegressionModel lrModel = lr.fit(inputTable);

        // Uses the LogisticRegression Model for predictions.
        Table outputTable = lrModel.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector features = (DenseVector) row.getField(lr.getFeaturesCol());
            double expectedResult = (Double) row.getField(lr.getLabelCol());
            double predictionResult = (Double) row.getField(lr.getPredictionCol());
            DenseVector rawPredictionResult = (DenseVector) row.getField(lr.getRawPredictionCol());
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
# Simple program that trains a LogisticRegression model and uses it for
# classification.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.classification.logisticregression import LogisticRegression
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data = t_env.from_data_stream(
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

# create a logistic regression object and initialize its parameters
logistic_regression = LogisticRegression().set_weight_col('weight')

# train the logistic regression model
model = logistic_regression.fit(input_data)

# use the logistic regression model for predictions
output = model.transform(input_data)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    features = result[field_names.index(logistic_regression.get_features_col())]
    expected_result = result[field_names.index(logistic_regression.get_label_col())]
    prediction_result = result[field_names.index(logistic_regression.get_prediction_col())]
    raw_prediction_result = result[field_names.index(logistic_regression.get_raw_prediction_col())]
    print('Features: ' + str(features) + ' \tExpected Result: ' + str(expected_result)
          + ' \tPrediction Result: ' + str(prediction_result)
          + ' \tRaw Prediction Result: ' + str(raw_prediction_result))

```
{{< /tab>}}

{{< /tabs>}}

## OnlineLogisticRegression

Online Logistic Regression supports training online regression model on an
unbounded stream of training data. 

The online optimizer of this algorithm is The FTRL-Proximal proposed by
H.Brendan McMahan et al. See [H. Brendan McMahan et al., Ad click prediction: a
view from the trenches.](https://doi.org/10.1145/2487575.2488200)

### Input Columns

| Param name  | Type    | Default      | Description      |
| :---------- | :------ | :----------- | :--------------- |
| featuresCol | Vector  | `"features"` | Feature vector   |
| labelCol    | Integer | `"label"`    | Label to predict |
| weightCol   | Double  | `"weight"`   | Weight of sample |

### Output Columns

| Param name       | Type    | Default           | Description                                            |
| :--------------- | :------ | :---------------- | :----------------------------------------------------- |
| predictionCol    | Integer | `"prediction"`    | Label of the max probability                           |
| rawPredictionCol | Vector  | `"rawPrediction"` | Vector of the probability of each label                |
| modelVersionCol  | Long    | `"modelVersion"`  | The version of the model data used for this prediction |

### Parameters

Below are the parameters required by `OnlineLogisticRegressionModel`.

| Key              | Default           | Type   | Required | Description                 |
| ---------------- | ----------------- | ------ | -------- | --------------------------- |
| featuresCol      | `"features"`      | String | no       | Features column name.       |
| predictionCol    | `"prediction"`    | String | no       | Prediction column name.     |
| rawPredictionCol | `"rawPrediction"` | String | no       | Raw prediction column name. |
| modelVersionCol  | `"modelVersion"`  | String | no       | Model version column name.  |

`OnlineLogisticRegression` needs parameters above and also below.

| Key             | Default          | Type    | Required | Description                                           |
| --------------- | ---------------- | ------- | -------- | ----------------------------------------------------- |
| labelCol        | `"label"`        | String  | no       | Label column name.                                    |
| weightCol       | `null`           | String  | no       | Weight column name.                                   |
| batchStrategy   | `COUNT_STRATEGY` | String  | no       | Strategy to create mini batch from online train data. |
| globalBatchSize | `32`             | Integer | no       | Global batch size of training algorithms.             |
| reg             | `0.`             | Double  | no       | Regularization parameter.                             |
| elasticNet      | `0.`             | Double  | no       | ElasticNet parameter.                                 |

### Examples

{{< tabs online_examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegression;
import org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegressionModel;
import org.apache.flink.ml.examples.util.PeriodicSourceFunction;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** Simple program that trains an OnlineLogisticRegression model and uses it for classification. */
public class OnlineLogisticRegressionExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data. Both are infinite streams that periodically
        // sends out provided data to trigger model update and prediction.
        List<Row> trainData1 =
                Arrays.asList(
                        Row.of(Vectors.dense(0.1, 2.), 0.),
                        Row.of(Vectors.dense(0.2, 2.), 0.),
                        Row.of(Vectors.dense(0.3, 2.), 0.),
                        Row.of(Vectors.dense(0.4, 2.), 0.),
                        Row.of(Vectors.dense(0.5, 2.), 0.),
                        Row.of(Vectors.dense(11., 12.), 1.),
                        Row.of(Vectors.dense(12., 11.), 1.),
                        Row.of(Vectors.dense(13., 12.), 1.),
                        Row.of(Vectors.dense(14., 12.), 1.),
                        Row.of(Vectors.dense(15., 12.), 1.));

        List<Row> trainData2 =
                Arrays.asList(
                        Row.of(Vectors.dense(0.2, 3.), 0.),
                        Row.of(Vectors.dense(0.8, 1.), 0.),
                        Row.of(Vectors.dense(0.7, 1.), 0.),
                        Row.of(Vectors.dense(0.6, 2.), 0.),
                        Row.of(Vectors.dense(0.2, 2.), 0.),
                        Row.of(Vectors.dense(14., 17.), 1.),
                        Row.of(Vectors.dense(15., 10.), 1.),
                        Row.of(Vectors.dense(16., 16.), 1.),
                        Row.of(Vectors.dense(17., 10.), 1.),
                        Row.of(Vectors.dense(18., 13.), 1.));

        List<Row> predictData =
                Arrays.asList(
                        Row.of(Vectors.dense(0.8, 2.7), 0.0),
                        Row.of(Vectors.dense(15.5, 11.2), 1.0));

        RowTypeInfo typeInfo =
                new RowTypeInfo(
                        new TypeInformation[] {DenseVectorTypeInfo.INSTANCE, Types.DOUBLE},
                        new String[] {"features", "label"});

        SourceFunction<Row> trainSource =
                new PeriodicSourceFunction(1000, Arrays.asList(trainData1, trainData2));
        DataStream<Row> trainStream = env.addSource(trainSource, typeInfo);
        Table trainTable = tEnv.fromDataStream(trainStream).as("features");

        SourceFunction<Row> predictSource =
                new PeriodicSourceFunction(1000, Collections.singletonList(predictData));
        DataStream<Row> predictStream = env.addSource(predictSource, typeInfo);
        Table predictTable = tEnv.fromDataStream(predictStream).as("features");

        // Creates an online LogisticRegression object and initializes its parameters and initial
        // model data.
        Row initModelData = Row.of(Vectors.dense(0.41233679404769874, -0.18088118293232122), 0L);
        Table initModelDataTable = tEnv.fromDataStream(env.fromElements(initModelData));
        OnlineLogisticRegression olr =
                new OnlineLogisticRegression()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(10)
                        .setInitialModelData(initModelDataTable);

        // Trains the online LogisticRegression Model.
        OnlineLogisticRegressionModel onlineModel = olr.fit(trainTable);

        // Uses the online LogisticRegression Model for predictions.
        Table outputTable = onlineModel.transform(predictTable)[0];

        // Extracts and displays the results. As training data stream continuously triggers the
        // update of the internal model data, raw prediction results of the same predict dataset
        // would change over time.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector features = (DenseVector) row.getField(olr.getFeaturesCol());
            Double expectedResult = (Double) row.getField(olr.getLabelCol());
            Double predictionResult = (Double) row.getField(olr.getPredictionCol());
            DenseVector rawPredictionResult = (DenseVector) row.getField(olr.getRawPredictionCol());
            System.out.printf(
                    "Features: %-25s \tExpected Result: %s \tPrediction Result: %s \tRaw Prediction Result: %s\n",
                    features, expectedResult, predictionResult, rawPredictionResult);
        }
    }
}

```

{{< /tab>}}

{{< /tabs>}}
