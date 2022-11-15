---
title: "ChiSqSelector"
weight: 1
type: docs
aliases:
- /operators/feature/chisqselector.html
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

## ChiSqSelector

ChiSqSelector is an algorithm that selects categorical features to use for
predicting a categorical label.

The selector supports different selection methods as follows.

- `numTopFeatures` chooses a fixed number of top features according to a
  chi-squared test.
- `percentile` is similar but chooses a fraction of all features instead of a
  fixed number.
- `fpr` chooses all features whose p-value are below a threshold, thus
      controlling the false positive rate of selection.
- `fdr` uses the [Benjamini-Hochberg
      procedure](https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini.E2.80.93Hochberg_procedure)
      to choose all features whose false discovery rate is below a threshold.
- `fwe` chooses all features whose p-values are below a threshold. The threshold
      is scaled by 1/numFeatures, thus controlling the family-wise error rate of
      selection.

By default, the selection method is `numTopFeatures`, with the default number of
top features set to 50.

### Input Columns

| Param name  | Type   | Default      | Description            |
|:------------|:-------|:-------------|:-----------------------|
| featuresCol | Vector | `"features"` | Feature vector.        |
| labelCol    | Number | `"label"`    | Label of the features. |

### Output Columns

| Param name | Type   | Default    | Description        |
|:-----------|:-------|:-----------|:-------------------|
| outputCol  | Vector | `"output"` | Selected features. |

### Parameters

Below are the parameters required by `ChiSqSelectorModel`.

| Key         | Default      | Type   | Required | Description           |
|-------------|--------------|--------|----------|-----------------------|
| featuresCol | `"features"` | String | no       | Features column name. |
| outputCol   | `"output"`   | String | no       | Output column name.   |

`ChiSqSelector` needs parameters above and also below.

| Key            | Default            | Type    | Required | Description                                                                                                                                                    |
|----------------|--------------------|---------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| labelCol       | `"label"`          | String  | no       | Label column name.                                                                                                                                             |
| selectorType   | `"numTopFeatures"` | String  | no       | The selector type. Supported options: numTopFeatures, percentile, fpr, fdr, fwe.                                                                               |
| numTopFeatures | `50`               | Integer | no       | Number of features that selector will select, ordered by ascending p-value. If the number of features is < numTopFeatures, then this will select all features. |
| percentile     | `0.1`              | Double  | no       | Percentile of features that selector will select, ordered by ascending p-value.                                                                                |
| fpr            | `0.05`             | Double  | no       | The highest p-value for features to be kept.                                                                                                                   |
| fdr            | `0.05`             | Double  | no       | The upper bound of the expected false discovery rate.                                                                                                          |
| fwe            | `0.05`             | Double  | no       | The upper bound of the expected family-wise error rate.                                                                                                        |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.feature.chisqselector.ChiSqSelector;
import org.apache.flink.ml.feature.chisqselector.ChiSqSelectorModel;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

import static org.apache.flink.ml.feature.chisqselector.ChiSqSelectorParams.NUM_TOP_FEATURES_TYPE;

/** Simple program that trains a ChiSqSelector model and uses it for feature engineering. */
public class ChiSqSelectorExample {
      public static void main(String[] args) {
            StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
            StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

            // Generates input data.
            DataStream<Row> inputStream =
                    env.fromCollection(
                            Arrays.asList(
                                    Row.of(
                                            0.0,
                                            Vectors.sparse(
                                                    6,
                                                    new int[] {0, 1, 3, 4},
                                                    new double[] {6.0, 7.0, 7.0, 6.0})),
                                    Row.of(
                                            1.0,
                                            Vectors.sparse(
                                                    6,
                                                    new int[] {1, 2, 4, 5},
                                                    new double[] {9.0, 6.0, 5.0, 9.0})),
                                    Row.of(
                                            1.0,
                                            Vectors.sparse(
                                                    6,
                                                    new int[] {1, 2, 4, 5},
                                                    new double[] {9.0, 3.0, 5.0, 5.0})),
                                    Row.of(1.0, Vectors.dense(0.0, 9.0, 8.0, 5.0, 6.0, 4.0)),
                                    Row.of(2.0, Vectors.dense(8.0, 9.0, 6.0, 5.0, 4.0, 4.0)),
                                    Row.of(2.0, Vectors.dense(8.0, 9.0, 6.0, 4.0, 0.0, 0.0))),
                            new RowTypeInfo(Types.DOUBLE, VectorTypeInfo.INSTANCE));
            Table inputTable = tEnv.fromDataStream(inputStream).as("label", "features");

            // Creates a ChiSqSelector object and initializes its parameters.
            ChiSqSelector selector =
                    new ChiSqSelector()
                            .setFeaturesCol("features")
                            .setLabelCol("label")
                            .setOutputCol("prediction")
                            .setSelectorType(NUM_TOP_FEATURES_TYPE)
                            .setNumTopFeatures(1);

            // Trains the ChiSqSelector Model.
            ChiSqSelectorModel model = selector.fit(inputTable);

            // Uses the ChiSqSelector Model for predictions.
            Table outputTable = model.transform(inputTable)[0];

            // Extracts and displays the results.
            for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
                  Row row = it.next();
                  Vector inputValue = row.getFieldAs("features");
                  Vector outputValue = row.getFieldAs("prediction");
                  System.out.printf("Input Value: %s \tOutput Value: %s\n", inputValue, outputValue);
            }
      }
}
```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that trains a ChiSqSelector model and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.core.linalg import Vectors, VectorTypeInfo
from pyflink.ml.lib.feature.chisqselector import ChiSqSelector
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_table = t_env.from_data_stream(
      env.from_collection([
            (0.0, Vectors.sparse(6, [0, 1, 3, 4], [6.0, 7.0, 7.0, 6.0]),),
            (0.0, Vectors.sparse(6, [1, 2, 4, 5], [9.0, 6.0, 5.0, 9.0]),),
            (0.0, Vectors.sparse(6, [1, 2, 4, 5], [9.0, 3.0, 5.0, 5.0]),),
            (1.0, Vectors.dense(0.0, 9.0, 8.0, 5.0, 6.0, 4.0),),
            (2.0, Vectors.dense(8.0, 9.0, 6.0, 5.0, 4.0, 4.0),),
            (2.0, Vectors.dense(8.0, 9.0, 6.0, 4.0, 0.0, 0.0),),
      ],
            type_info=Types.ROW_NAMED(
                  ['label', 'features'],
                  [Types.DOUBLE(), VectorTypeInfo()])))

# create a ChiSqSelector object and initialize its parameters
selector = ChiSqSelector()

# train the ChiSqSelector model
model = selector.fit(input_table)

# use the ChiSqSelector model for predictions
output_table = model.transform(input_table)[0]

# extract and display the results
field_names = output_table.get_schema().get_field_names()
for result in t_env.to_data_stream(output_table).execute_and_collect():
      input_value = result[field_names.index(selector.get_features_col())]
      output_value = result[field_names.index(selector.get_output_col())]
      print('Input Value: ' + str(input_value) + ' \tOutput Value: ' + str(output_value))
```

{{< /tab>}}

{{< /tabs>}}
