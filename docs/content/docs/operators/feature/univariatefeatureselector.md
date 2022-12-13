---
title: "Univariate Feature Selector"
weight: 1
type: docs
aliases:
- /operators/feature/univariatefeatureselector.html
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

## Univariate Feature Selector
Univariate Feature Selector is an algorithm that selects features based on 
univariate statistical tests against labels.

Currently, Flink supports three Univariate Feature Selectors: chi-squared, 
ANOVA F-test and F-value. User can choose Univariate Feature Selector by 
setting `featureType` and `labelType`, and Flink will pick the score function
based on the specified `featureType` and `labelType`.

The following combination of `featureType` and `labelType` are supported:

<ul>
    <li>`featureType` `categorical` and `labelType` `categorical`: Flink uses 
        chi-squared, i.e. chi2 in sklearn.
    <li>`featureType` `continuous` and `labelType` `categorical`: Flink uses 
        ANOVA F-test, i.e. f_classif in sklearn.
    <li>`featureType` `continuous` and `labelType` `continuous`: Flink uses 
        F-value, i.e. f_regression in sklearn.
</ul>

Univariate Feature Selector supports different selection modes:

<ul>
    <li>numTopFeatures: chooses a fixed number of top features according to a 
        hypothesis.
    <li>percentile: similar to numTopFeatures but chooses a fraction of all 
        features instead of a fixed number.
    <li>fpr: chooses all features whose p-value are below a threshold, thus 
        controlling the false positive rate of selection.
    <li>fdr: uses the <a href="https://en.wikipedia.org/wiki/False_discovery_rate#
        Benjamini.E2.80.93Hochberg_procedure">Benjamini-Hochberg procedure</a> to 
        choose all features whose false discovery rate is below a threshold.
    <li>fwe: chooses all features whose p-values are below a threshold. The 
        threshold is scaled by 1/numFeatures, thus controlling the family-wise 
        error rate of selection.
</ul>

By default, the selection mode is `numTopFeatures`.

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

Below are the parameters required by `UnivariateFeatureSelectorModel`.

| Key         | Default      | Type   | Required | Description             |
|-------------|--------------|--------|----------|-------------------------|
| featuresCol | `"features"` | String | no       | Features column name.   |
| outputCol   | `"output"`   | String | no       | Output column name.     |

`UnivariateFeatureSelector` needs parameters above and also below.

| Key                | Default            | Type    | Required | Description                                                                                                                                                                                                                                                                                                                              |
| ------------------ | ------------------ | ------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| labelCol           | `"label"`          | String  | no       | Label column name.                                                                                                                                                                                                                                                                                                                       |
| featureType        | `null`             | String  | yes      | The feature type. Supported values: 'categorical', 'continuous'.                                                                                                                                                                                                                                                                         |
| labelType          | `null`             | String  | yes      | The label type. Supported values: 'categorical', 'continuous'.                                                                                                                                                                                                                                                                           |
| selectionMode      | `"numTopFeatures"` | String  | no       | The feature selection mode. Supported values: 'numTopFeatures', 'percentile', 'fpr', 'fdr', 'fwe'.                                                                                                                                                                                                                                       |
| selectionThreshold | `null`             | Number  | no       | The upper bound of the features that selector will select. If not set, it will be replaced with a meaningful value according to different selection modes at runtime. When the mode is numTopFeatures, it will be replaced with 50; when the mode is percentile, it will be replaced with 0.1; otherwise, it will be replaced with 0.05. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.univariatefeatureselector.UnivariateFeatureSelector;
import org.apache.flink.ml.feature.univariatefeatureselector.UnivariateFeatureSelectorModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/**
 * Simple program that trains a {@link UnivariateFeatureSelector} model and uses it for feature
 * selection.
 */
public class UnivariateFeatureSelectorExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(
                        Row.of(Vectors.dense(1.7, 4.4, 7.6, 5.8, 9.6, 2.3), 3.0),
                        Row.of(Vectors.dense(8.8, 7.3, 5.7, 7.3, 2.2, 4.1), 2.0),
                        Row.of(Vectors.dense(1.2, 9.5, 2.5, 3.1, 8.7, 2.5), 1.0),
                        Row.of(Vectors.dense(3.7, 9.2, 6.1, 4.1, 7.5, 3.8), 2.0),
                        Row.of(Vectors.dense(8.9, 5.2, 7.8, 8.3, 5.2, 3.0), 4.0),
                        Row.of(Vectors.dense(7.9, 8.5, 9.2, 4.0, 9.4, 2.1), 4.0));
        Table trainTable = tEnv.fromDataStream(trainStream).as("features", "label");

        // Creates a UnivariateFeatureSelector object and initializes its parameters.
        UnivariateFeatureSelector univariateFeatureSelector =
                new UnivariateFeatureSelector()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setFeatureType("continuous")
                        .setLabelType("categorical")
                        .setSelectionThreshold(1);

        // Trains the UnivariateFeatureSelector model.
        UnivariateFeatureSelectorModel model = univariateFeatureSelector.fit(trainTable);

        // Uses the UnivariateFeatureSelector model for predictions.
        Table outputTable = model.transform(trainTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector inputValue =
                    (DenseVector) row.getField(univariateFeatureSelector.getFeaturesCol());
            DenseVector outputValue =
                    (DenseVector) row.getField(univariateFeatureSelector.getOutputCol());
            System.out.printf("Input Value: %-15s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates a UnivariateFeatureSelector instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.lib.feature.univariatefeatureselector import UnivariateFeatureSelector
from pyflink.table import StreamTableEnvironment

from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo

env = StreamExecutionEnvironment.get_execution_environment()

t_env = StreamTableEnvironment.create(env)

# Generates input training and prediction data.
input_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(1.7, 4.4, 7.6, 5.8, 9.6, 2.3), 3.0,),
        (Vectors.dense(8.8, 7.3, 5.7, 7.3, 2.2, 4.1), 2.0,),
        (Vectors.dense(1.2, 9.5, 2.5, 3.1, 8.7, 2.5), 1.0,),
        (Vectors.dense(3.7, 9.2, 6.1, 4.1, 7.5, 3.8), 2.0,),
        (Vectors.dense(8.9, 5.2, 7.8, 8.3, 5.2, 3.0), 4.0,),
        (Vectors.dense(7.9, 8.5, 9.2, 4.0, 9.4, 2.1), 4.0,),
    ],
        type_info=Types.ROW_NAMED(
            ['features', 'label'],
            [DenseVectorTypeInfo(), Types.FLOAT()])
    ))

# Creates an UnivariateFeatureSelector object and initializes its parameters.
univariate_feature_selector = UnivariateFeatureSelector() \
    .set_features_col('features') \
    .set_label_col('label') \
    .set_feature_type('continuous') \
    .set_label_type('categorical') \
    .set_selection_threshold(1)

# Trains the UnivariateFeatureSelector Model.
model = univariate_feature_selector.fit(input_table)

# Uses the UnivariateFeatureSelector Model for predictions.
output = model.transform(input_table)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_index = field_names.index(univariate_feature_selector.get_features_col())
    output_index = field_names.index(univariate_feature_selector.get_output_col())
    print('Input Value: ' + str(result[input_index]) +
          '\tOutput Value: ' + str(result[output_index]))

```