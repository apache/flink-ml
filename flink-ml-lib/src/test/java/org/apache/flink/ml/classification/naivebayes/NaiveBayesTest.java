/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.classification.naivebayes;

import org.apache.commons.collections.IteratorUtils;
import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.iteration.config.IterationOptions;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.types.AbstractDataType;
import org.apache.flink.types.Row;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NaiveBayesTest {
    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    DataTypes.Field[] fields;
    Row[] inputData;
    Row[] expectedOutput;
    String[] feature;
    String label;
    String predictCol;
    double smoothing;
    String errorMessage;

    @Before
    public void Setup() throws IOException {
        env = StreamExecutionEnvironment.createLocalEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        env.setRuntimeMode(RuntimeExecutionMode.BATCH);
        Configuration configuration = new Configuration();
        configuration.set(RestOptions.PORT, 18082);
        configuration.set(
                IterationOptions.DATA_CACHE_PATH,
                "file://" + tempFolder.newFolder().getAbsolutePath());
        configuration.set(
                ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env.getConfig().setGlobalJobParameters(configuration);

        fields = new DataTypes.Field[]{
                DataTypes.FIELD("weight", DataTypes.DOUBLE()),
                DataTypes.FIELD("f0", DataTypes.INT()),
                DataTypes.FIELD("f1", DataTypes.DOUBLE()),
                DataTypes.FIELD("f2", DataTypes.DOUBLE()),
                DataTypes.FIELD("f3", DataTypes.DOUBLE()),
                DataTypes.FIELD("f4", DataTypes.DOUBLE()),
                DataTypes.FIELD("label", DataTypes.STRING())
        };
        
        inputData = new Row[] {
                Row.of(1., 1, 1., 1., 1., 2., "l1"),
                Row.of(1., 1, 1., 0., 1., 2., "l1"),
                Row.of(1., 2, 0., 1., 1., 3., "l1"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0"),
                Row.of(2., 4, 1., 1., 0., 1., "l0")
        };

        expectedOutput = new Row[] {
                Row.of(1., 1, 1., 1., 1., 2., "l1", "l1"),
                Row.of(1., 1, 1., 0., 1., 2., "l1", "l1"),
                Row.of(1., 2, 0., 1., 1., 3., "l1", "l1"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1", "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0", "l0"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0", "l0"),
                Row.of(2., 4, 1., 1., 0., 1., "l0", "l0")
        };

        feature = new String[] {"f0", "f4"};
        label = "label";
        predictCol = "predict";
        smoothing = 1.0;
    }

    @Test
    public void testNaiveBayes() {
        errorMessage = "normal test for Naive Bayes";
        runAndCheck(tEnv, fields, inputData, expectedOutput, feature, label, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testEmptyFeature() {
        errorMessage = "Naive Bayes should throw exception if feature columns is empty";
        feature = new String[0];
        runAndCheck(tEnv, fields, inputData, expectedOutput, feature, label, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testDuplicateFeature() {
        errorMessage = "Naive Bayes should throw exception if feature columns duplicate.";
        feature = new String[]{"f0", "f0"};
        runAndCheck(tEnv, fields, inputData, expectedOutput, feature, label, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testNullFeature() {
        errorMessage = "Naive Bayes should throw exception if feature columns is not set";
        feature = null;
        runAndCheck(tEnv, fields, inputData, expectedOutput, feature, label, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testInputNotContainFeature() {
        errorMessage = "Naive Bayes should throw exception if some feature columns are missing from train data";
        feature = new String[]{"fa", "fb", "f0"};
        runAndCheck(tEnv, fields, inputData, expectedOutput, feature, label, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testInputNotContainAllFeature() {
        errorMessage = "Naive Bayes should throw exception if all feature columns are missing from train data";
        feature = new String[]{"fa", "fb", "fc"};
        runAndCheck(tEnv, fields, inputData, expectedOutput, feature, label, predictCol, smoothing, errorMessage);
    }

    @Test
    public void testPredictUnseenFeature() {
        errorMessage = "Naive Bayes should run without throwing exception if unseen feature values are met in prediction";
        Row[] predictData = new Row[] {
                Row.of(1., 5, 1., 1., 1., 2., "l1"),
                Row.of(1., 5, 1., 0., 1., 2., "l1"),
                Row.of(1., 2, 0., 1., 1., 3., "l1"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0"),
                Row.of(2., 4, 1., 1., 0., 1., "l0")
        };
        expectedOutput = new Row[] {
                Row.of(1., 5, 1., 1., 1., 2., "l1", "l1"),
                Row.of(1., 5, 1., 0., 1., 2., "l1", "l1"),
                Row.of(1., 2, 0., 1., 1., 3., "l1", "l1"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1", "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0", "l0"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0", "l0"),
                Row.of(2., 4, 1., 1., 0., 1., "l0", "l0")
        };
        runAndCheck(tEnv, fields, inputData, predictData, expectedOutput, feature, label, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testEmptyLabel() {
        errorMessage = "Naive Bayes should throw exception if label is empty";
        feature = new String[0];
        runAndCheck(tEnv, fields, inputData, expectedOutput, feature, label, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testNullLabel() {
        errorMessage = "Naive Bayes should throw exception if label is empty";
        feature = null;
        runAndCheck(tEnv, fields, inputData, expectedOutput, feature, label, predictCol, smoothing, errorMessage);
    }

    @Test
    public void testNullLabelValue() {
        errorMessage = "Naive Bayes should ignore input rows that does not contain label";
        inputData = new Row[] {
                Row.of(1., 1., 1., 1., 1., 2., null),
                Row.of(1., 1., 1., 0., 1., 2., null),
                Row.of(1., 2., 0., 1., 1., 3., null),
                Row.of(1., 2., 0., 1., 1.5, 2., "l1"),
                Row.of(2., 3., 1.5, 1., 0.5, 3., "l0"),
                Row.of(1., 1., 1., 1.5, 0., 1., "l0"),
                Row.of(2., 4., 1., 1., 0., 1., "l0")
        };

        expectedOutput = new Row[] {
                Row.of(1., 1, 1., 1., 1., 2., null, "l0"),
                Row.of(1., 1, 1., 0., 1., 2., null, "l0"),
                Row.of(1., 2, 0., 1., 1., 3., null, "l0"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1", "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0", "l0"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0", "l0"),
                Row.of(2., 4, 1., 1., 0., 1., "l0", "l0")
        };
        runAndCheck(tEnv, fields, inputData, expectedOutput, feature, label, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testInputNotContainLabel() {
        errorMessage = "Naive Bayes should throw exception if input table schema does not contain label column.";
        label = "non-label";
        runAndCheck(tEnv, fields, inputData, expectedOutput, feature, label, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testNullPredict() {
        errorMessage = "Naive Bayes should throw exception if predict col is not set.";
        predictCol = null;
        runAndCheck(tEnv, fields, inputData, expectedOutput, feature, label, predictCol, smoothing, errorMessage);
    }

    private static void runAndCheck(
            StreamTableEnvironment tEnv,
            DataTypes.Field[] inputType,
            Row[] inputData,
            Row[] expected,
            String[] featureCols,
            String label,
            String predictCol,
            double smoothing,
            String errorMessage
    ) {
        runAndCheck(tEnv, inputType, inputData, inputData, expected, featureCols, label, predictCol, smoothing, errorMessage);
    }

    private static void runAndCheck(
            StreamTableEnvironment tEnv,
            DataTypes.Field[] inputType,
            Row[] trainData,
            Row[] predictData,
            Row[] expected,
            String[] featureCols,
            String label,
            String predictCol,
            double smoothing,
            String errorMessage
    ) {
        
        Table trainTable = tEnv.fromValues(DataTypes.ROW(inputType), (Object[]) trainData);
        Table predictTable = tEnv.fromValues(DataTypes.ROW(inputType), (Object[]) predictData);

        NaiveBayes trainer = new NaiveBayes();
        trainer.setSmoothing(smoothing);
        if (featureCols != null)        trainer.setFeatureCols(featureCols);
        if (label != null)              trainer.setLabelCol(label);
        if (predictCol != null)         trainer.setPredictionCol(predictCol);

        Table output = trainer.fit(trainTable).transform(predictTable)[0];

        Object[] actualObjects = IteratorUtils.toArray(output.execute().collect());
        Row[] actual = new Row[actualObjects.length];
        for (int i=0; i<actualObjects.length;i++) {
            actual[i] = (Row) actualObjects[i];
        }
        
        Assert.assertEquals(errorMessage, getFrequencyMap(expected), getFrequencyMap(actual));
    }

    private static Map<Object, Integer> getFrequencyMap(Row[] rows) {
        Map<Object, Integer> map = new HashMap<>();
        for (Row row: rows) {
            List<Object> list = toList(row);
            map.put(list, map.getOrDefault(list, 0) + 1);
        }
        return map;
    }

    private static List<Object> toList(Row row) {
        List<Object> list = new ArrayList<>();
        for (int i = 0; i < row.getArity(); i++) {
            list.add(row.getField(i));
        }
        return list;
    }
}
