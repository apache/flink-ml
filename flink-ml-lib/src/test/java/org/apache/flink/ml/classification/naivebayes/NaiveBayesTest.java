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
import org.apache.flink.ml.util.TableUtils;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
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
import java.util.*;

public class NaiveBayesTest {
    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    DataTypes.Field[] fields;
    Row[] inputData;
    Row[] expectedOutput;
    String[] expectedOutputNames;
    AbstractDataType[] expectedOutputTypes;
    String[] reserved;
    String[] feature;
    String[] categorical;
    String label;
    String weight;
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
                Row.of(1., 1, 1., 1.5, 0., 1., "l0", "l1"),
                Row.of(2., 4, 1., 1., 0., 1., "l0", "l0")
        };

        expectedOutputNames = new String[]{
                "weight",
                "f0",
                "f1",
                "f2",
                "f3",
                "f4",
                "label",
                "predict",
        };

        expectedOutputTypes = new AbstractDataType[]{
                DataTypes.DOUBLE(),
                DataTypes.INT(),
                DataTypes.DOUBLE(),
                DataTypes.DOUBLE(),
                DataTypes.DOUBLE(),
                DataTypes.DOUBLE(),
                DataTypes.STRING(),
                DataTypes.STRING()
        };

        feature = new String[] {"f0", "f4"};
        categorical = new String[]{"f0"};
        reserved = new String[]{"weight", "f0", "f1", "f2", "f3", "f4", "label"};
        label = "label";
        weight = "weight";
        predictCol = "predict";
        smoothing = 1.0;
    }

    @Test
    public void testNaiveBayes() {
        errorMessage = "normal test for Naive Bayes";
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testCategoricalDouble() {
        errorMessage = "Naive Bayes should throw exception when feature columns of type double is set to be categorical";
        feature = new String[] {"f0", "f1", "f4"};
        categorical = new String[] {"f0", "f1", "f4"};
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

//      Due to a bug of .window().reduce() this test case is not able to be passed yet
//    @Test
//    public void testStringFeature() {
//        errorMessage = "Naive Bayes should be able to deal with columns that are string typed.";
//        fields = new DataTypes.Field[]{
//                DataTypes.FIELD("weight", DataTypes.DOUBLE()),
//                DataTypes.FIELD("f0", DataTypes.STRING()),
//                DataTypes.FIELD("f1", DataTypes.DOUBLE()),
//                DataTypes.FIELD("f2", DataTypes.DOUBLE()),
//                DataTypes.FIELD("f3", DataTypes.DOUBLE()),
//                DataTypes.FIELD("f4", DataTypes.DOUBLE()),
//                DataTypes.FIELD("label", DataTypes.STRING())
//        };
//        inputData = new Row[] {
//                Row.of(1., "1", 1., 1., 1., 2., "l1"),
//                Row.of(1., "1", 1., 0., 1., 2., "l1"),
//                Row.of(1., "2", 0., 1., 1., 3., "l1"),
//                Row.of(1., "2", 0., 1., 1.5, 2., "l1"),
//                Row.of(2., "3", 1.5, 1., 0.5, 3., "l0"),
//                Row.of(1., "1", 1., 1.5, 0., 1., "l0"),
//                Row.of(2., "4", 1., 1., 0., 1., "l0")
//        };
//        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
//    }

    @Test (expected = Exception.class)
    public void testEmptyFeature() {
        errorMessage = "Naive Bayes should throw exception if feature columns is empty";
        feature = new String[0];
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testDuplicateFeature() {
        errorMessage = "Naive Bayes should throw exception if feature columns duplicate.";
        feature = new String[]{"f0", "f0"};
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testNullFeature() {
        errorMessage = "Naive Bayes should throw exception if feature columns is not set";
        feature = null;
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test
    public void testEmptyCategorical() {
        errorMessage = "Naive Bayes should run without throwing exception if categorical columns is empty";
        categorical = new String[0];
        expectedOutput = new Row[] {
                Row.of(1., 1, 1., 1., 1., 2., "l1", "l1"),
                Row.of(1., 1, 1., 0., 1., 2., "l1", "l1"),
                Row.of(1., 2, 0., 1., 1., 3., "l1", "l1"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1", "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0", "l1"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0", "l1"),
                Row.of(2., 4, 1., 1., 0., 1., "l0", "l1")
        };
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test
    public void testNullCategorical() {
        errorMessage = "Naive Bayes should run without throwing exception if categorical columns is null";
        categorical = null;
        expectedOutput = new Row[] {
                Row.of(1., 1, 1., 1., 1., 2., "l1", "l1"),
                Row.of(1., 1, 1., 0., 1., 2., "l1", "l1"),
                Row.of(1., 2, 0., 1., 1., 3., "l1", "l1"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1", "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0", "l1"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0", "l1"),
                Row.of(2., 4, 1., 1., 0., 1., "l0", "l1")
        };
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testDuplicateCategorical() {
        errorMessage = "Naive Bayes should throw exception if categorical columns duplicate.";
        categorical = new String[]{"f0", "f0"};
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testInputNotContainFeature() {
        errorMessage = "Naive Bayes should throw exception if some feature columns are missing from train data";
        feature = new String[]{"fa", "fb", "f0"};
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testInputNotContainAllFeature() {
        errorMessage = "Naive Bayes should throw exception if all feature columns are missing from train data";
        feature = new String[]{"fa", "fb", "fc"};
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
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
                Row.of(1., 5, 1., 1., 1., 2., "l1", "l0"),
                Row.of(1., 5, 1., 0., 1., 2., "l1", "l0"),
                Row.of(1., 2, 0., 1., 1., 3., "l1", "l1"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1", "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0", "l0"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0", "l1"),
                Row.of(2., 4, 1., 1., 0., 1., "l0", "l0")
        };
        runAndCheck(tEnv, fields, inputData, predictData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testFeatureNotContainCategorical() {
        errorMessage = "Naive Bayes should throw exception if features do not contain columns listed in categorical";
        feature = new String[] {"f0"};
        categorical = new String[] {"f0", "f4"};
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testEmptyLabel() {
        errorMessage = "Naive Bayes should throw exception if label is empty";
        feature = new String[0];
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testNullLabel() {
        errorMessage = "Naive Bayes should throw exception if label is empty";
        feature = null;
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
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
                Row.of(1., 1, 1., 1., 1., 2., null, "l1"),
                Row.of(1., 1, 1., 0., 1., 2., null, "l1"),
                Row.of(1., 2, 0., 1., 1., 3., null, "l1"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1", "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0", "l0"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0", "l1"),
                Row.of(2., 4, 1., 1., 0., 1., "l0", "l1")
        };
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testInputNotContainLabel() {
        errorMessage = "Naive Bayes should throw exception if input table schema does not contain label column.";
        label = "non-label";
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test
    public void testEmptyWeight() {
        errorMessage = "Naive Bayes should not throw exception if weight is set to be empty string.";
        weight = "";
        expectedOutput = new Row[] {
                Row.of(1., 1, 1., 1., 1., 2., "l1", "l0"),
                Row.of(1., 1, 1., 0., 1., 2., "l1", "l0"),
                Row.of(1., 2, 0., 1., 1., 3., "l1", "l1"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1", "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0", "l0"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0", "l0"),
                Row.of(2., 4, 1., 1., 0., 1., "l0", "l0")
        };
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test
    public void testNullWeight() {
        errorMessage = "Naive Bayes should run without throwing exception if weight is not set.";
        weight = null;
        expectedOutput = new Row[] {
                Row.of(1., 1, 1., 1., 1., 2., "l1", "l0"),
                Row.of(1., 1, 1., 0., 1., 2., "l1", "l0"),
                Row.of(1., 2, 0., 1., 1., 3., "l1", "l1"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1", "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0", "l0"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0", "l0"),
                Row.of(2., 4, 1., 1., 0., 1., "l0", "l0")
        };
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testNullPredict() {
        errorMessage = "Naive Bayes should throw exception if predict col is not set.";
        predictCol = null;
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, feature, categorical, reserved, label, weight, predictCol, smoothing, errorMessage);
    }

    private static void runAndCheck(
            StreamTableEnvironment tEnv,
            DataTypes.Field[] inputType,
            Row[] inputData,
            Row[] expected,
            String[] expectedOutputNames,
            AbstractDataType[] expectedOutputTypes,
            String[] featureCols,
            String[] categoricalCols,
            String[] reservedCols,
            String label,
            String weightCol,
            String predictCol,
            double smoothing,
            String errorMessage
    ) {
        runAndCheck(tEnv, inputType, inputData, inputData, expected, expectedOutputNames, expectedOutputTypes, featureCols, categoricalCols, reservedCols, label, weightCol, predictCol, smoothing, errorMessage);
    }

    private static void runAndCheck(
            StreamTableEnvironment tEnv,
            DataTypes.Field[] inputType,
            Row[] trainData,
            Row[] predictData,
            Row[] expected,
            String[] expectedOutputNames,
            AbstractDataType[] expectedOutputTypes,
            String[] featureCols,
            String[] categoricalCols,
            String[] reservedCols,
            String label,
            String weightCol,
            String predictCol,
            double smoothing,
            String errorMessage
    ) {
        Schema.Builder builder = Schema.newBuilder();
        builder.fromFields(expectedOutputNames, expectedOutputTypes);
        Schema expectedSchema = builder.build();
        
        Table trainTable = tEnv.fromValues(DataTypes.ROW(inputType), (Object[]) trainData);
        Table predictTable = tEnv.fromValues(DataTypes.ROW(inputType), (Object[]) predictData);

        NaiveBayes trainer = new NaiveBayes();
        trainer.setSmoothing(smoothing);
        if (featureCols != null)        trainer.setFeatureCols(featureCols);
        if (categoricalCols != null)    trainer.setCategoricalCols(categoricalCols);
        if (weightCol != null)          trainer.setWeightCol(weightCol);
        if (label != null)              trainer.setLabelCol(label);
        if (reservedCols != null)       trainer.setReservedCols(reservedCols);
        if (predictCol != null)         trainer.setPredictionCol(predictCol);

        Table output = trainer.fit(trainTable).transform(predictTable)[0];
        Assert.assertEquals(errorMessage, expectedSchema, TableUtils.toSchema(output.getResolvedSchema()));

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
