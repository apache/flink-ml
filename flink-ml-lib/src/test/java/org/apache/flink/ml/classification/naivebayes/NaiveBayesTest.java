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
import org.apache.flink.types.Row;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NaiveBayesTest {
    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    DataTypes.Field[] inputType;
    Row[] trainData;
    Row[] predictData;
    Row[] expectedOutput;
    String[] featureCols;
    String labelCol;
    String predictCol;
    double smoothing;
    boolean isSaveLoad;
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

        inputType = new DataTypes.Field[]{
                DataTypes.FIELD("weight", DataTypes.DOUBLE()),
                DataTypes.FIELD("f0", DataTypes.INT()),
                DataTypes.FIELD("f1", DataTypes.DOUBLE()),
                DataTypes.FIELD("f2", DataTypes.DOUBLE()),
                DataTypes.FIELD("f3", DataTypes.DOUBLE()),
                DataTypes.FIELD("f4", DataTypes.DOUBLE()),
                DataTypes.FIELD("label", DataTypes.STRING())
        };
        
        trainData = new Row[] {
                Row.of(1., 1, 1., 1., 1., 2., "l1"),
                Row.of(1., 1, 1., 0., 1., 2., "l1"),
                Row.of(1., 2, 0., 1., 1., 3., "l1"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0"),
                Row.of(2., 4, 1., 1., 0., 1., "l0")
        };

        predictData = trainData;

        expectedOutput = new Row[] {
                Row.of(1., 1, 1., 1., 1., 2., "l1", "l1"),
                Row.of(1., 1, 1., 0., 1., 2., "l1", "l1"),
                Row.of(1., 2, 0., 1., 1., 3., "l1", "l1"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1", "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0", "l0"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0", "l0"),
                Row.of(2., 4, 1., 1., 0., 1., "l0", "l0")
        };

        featureCols = new String[] {"f0", "f4"};
        labelCol = "label";
        predictCol = "predict";
        smoothing = 1.0;
        isSaveLoad = false;
    }

    @Test
    public void testNaiveBayes() throws Exception {
        errorMessage = "normal test for Naive Bayes";
        runAndCheck();
    }

    @Test (expected = Exception.class)
    public void testEmptyFeature() throws Exception {
        errorMessage = "Naive Bayes should throw exception if feature columns is empty";
        featureCols = new String[0];
        runAndCheck();
    }

    @Test (expected = Exception.class)
    public void testDuplicateFeature() throws Exception {
        errorMessage = "Naive Bayes should throw exception if feature columns duplicate.";
        featureCols = new String[]{"f0", "f0"};
        runAndCheck();
    }

    @Test (expected = Exception.class)
    public void testNullFeature() throws Exception {
        errorMessage = "Naive Bayes should throw exception if feature columns is not set";
        featureCols = null;
        runAndCheck();
    }

    @Test (expected = Exception.class)
    public void testInputNotContainFeature() throws Exception {
        errorMessage = "Naive Bayes should throw exception if some feature columns are missing from train data";
        featureCols = new String[]{"fa", "fb", "f0"};
        runAndCheck();
    }

    @Test (expected = Exception.class)
    public void testInputNotContainAllFeature() throws Exception {
        errorMessage = "Naive Bayes should throw exception if all feature columns are missing from train data";
        featureCols = new String[]{"fa", "fb", "fc"};
        runAndCheck();
    }

    @Test
    public void testPredictUnseenFeature() throws Exception {
        errorMessage = "Naive Bayes should run without throwing exception if unseen feature values are met in prediction";
        predictData = new Row[] {
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
        runAndCheck();
    }

    @Test (expected = Exception.class)
    public void testEmptyLabel() throws Exception {
        errorMessage = "Naive Bayes should throw exception if label is empty";
        featureCols = new String[0];
        runAndCheck();
    }

    @Test (expected = Exception.class)
    public void testNullLabel() throws Exception {
        errorMessage = "Naive Bayes should throw exception if label is empty";
        featureCols = null;
        runAndCheck();
    }

    @Test
    public void testNullLabelValue() throws Exception {
        errorMessage = "Naive Bayes should ignore input rows that does not contain label";
        trainData = new Row[] {
                Row.of(1., 1., 1., 1., 1., 2., null),
                Row.of(1., 1., 1., 0., 1., 2., null),
                Row.of(1., 2., 0., 1., 1., 3., null),
                Row.of(1., 2., 0., 1., 1.5, 2., "l1"),
                Row.of(2., 3., 1.5, 1., 0.5, 3., "l0"),
                Row.of(1., 1., 1., 1.5, 0., 1., "l0"),
                Row.of(2., 4., 1., 1., 0., 1., "l0")
        };

        predictData = trainData;

        expectedOutput = new Row[] {
                Row.of(1., 1, 1., 1., 1., 2., null, "l0"),
                Row.of(1., 1, 1., 0., 1., 2., null, "l0"),
                Row.of(1., 2, 0., 1., 1., 3., null, "l0"),
                Row.of(1., 2, 0., 1., 1.5, 2., "l1", "l1"),
                Row.of(2., 3, 1.5, 1., 0.5, 3., "l0", "l0"),
                Row.of(1., 1, 1., 1.5, 0., 1., "l0", "l0"),
                Row.of(2., 4, 1., 1., 0., 1., "l0", "l0")
        };
        runAndCheck();
    }

    @Test (expected = Exception.class)
    public void testInputNotContainLabel() throws Exception {
        errorMessage = "Naive Bayes should throw exception if input table schema does not contain label column.";
        labelCol = "non-label";
        runAndCheck();
    }

    @Test (expected = Exception.class)
    public void testNullPredict() throws Exception {
        errorMessage = "Naive Bayes should throw exception if predict col is not set.";
        predictCol = null;
        runAndCheck();
    }

    @Test (expected = Exception.class)
    public void testSaveLoad() throws Exception {
        errorMessage = "Naive Bayes should be able to save Model to filesystem and load correctly.";
        isSaveLoad = true;
        runAndCheck();
    }

    private void runAndCheck() throws Exception {
        Table trainTable = tEnv.fromValues(DataTypes.ROW(inputType), (Object[]) trainData);
        Table predictTable = tEnv.fromValues(DataTypes.ROW(inputType), (Object[]) predictData);

        NaiveBayes estimator = new NaiveBayes();
        estimator.setSmoothing(smoothing);
        if (featureCols != null)        estimator.setFeatureCols(featureCols);
        if (labelCol != null)              estimator.setLabelCol(labelCol);
        if (predictCol != null)         estimator.setPredictionCol(predictCol);

        NaiveBayesModel model = estimator.fit(trainTable);

        if (isSaveLoad) {
            String tempDir = Files.createTempDirectory("").toString();
            model.save(tempDir);
            env.execute();

            model = NaiveBayesModel.load(tempDir);
        }

        Table output = model.transform(predictTable)[0];

        Object[] actualObjects = IteratorUtils.toArray(output.execute().collect());
        Row[] actual = new Row[actualObjects.length];
        for (int i=0; i<actualObjects.length;i++) {
            actual[i] = (Row) actualObjects[i];
        }
        
        Assert.assertEquals(errorMessage, getFrequencyMap(expectedOutput), getFrequencyMap(actual));
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
