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

package org.apache.flink.ml.api.core;

import org.apache.flink.ml.api.core.ExampleStages.SumEstimator;
import org.apache.flink.ml.api.core.ExampleStages.SumModel;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/** Tests the behavior of Pipeline and PipelineModel. */
public class PipelineTest extends AbstractTestBase {

    // Executes the given stage and verifies that it produces the expected output.
    private static void executeAndCheckOutput(
            Stage<?> stage, List<Integer> input, List<Integer> expectedOutput) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        env.setParallelism(4);

        Table inputTable = tEnv.fromDataStream(env.fromCollection(input));

        Table outputTable;

        if (stage instanceof AlgoOperator) {
            outputTable = ((AlgoOperator<?>) stage).transform(inputTable)[0];
        } else {
            Estimator<?, ?> estimator = (Estimator<?, ?>) stage;
            Model<?> model = estimator.fit(inputTable);
            outputTable = model.transform(inputTable)[0];
        }

        List<Integer> output =
                IteratorUtils.toList(
                        tEnv.toDataStream(outputTable, Integer.class).executeAndCollect());
        compareResultCollections(expectedOutput, output, Comparator.naturalOrder());
    }

    @Test
    public void testPipelineModel() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        // Builds a PipelineModel that increments input value by 60. This PipelineModel consists of
        // three stages where each stage increments input value by 10, 20, and 30 respectively.
        SumModel modelA = new SumModel();
        modelA.setModelData(tEnv.fromValues(10));
        SumModel modelB = new SumModel();
        modelB.setModelData(tEnv.fromValues(20));
        SumModel modelC = new SumModel();
        modelC.setModelData(tEnv.fromValues(30));

        List<Stage<?>> stages = Arrays.asList(modelA, modelB, modelC);
        Model<?> model = new PipelineModel(stages);

        // Executes the original PipelineModel and verifies that it produces the expected output.
        executeAndCheckOutput(model, Arrays.asList(1, 2, 3), Arrays.asList(61, 62, 63));

        // Saves and loads the PipelineModel.
        Path tempDir = Files.createTempDirectory("PipelineTest");
        String path = Paths.get(tempDir.toString(), "testPipelineModelSaveLoad").toString();
        model.save(path);
        Model<?> loadedModel = PipelineModel.load(path);

        // Executes the loaded PipelineModel and verifies that it produces the expected output.
        executeAndCheckOutput(loadedModel, Arrays.asList(1, 2, 3), Arrays.asList(61, 62, 63));
    }

    @Test
    public void testPipeline() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        // Builds a Pipeline that consists of a Model, an Estimator, and a model.
        SumModel modelA = new SumModel();
        modelA.setModelData(tEnv.fromValues(10));
        SumModel modelB = new SumModel();
        modelB.setModelData(tEnv.fromValues(30));

        List<Stage<?>> stages = Arrays.asList(modelA, new SumEstimator(), modelB);
        Estimator<?, ?> estimator = new Pipeline(stages);

        // Executes the original Pipeline and verifies that it produces the expected output.
        executeAndCheckOutput(estimator, Arrays.asList(1, 2, 3), Arrays.asList(77, 78, 79));

        // Saves and loads the Pipeline.
        Path tempDir = Files.createTempDirectory("PipelineTest");
        String path = Paths.get(tempDir.toString(), "testPipeline").toString();
        estimator.save(path);
        Estimator<?, ?> loadedEstimator = Pipeline.load(path);

        // Executes the loaded Pipeline and verifies that it produces the expected output.
        executeAndCheckOutput(loadedEstimator, Arrays.asList(1, 2, 3), Arrays.asList(77, 78, 79));
    }
}
