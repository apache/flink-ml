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

package org.apache.flink.ml.api;

import org.apache.flink.ml.api.ExampleStages.SumEstimator;
import org.apache.flink.ml.api.ExampleStages.SumModel;
import org.apache.flink.ml.api.ExampleStages.UnionAlgoOperator;
import org.apache.flink.ml.builder.Graph;
import org.apache.flink.ml.builder.GraphBuilder;
import org.apache.flink.ml.builder.GraphModel;
import org.apache.flink.ml.builder.TableId;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;

import org.junit.Before;
import org.junit.Test;

import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** Tests the behavior of {@link Graph} and {@link GraphModel}. */
public class GraphTest extends AbstractTestBase {
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
    }

    // Executes the given stage using the given inputs and verifies that it produces the expected
    // output. Then repeats this procedure after saving and loading the given stage.
    private static void executeSaveLoadAndCheckOutput(
            StreamTableEnvironment tEnv,
            Stage<?> stage,
            List<List<Integer>> inputs,
            List<Integer> expectedOutput,
            List<List<Integer>> modelDataInputs,
            List<Integer> expectedModelDataOutput,
            boolean modelDataExists)
            throws Exception {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);

        // Executes the given stage and verifies that it produces the expected output.
        TestUtils.executeAndCheckOutput(
                env, stage, inputs, expectedOutput, modelDataInputs, expectedModelDataOutput);
        // Saves and loads the given stage.
        String path = Files.createTempDirectory("").toString();
        stage.save(path);

        if (modelDataExists) {
            env.execute();
        }

        Stage<?> loadedStage = null;
        if (stage instanceof Estimator) {
            loadedStage = Graph.load(tEnv, path);
        } else {
            loadedStage = GraphModel.load(tEnv, path);
        }
        // Executes the loaded stage and verifies that it produces the expected output.
        TestUtils.executeAndCheckOutput(
                env, loadedStage, inputs, expectedOutput, modelDataInputs, expectedModelDataOutput);
    }

    @Test
    public void testGraphModelWithoutEstimator() throws Exception {
        GraphBuilder builder = new GraphBuilder();
        // Creates nodes.
        SumModel stage1 = new SumModel().setModelData(tEnv.fromValues(2));
        SumModel stage2 = new SumModel().setModelData(tEnv.fromValues(1));
        AlgoOperator<?> stage3 = new UnionAlgoOperator();
        // Creates inputs.
        TableId input1 = builder.createTableId();
        TableId input2 = builder.createTableId();
        // Feeds inputs and gets outputs.
        TableId output1 = builder.addAlgoOperator(stage1, input1)[0];
        TableId output2 = builder.addAlgoOperator(stage2, input2)[0];
        TableId output3 = builder.addAlgoOperator(stage3, output1, output2)[0];

        // Builds a Model from the graph.
        TableId[] inputs = new TableId[] {input1, input2};
        TableId[] outputs = new TableId[] {output3};
        Model<?> model = builder.buildModel(inputs, outputs);
        // Executes the GraphModel and verifies that it produces the expected output.
        List<List<Integer>> inputValues = new ArrayList<>();
        inputValues.add(Arrays.asList(1, 2, 3));
        inputValues.add(Arrays.asList(10, 11, 12));
        List<Integer> expectedOutputValues = Arrays.asList(3, 4, 5, 11, 12, 13);
        executeSaveLoadAndCheckOutput(
                tEnv, model, inputValues, expectedOutputValues, null, null, true);
    }

    @Test
    public void testGraphModelWithEstimator() throws Exception {
        GraphBuilder builder = new GraphBuilder();
        // Creates nodes.
        Estimator<?, ?> stage1 = new SumEstimator();
        Estimator<?, ?> stage2 = new SumEstimator();
        AlgoOperator<?> stage3 = new UnionAlgoOperator();
        // Creates inputs.
        TableId input1 = builder.createTableId();
        TableId input2 = builder.createTableId();
        // Feeds inputs and gets outputs.
        TableId output1 = builder.addEstimator(stage1, input1)[0];
        TableId output2 = builder.addEstimator(stage2, input2)[0];
        TableId output3 = builder.addAlgoOperator(stage3, output1, output2)[0];

        // Builds a Model from the graph.
        TableId[] inputs = new TableId[] {input1, input2};
        TableId[] outputs = new TableId[] {output3};
        Model<?> model = builder.buildModel(inputs, outputs);
        // Executes the GraphModel and verifies that it produces the expected output.
        List<List<Integer>> inputValues = new ArrayList<>();
        inputValues.add(Arrays.asList(1, 2, 3));
        inputValues.add(Arrays.asList(10, 11, 12));
        List<Integer> expectedOutputValues = Arrays.asList(7, 8, 9, 43, 44, 45);
        executeSaveLoadAndCheckOutput(
                tEnv, model, inputValues, expectedOutputValues, null, null, false);
    }

    @Test
    public void testGraphModelWithSetGetModelData() throws Exception {
        GraphBuilder builder = new GraphBuilder();
        // Creates nodes.
        SumModel stage1 = new SumModel().setModelData(tEnv.fromValues(1));
        SumModel stage2 = new SumModel();
        SumModel stage3 = new SumModel().setModelData(tEnv.fromValues(3));
        // Creates inputs and modelDataInputs.
        TableId input = builder.createTableId();
        TableId modelDataInput = builder.createTableId();
        // Feeds inputs and gets outputs.
        TableId output1 = builder.addAlgoOperator(stage1, input)[0];
        TableId output2 = builder.addAlgoOperator(stage2, output1)[0];
        builder.setModelDataOnModel(stage2, modelDataInput);
        TableId output3 = builder.addAlgoOperator(stage3, output2)[0];
        TableId modelDataOutput = builder.getModelDataFromModel(stage3)[0];

        // Builds a Model from the graph.
        TableId[] inputs = new TableId[] {input};
        TableId[] outputs = new TableId[] {output3};
        TableId[] modelDataInputs = new TableId[] {modelDataInput};
        TableId[] modelDataOutputs = new TableId[] {modelDataOutput};
        Model<?> model = builder.buildModel(inputs, outputs, modelDataInputs, modelDataOutputs);
        // Executes the GraphModel and verifies that it produces the expected output.
        List<List<Integer>> inputValues = Collections.singletonList(Arrays.asList(1, 2, 3));
        List<Integer> expectedOutputValues = Arrays.asList(7, 8, 9);
        List<List<Integer>> inputModelDataValues =
                Collections.singletonList(Collections.singletonList(2));
        List<Integer> expectedModelDataOutputValues = Collections.singletonList(3);
        executeSaveLoadAndCheckOutput(
                tEnv,
                model,
                inputValues,
                expectedOutputValues,
                inputModelDataValues,
                expectedModelDataOutputValues,
                true);
    }

    @Test
    public void testGraphWithEstimator() throws Exception {
        GraphBuilder builder = new GraphBuilder();
        // Creates nodes.
        Estimator<?, ?> stage1 = new SumEstimator();
        Estimator<?, ?> stage2 = new SumEstimator();
        AlgoOperator<?> stage3 = new UnionAlgoOperator();
        // Creates inputs.
        TableId input1 = builder.createTableId();
        TableId input2 = builder.createTableId();
        // Feeds inputs and gets outputs.
        TableId output1 = builder.addEstimator(stage1, input1)[0];
        TableId output2 = builder.addEstimator(stage2, input2)[0];
        TableId output3 = builder.addAlgoOperator(stage3, output1, output2)[0];

        // Builds an Estimator from the graph.
        TableId[] inputs = new TableId[] {input1, input2};
        TableId[] outputs = new TableId[] {output3};
        Estimator<?, ?> estimator = builder.buildEstimator(inputs, outputs);
        // Executes the Graph and verifies that it produces the expected output.
        List<List<Integer>> inputValues = new ArrayList<>();
        inputValues.add(Arrays.asList(1, 2, 3));
        inputValues.add(Arrays.asList(10, 11, 12));
        List<Integer> expectedOutputValues = Arrays.asList(7, 8, 9, 43, 44, 45);
        executeSaveLoadAndCheckOutput(
                tEnv, estimator, inputValues, expectedOutputValues, null, null, false);
    }

    @Test
    public void testGraphWithSetGetModelData() throws Exception {
        GraphBuilder builder = new GraphBuilder();
        // Creates nodes.
        Estimator<?, ?> stage1 = new SumEstimator();
        SumModel stage2 = new SumModel();
        AlgoOperator<?> stage3 = new UnionAlgoOperator();
        // Creates inputs.
        TableId input1 = builder.createTableId();
        TableId input2 = builder.createTableId();
        // Feeds inputs and gets outputs.
        TableId output1 = builder.addEstimator(stage1, input1)[0];
        TableId modelDataOutput = builder.getModelDataFromEstimator(stage1)[0];
        TableId output2 = builder.addAlgoOperator(stage2, input2)[0];
        builder.setModelDataOnModel(stage2, modelDataOutput);
        TableId output3 = builder.addAlgoOperator(stage3, output1, output2)[0];

        // Builds an Estimator from the graph.
        TableId[] inputs = new TableId[] {input1, input2};
        TableId[] outputs = new TableId[] {output3};
        TableId[] modelDataOutputs = new TableId[] {modelDataOutput};
        Estimator<?, ?> estimator = builder.buildEstimator(inputs, outputs, null, modelDataOutputs);
        // Executes the Graph and verifies that it produces the expected output.
        List<List<Integer>> inputValues = new ArrayList<>();
        inputValues.add(Arrays.asList(1, 2, 3));
        inputValues.add(Arrays.asList(10, 11, 12));
        List<Integer> expectedOutputValues = Arrays.asList(7, 8, 9, 16, 17, 18);
        List<Integer> expectedModelDataOutputValues = Collections.singletonList(6);
        executeSaveLoadAndCheckOutput(
                tEnv,
                estimator,
                inputValues,
                expectedOutputValues,
                null,
                expectedModelDataOutputValues,
                true);
    }
}
