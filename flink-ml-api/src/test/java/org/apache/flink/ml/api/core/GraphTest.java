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
import org.apache.flink.ml.api.core.ExampleStages.UnionAlgoOperator;
import org.apache.flink.ml.api.graph.Graph;
import org.apache.flink.ml.api.graph.GraphBuilder;
import org.apache.flink.ml.api.graph.TableId;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Test;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/** Tests the behavior of {@link Graph}. */
public class GraphTest extends AbstractTestBase {
    @Test
    public void testTransformerChain() throws Exception {
        GraphBuilder builder = new GraphBuilder();
        int delta = 1;

        // Creates nodes
        Stage<?> stage1 = new SumModel(delta);
        Stage<?> stage2 = new SumModel(delta);
        Stage<?> stage3 = new SumModel(delta);
        // Creates inputs and inputStates
        TableId input1 = builder.createTableId();
        // Feeds inputs to nodes and gets outputs.
        TableId output1 = builder.getOutputs(stage1, input1)[0];
        TableId output2 = builder.getOutputs(stage2, output1)[0];
        TableId output3 = builder.getOutputs(stage3, output2)[0];

        TableId[] inputs = new TableId[] {input1};
        TableId[] outputs = new TableId[] {output3};

        Model<?> model = builder.buildModel(inputs, outputs);

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        env.setParallelism(4);

        Table inputTable = tEnv.fromDataStream(env.fromCollection(Arrays.asList(1, 2, 3)));
        Table outputTable = model.transform(inputTable)[0];

        List<Integer> output =
                IteratorUtils.toList(
                        tEnv.toDataStream(outputTable, Integer.class).executeAndCollect());

        List<Integer> expectedOutput = Arrays.asList(4, 5, 6);
        compareResultCollections(expectedOutput, output, Comparator.naturalOrder());
    }

    @Test
    public void testTransformerDAG() throws Exception {
        GraphBuilder builder = new GraphBuilder();
        int delta = 1;

        // Creates nodes
        Stage<?> stage1 = new SumModel(delta);
        Stage<?> stage2 = new SumModel(delta);
        Stage<?> stage3 = new UnionAlgoOperator();
        // Creates inputs and inputStates
        TableId input1 = builder.createTableId();
        TableId input2 = builder.createTableId();
        // Feeds inputs to nodes and gets outputs.
        TableId output1 = builder.getOutputs(stage1, input1)[0];
        TableId output2 = builder.getOutputs(stage2, input2)[0];
        TableId output3 = builder.getOutputs(stage3, output1, output2)[0];

        TableId[] inputs = new TableId[] {input1, input2};
        TableId[] outputs = new TableId[] {output3};

        Model<?> model = builder.buildModel(inputs, outputs);

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        env.setParallelism(4);

        Table inputA = tEnv.fromDataStream(env.fromCollection(Arrays.asList(1, 2, 3)));
        Table inputB = tEnv.fromDataStream(env.fromCollection(Arrays.asList(10, 11, 12)));
        Table outputTable = model.transform(inputA, inputB)[0];

        List<Integer> output =
                IteratorUtils.toList(
                        tEnv.toDataStream(outputTable, Integer.class).executeAndCollect());

        List<Integer> expectedOutput = Arrays.asList(2, 3, 4, 11, 12, 13);
        compareResultCollections(expectedOutput, output, Comparator.naturalOrder());
    }

    @Test
    public void testEstimatorDAGWithGraphModel() throws Exception {
        GraphBuilder builder = new GraphBuilder();
        // Creates nodes
        Stage<?> stage1 = new SumEstimator();
        Stage<?> stage2 = new SumEstimator();
        Stage<?> stage3 = new UnionAlgoOperator();
        // Creates inputs and inputStates
        TableId input1 = builder.createTableId();
        TableId input2 = builder.createTableId();
        // Feeds inputs to nodes and gets outputs.
        TableId output1 = builder.getOutputs(stage1, input1)[0];
        TableId output2 = builder.getOutputs(stage2, input2)[0];
        TableId output3 = builder.getOutputs(stage3, output1, output2)[0];

        TableId[] inputs = new TableId[] {input1, input2};
        TableId[] outputs = new TableId[] {output3};

        Model<?> model = builder.buildModel(inputs, outputs);

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        env.setParallelism(4);

        Table inputA = tEnv.fromDataStream(env.fromCollection(Arrays.asList(1, 2, 3)));
        Table inputB = tEnv.fromDataStream(env.fromCollection(Arrays.asList(10, 11, 12)));
        Table outputTable = model.transform(inputA, inputB)[0];

        List<Integer> output =
                IteratorUtils.toList(
                        tEnv.toDataStream(outputTable, Integer.class).executeAndCollect());

        List<Integer> expectedOutput = Arrays.asList(7, 8, 9, 43, 44, 45);
        compareResultCollections(expectedOutput, output, Comparator.naturalOrder());
    }

    @Test
    public void testEstimatorDAGWithGraph() throws Exception {
        GraphBuilder builder = new GraphBuilder();
        // Creates nodes
        Stage<?> stage1 = new SumEstimator();
        Stage<?> stage2 = new SumEstimator();
        Stage<?> stage3 = new UnionAlgoOperator();
        // Creates inputs and inputStates
        TableId input1 = builder.createTableId();
        TableId input2 = builder.createTableId();
        // Feeds inputs to nodes and gets outputs.
        TableId output1 = builder.getOutputs(stage1, input1)[0];
        TableId output2 = builder.getOutputs(stage2, input2)[0];
        TableId output3 = builder.getOutputs(stage3, output1, output2)[0];

        TableId[] inputs = new TableId[] {input1, input2};
        TableId[] outputs = new TableId[] {output3};

        Estimator<?, ?> estimator = builder.buildEstimator(inputs, outputs);

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        env.setParallelism(4);

        Table inputA = tEnv.fromDataStream(env.fromCollection(Arrays.asList(1, 2, 3)));
        Table inputB = tEnv.fromDataStream(env.fromCollection(Arrays.asList(10, 11, 12)));

        Model<?> model = estimator.fit(inputA, inputB);
        Table outputTable = model.transform(inputA, inputB)[0];

        List<Integer> output =
                IteratorUtils.toList(
                        tEnv.toDataStream(outputTable, Integer.class).executeAndCollect());

        List<Integer> expectedOutput = Arrays.asList(7, 8, 9, 43, 44, 45);
        compareResultCollections(expectedOutput, output, Comparator.naturalOrder());
    }

    //    @Test
    //    public void testGraphBuilderWithDifferentSchemas() {
    //        GraphBuilder builder = new GraphBuilder();
    //
    //        // Creates nodes
    //        Stage<?> stage1 = new EstimatorA();
    //        Stage<?> stage2 = new ModelC();
    //        // Creates inputs
    //        TableId estimatorInput1 = builder.createTableId();
    //        TableId estimatorInput2 = builder.createTableId();
    //        TableId transformerInput1 = builder.createTableId();
    //
    //        // Feeds inputs to nodes and gets outputs.
    //        TableId output1 =
    //                builder.getOutputs(
    //                                stage1,
    //                                new TableId[] {estimatorInput1, estimatorInput2},
    //                                new TableId[] {transformerInput1})[0];
    //        TableId output2 = builder.getOutputs(stage2, output1)[0];
    //
    //        // Specifies the ordered lists of estimator inputs, transformer inputs, outputs, input
    //        // states and output states
    //        // that will be used as the inputs/outputs of the corresponding Graph and
    // GraphTransformer
    //        // APIs.
    //        TableId[] estimatorInputs = new TableId[] {estimatorInput1, estimatorInput2};
    //        TableId[] transformerInputs = new TableId[] {transformerInput1};
    //        TableId[] outputs = new TableId[] {output2};
    //        TableId[] inputStates = new TableId[] {};
    //        TableId[] outputStates = new TableId[] {};
    //
    //        Estimator estimator =
    //                builder.buildEstimator(
    //                        estimatorInputs, transformerInputs, outputs, inputStates,
    // outputStates);
    //        Model model = estimator.fit();
    //        Table[] results = model.transform();
    //
    //        assert results.length == 0;
    //    }
}
