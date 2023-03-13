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
import org.apache.flink.ml.builder.Pipeline;
import org.apache.flink.ml.builder.PipelineModel;
import org.apache.flink.ml.servable.api.DataFrame;
import org.apache.flink.ml.servable.api.Row;
import org.apache.flink.ml.servable.builder.PipelineModelServable;
import org.apache.flink.ml.servable.types.DataTypes;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.apache.flink.ml.servable.TestUtils.assertDataFrameEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Tests the behavior of Pipeline and PipelineModel. */
public class PipelineTest extends AbstractTestBase {
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
    }

    @Test
    public void testPipelineModel() throws Exception {
        // Builds a PipelineModel that increments input value by 60. This PipelineModel consists of
        // three stages where each stage increments input value by 10, 20, and 30 respectively.
        SumModel modelA = new SumModel().setModelData(tEnv.fromValues(10));
        SumModel modelB = new SumModel().setModelData(tEnv.fromValues(20));
        SumModel modelC = new SumModel().setModelData(tEnv.fromValues(30));

        List<Stage<?>> stages = Arrays.asList(modelA, modelB, modelC);
        Model<?> model = new PipelineModel(stages);
        List<List<Integer>> inputs = Collections.singletonList(Arrays.asList(1, 2, 3));
        List<Integer> output = Arrays.asList(61, 62, 63);

        // Executes the original PipelineModel and verifies that it produces the expected output.
        TestUtils.executeAndCheckOutput(env, model, inputs, output, null, null);

        // Saves and loads the PipelineModel.
        String path = Files.createTempDirectory("").toString();
        model.save(path);
        env.execute();

        Model<?> loadedModel = PipelineModel.load(tEnv, path);
        // Executes the loaded PipelineModel and verifies that it produces the expected output.
        TestUtils.executeAndCheckOutput(env, loadedModel, inputs, output, null, null);
    }

    @Test
    public void testPipeline() throws Exception {
        // Builds a Pipeline that consists of a Model, an Estimator, and a model.
        SumModel modelA = new SumModel().setModelData(tEnv.fromValues(10));
        SumModel modelB = new SumModel().setModelData(tEnv.fromValues(30));

        List<Stage<?>> stages = Arrays.asList(modelA, new SumEstimator(), modelB);
        Estimator<?, ?> estimator = new Pipeline(stages);
        List<List<Integer>> inputs = Collections.singletonList(Arrays.asList(1, 2, 3));
        List<Integer> output = Arrays.asList(77, 78, 79);

        // Executes the original Pipeline and verifies that it produces the expected output.
        TestUtils.executeAndCheckOutput(env, estimator, inputs, output, null, null);

        // Saves and loads the Pipeline.
        String path = Files.createTempDirectory("").toString();
        estimator.save(path);
        env.execute();

        Estimator<?, ?> loadedEstimator = Pipeline.load(tEnv, path);
        // Executes the loaded Pipeline and verifies that it produces the expected output.
        TestUtils.executeAndCheckOutput(env, loadedEstimator, inputs, output, null, null);
    }

    @Test
    public void testSupportServable() {
        SumEstimator estimatorA = new SumEstimator();
        UnionAlgoOperator algoOperatorA = new UnionAlgoOperator();
        SumModel modelA = new SumModel();
        SumModel modelB = new SumModel();

        List<Stage<?>> stages = Arrays.asList(modelA, modelB);
        PipelineModel pipelineModel = new PipelineModel(stages);
        assertTrue(pipelineModel.supportServable());

        stages = Arrays.asList(estimatorA, modelA);
        pipelineModel = new PipelineModel(stages);
        assertFalse(pipelineModel.supportServable());

        stages = Arrays.asList(algoOperatorA, modelA);
        pipelineModel = new PipelineModel(stages);
        assertFalse(pipelineModel.supportServable());
    }

    @Test
    public void testPipelineModelServable() throws Exception {
        SumModel modelA = new SumModel().setModelData(tEnv.fromValues(10));
        SumModel modelB = new SumModel().setModelData(tEnv.fromValues(20));
        SumModel modelC = new SumModel().setModelData(tEnv.fromValues(30));

        List<Stage<?>> stages = Arrays.asList(modelA, modelB, modelC);
        Model<?> model = new PipelineModel(stages);

        PipelineModelServable servable =
                TestUtils.saveAndLoadServable(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        PipelineModel::loadServable);

        DataFrame input =
                new DataFrame(
                        Collections.singletonList("input"),
                        Collections.singletonList(DataTypes.INT),
                        Arrays.asList(
                                new Row(Collections.singletonList(1)),
                                new Row(Collections.singletonList(2)),
                                new Row(Collections.singletonList(3))));

        DataFrame output = servable.transform(input);

        DataFrame expectedOutput =
                new DataFrame(
                        Collections.singletonList("input"),
                        Collections.singletonList(DataTypes.INT),
                        Arrays.asList(
                                new Row(Collections.singletonList(61)),
                                new Row(Collections.singletonList(62)),
                                new Row(Collections.singletonList(63))));

        assertDataFrameEquals(expectedOutput, output);
    }
}
