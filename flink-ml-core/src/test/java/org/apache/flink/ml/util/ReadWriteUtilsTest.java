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

package org.apache.flink.ml.util;

import org.apache.flink.ml.api.ExampleStages;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hdfs.MiniDFSCluster;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

/** Tests {@link ReadWriteUtils}. */
public class ReadWriteUtilsTest extends AbstractTestBase {

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private MiniDFSCluster hdfsCluster;

    @Before
    public void before() throws IOException {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        Configuration conf = new Configuration();
        conf.set(MiniDFSCluster.HDFS_MINIDFS_BASEDIR, tempFolder.newFolder().getAbsolutePath());
        MiniDFSCluster.Builder builder = new MiniDFSCluster.Builder(conf);
        hdfsCluster = builder.build();
    }

    @After
    public void after() {
        hdfsCluster.shutdown();
    }

    @Test
    public void testModelSaveLoad() throws Exception {
        // Builds a SumModel that increments input value by 10.
        ExampleStages.SumModel model =
                new ExampleStages.SumModel().setModelData(tEnv.fromValues(10));
        List<List<Integer>> inputs = Collections.singletonList(Collections.singletonList(1));
        List<Integer> output = Collections.singletonList(11);

        // Save and load the model.
        String path = "hdfs://localhost:" + hdfsCluster.getNameNodePort() + "/sumModel";
        model.save(path);
        env.execute();

        ExampleStages.SumModel loadedModel = ExampleStages.SumModel.load(tEnv, path);
        // Executes the loaded SumModel and verifies that it produces the expected output.
        TestUtils.executeAndCheckOutput(env, loadedModel, inputs, output, null, null);
    }
}
