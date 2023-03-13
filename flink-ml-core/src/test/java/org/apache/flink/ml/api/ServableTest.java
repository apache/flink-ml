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

import org.apache.flink.ml.api.ExampleStages.SumModel;
import org.apache.flink.ml.servable.api.DataFrame;
import org.apache.flink.ml.servable.api.Row;
import org.apache.flink.ml.servable.builder.ExampleServables.SumModelServable;
import org.apache.flink.ml.servable.types.DataTypes;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayInputStream;
import java.util.Arrays;
import java.util.Collections;

import static org.apache.flink.ml.servable.TestUtils.assertDataFrameEquals;

/** Tests the behavior of integration between Transformer and Servable. */
public class ServableTest extends AbstractTestBase {

    private StreamTableEnvironment tEnv;

    private static final DataFrame INPUT =
            new DataFrame(
                    Collections.singletonList("input"),
                    Collections.singletonList(DataTypes.INT),
                    Arrays.asList(
                            new Row(Collections.singletonList(1)),
                            new Row(Collections.singletonList(2)),
                            new Row(Collections.singletonList(3))));

    private static final DataFrame EXPECTED_OUTPUT =
            new DataFrame(
                    Collections.singletonList("input"),
                    Collections.singletonList(DataTypes.INT),
                    Arrays.asList(
                            new Row(Collections.singletonList(11)),
                            new Row(Collections.singletonList(12)),
                            new Row(Collections.singletonList(13))));

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    @Before
    public void before() {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
    }

    @Test
    public void testSaveModelLoadServable() throws Exception {
        String modelPath = tempFolder.newFolder().getAbsolutePath();

        SumModel model = new SumModel().setModelData(tEnv.fromValues(10));

        SumModelServable servable =
                TestUtils.saveAndLoadServable(tEnv, model, modelPath, SumModel::loadServable);

        DataFrame output = servable.transform(INPUT);

        assertDataFrameEquals(EXPECTED_OUTPUT, output);
    }

    @Test
    public void testSetModelData() throws Exception {
        SumModel model = new SumModel().setModelData(tEnv.fromValues(10));
        Table modelDataTable = model.getModelData()[0];

        byte[] serializedModelData =
                tEnv.toDataStream(modelDataTable)
                        .map(x -> SumModelServable.serialize(x.getField(0)))
                        .executeAndCollect()
                        .next();

        SumModelServable servable =
                new SumModelServable().setModelData(new ByteArrayInputStream(serializedModelData));

        DataFrame output = servable.transform(INPUT);

        assertDataFrameEquals(EXPECTED_OUTPUT, output);
    }
}
