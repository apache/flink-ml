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

package org.apache.flink.ml.classification;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.connector.source.Source;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.sink.FileSink;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModelData;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.filesystem.bucketassigners.BasePathBucketAssigner;
import org.apache.flink.streaming.api.functions.sink.filesystem.rollingpolicies.OnCheckpointRollingPolicy;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.TestBaseUtils;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Tests online model save and load. */
public class OnlineModelSaveLoadTest {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamTableEnvironment tEnv;
    private StreamExecutionEnvironment env;
    private static final List<Row> modelRows =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(Vectors.dense(2.0, 4.5, 3.0)),
                            Row.of(Vectors.dense(3.1, 4.6, 3.1)),
                            Row.of(Vectors.dense(20.1, 5.6, 3.1)),
                            Row.of(Vectors.dense(2.1, 4.7, 3.1))));
    private String tmpPath;

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        env.setParallelism(1);
        tEnv = StreamTableEnvironment.create(env);
    }

    private void saveModel() {
        Schema modelSchema =
                Schema.newBuilder().column("f0", DataTypes.of(DenseVector.class)).build();
        Table modelDataTable = tEnv.fromDataStream(env.fromCollection(modelRows), modelSchema);
        try {
            tmpPath = tempFolder.newFolder().getAbsolutePath();
            /* Saves online model data to given path (tmpPath). */
            saveModelData(
                    LogisticRegressionModelData.getModelDataStream(modelDataTable),
                    tmpPath,
                    new LogisticRegressionModelData.ModelDataEncoder());
            env.execute();
        } catch (Exception e) {
            throw new RuntimeException("Saves model data failed!");
        }
    }

    @Test
    public void testSaveAndLoadSingleModel() throws Exception {
        saveModel();
        /* Loads model with modelVersion equals "2" in model path. */
        String modelVersion = "2";
        Table modelData =
                tEnv.fromDataStream(
                        loadModelData(
                                env,
                                tmpPath,
                                new LogisticRegressionModelData.ModelDataDecoder(),
                                modelVersion));
        List<Row> loadedModelRows =
                IteratorUtils.toList(tEnv.toDataStream(modelData).executeAndCollect());
        Assert.assertEquals(modelRows.get(2), loadedModelRows.get(0));
    }

    @Test
    public void testSaveAndLoadOnlineModel() throws Exception {
        saveModel();
        Table modelData =
                tEnv.fromDataStream(
                        ReadWriteUtils.loadModelData(
                                env, tmpPath, new LogisticRegressionModelData.ModelDataDecoder()));

        List<Row> loadedModelRows =
                IteratorUtils.toList(tEnv.toDataStream(modelData).executeAndCollect());

        TestBaseUtils.compareResultCollections(
                loadedModelRows,
                modelRows,
                (o1, o2) -> {
                    DenseVector v1 = (DenseVector) o1.getField(0);
                    DenseVector v2 = (DenseVector) o2.getField(0);
                    for (int i = 0; i < Math.min(v1.size(), v2.size()); i++) {
                        int cmp = Double.compare(v1.get(i), v2.get(i));
                        if (cmp != 0) {
                            return cmp;
                        }
                    }
                    return 0;
                });
    }

    /** Assigns model version for every model data when sinking to files. */
    private static class ModelVersionAssigner<T> extends BasePathBucketAssigner<T> {
        private long modelVersion = 0L;

        @Override
        public String getBucketId(T element, Context context) {
            return String.valueOf(modelVersion++);
        }
    }

    /**
     * Saves the model data stream to the given path using the model encoder.
     *
     * @param model The model data stream.
     * @param path The parent directory of the model data file.
     * @param modelEncoder The encoder to encode the model data.
     * @param <T> The class type of the model data.
     */
    private static <T> void saveModelData(
            DataStream<T> model, String path, Encoder<T> modelEncoder) {
        FileSink<T> sink =
                FileSink.forRowFormat(
                                new org.apache.flink.core.fs.Path(path + "/data/"), modelEncoder)
                        .withRollingPolicy(OnCheckpointRollingPolicy.build())
                        .withBucketAssigner(new ModelVersionAssigner<>())
                        .build();
        model.sinkTo(sink);
    }

    /**
     * Loads model data from file path with assigned version.
     *
     * @param env A StreamExecutionEnvironment instance.
     * @param path The parent directory of the model data file.
     * @param modelDecoder The decoder used to decode the model data.
     * @param modelVersion Assigned model version.
     * @param <T> The class type of the model data.
     * @return The loaded model data.
     */
    private static <T> DataStream<T> loadModelData(
            StreamExecutionEnvironment env,
            String path,
            SimpleStreamFormat<T> modelDecoder,
            String modelVersion) {
        Source<T, ?, ?> source =
                FileSource.forRecordStreamFormat(
                                modelDecoder,
                                new org.apache.flink.core.fs.Path(path + "/data/" + modelVersion))
                        .build();
        return env.fromSource(source, WatermarkStrategy.noWatermarks(), "modelData");
    }
}
