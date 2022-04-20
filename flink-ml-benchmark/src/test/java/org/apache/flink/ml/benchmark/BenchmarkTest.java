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

package org.apache.flink.ml.benchmark;

import org.apache.flink.ml.benchmark.datagenerator.common.DenseVectorGenerator;
import org.apache.flink.ml.clustering.kmeans.KMeans;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;

import org.apache.commons.io.FileUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** Tests benchmarks. */
public class BenchmarkTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    @Test
    public void testParseJsonFile() throws Exception {
        File configFile = new File(tempFolder.newFolder().getAbsolutePath() + "/test-conf.json");
        InputStream inputStream =
                this.getClass().getClassLoader().getResourceAsStream("benchmark-conf.json");
        FileUtils.copyInputStreamToFile(inputStream, configFile);

        Map<String, ?> benchmarks = BenchmarkUtils.parseJsonFile(configFile.getAbsolutePath());
        assertEquals(benchmarks.size(), 8);
        assertTrue(benchmarks.containsKey("KMeans-1"));
        assertTrue(benchmarks.containsKey("KMeansModel-1"));
    }

    @Test
    public void testRunBenchmark() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        Map<String, Map<String, ?>> params = new HashMap<>();

        Map<String, Object> stageParams = new HashMap<>();
        stageParams.put("className", KMeans.class.getCanonicalName());
        stageParams.put(
                "paramMap",
                new HashMap<String, Object>() {
                    {
                        put("k", 5);
                        put("featuresCol", "test_feature");
                    }
                });
        params.put("stage", stageParams);

        Map<String, Object> inputDataParams = new HashMap<>();
        inputDataParams.put("className", DenseVectorGenerator.class.getCanonicalName());
        inputDataParams.put(
                "paramMap",
                new HashMap<String, Object>() {
                    {
                        put("colNames", new String[] {"test_feature"});
                        put("numValues", 1000L);
                        put("vectorDim", 10);
                    }
                });
        params.put("inputData", inputDataParams);

        long estimatedTime = System.currentTimeMillis();
        BenchmarkResult result = BenchmarkUtils.runBenchmark(tEnv, "testBenchmarkName", params);
        estimatedTime = System.currentTimeMillis() - estimatedTime;

        assertEquals("testBenchmarkName", result.name);
        assertTrue(result.totalTimeMs > 0);
        assertTrue(result.totalTimeMs <= estimatedTime);
        assertEquals(1000L, (long) result.inputRecordNum);
        assertEquals(1L, (long) result.outputRecordNum);
        assertEquals(
                result.inputRecordNum * 1000.0 / result.totalTimeMs, result.inputThroughput, 1e-5);
        assertEquals(
                result.outputRecordNum * 1000.0 / result.totalTimeMs,
                result.outputThroughput,
                1e-5);
    }
}
