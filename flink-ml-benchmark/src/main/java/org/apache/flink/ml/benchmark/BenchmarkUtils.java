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

import org.apache.flink.api.common.JobExecutionResult;
import org.apache.flink.api.common.accumulators.LongCounter;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.api.Stage;
import org.apache.flink.ml.benchmark.datagenerator.DataGenerator;
import org.apache.flink.ml.benchmark.datagenerator.InputDataGenerator;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.util.Preconditions;

import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.core.JsonProcessingException;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/** Utility methods for benchmarks. */
public class BenchmarkUtils {
    /** Loads benchmark configuration maps from the provided json file. */
    @SuppressWarnings("unchecked")
    public static Map<String, ?> parseJsonFile(String path) throws IOException {
        InputStream inputStream = new FileInputStream(path);
        Map<String, ?> jsonMap = ReadWriteUtils.OBJECT_MAPPER.readValue(inputStream, Map.class);
        Preconditions.checkArgument(
                jsonMap.containsKey(Benchmark.VERSION_KEY)
                        && jsonMap.get(Benchmark.VERSION_KEY).equals(1));

        return jsonMap.entrySet().stream()
                .filter(x -> !x.getKey().equals(Benchmark.VERSION_KEY))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    /**
     * Instantiates a benchmark from its parameter map and executes the benchmark in the provided
     * environment.
     *
     * @return Results of the executed benchmark.
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    public static BenchmarkResult runBenchmark(
            StreamTableEnvironment tEnv, String name, Map<String, ?> params) throws Exception {
        Stage stage = ReadWriteUtils.instantiateWithParams((Map<String, ?>) params.get("stage"));
        InputDataGenerator inputDataGenerator =
                ReadWriteUtils.instantiateWithParams((Map<String, ?>) params.get("inputData"));
        DataGenerator modelDataGenerator = null;
        if (params.containsKey("modelData")) {
            modelDataGenerator =
                    ReadWriteUtils.instantiateWithParams((Map<String, ?>) params.get("modelData"));
        }

        return runBenchmark(tEnv, name, stage, inputDataGenerator, modelDataGenerator);
    }

    /**
     * Executes a benchmark from a stage with its inputDataGenerator and modelDataGenerator in the
     * provided environment.
     *
     * @return Results of the executed benchmark.
     */
    private static BenchmarkResult runBenchmark(
            StreamTableEnvironment tEnv,
            String name,
            Stage<?> stage,
            InputDataGenerator<?> inputDataGenerator,
            DataGenerator<?> modelDataGenerator)
            throws Exception {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);

        Table[] inputTables = inputDataGenerator.getData(tEnv);
        if (modelDataGenerator != null) {
            ((Model<?>) stage).setModelData(modelDataGenerator.getData(tEnv));
        }

        Table[] outputTables;
        if (stage instanceof Estimator) {
            outputTables = ((Estimator<?, ?>) stage).fit(inputTables).getModelData();
        } else if (stage instanceof AlgoOperator) {
            outputTables = ((AlgoOperator<?>) stage).transform(inputTables);
        } else {
            throw new IllegalArgumentException("Unsupported Stage class " + stage.getClass());
        }

        for (Table table : outputTables) {
            tEnv.toDataStream(table).addSink(new CountingAndDiscardingSink<>());
        }

        JobExecutionResult executionResult = env.execute("Flink ML Benchmark Job " + name);

        double totalTimeMs = (double) executionResult.getNetRuntime(TimeUnit.MILLISECONDS);
        long inputRecordNum = inputDataGenerator.getNumValues();
        double inputThroughput = inputRecordNum * 1000.0 / totalTimeMs;
        long outputRecordNum =
                executionResult.getAccumulatorResult(CountingAndDiscardingSink.COUNTER_NAME);
        double outputThroughput = outputRecordNum * 1000.0 / totalTimeMs;

        return new BenchmarkResult(
                name,
                totalTimeMs,
                inputRecordNum,
                inputThroughput,
                outputRecordNum,
                outputThroughput);
    }

    /** Converts the benchmark results to a json string as a map. */
    public static String getResultsMapAsJson(BenchmarkResult... results)
            throws JsonProcessingException {
        List<Map<String, ?>> resultsMap = new ArrayList<>();
        for (BenchmarkResult result : results) {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("name", result.name);
            map.put("totalTimeMs", result.totalTimeMs);
            map.put("inputRecordNum", result.inputRecordNum);
            map.put("inputThroughput", result.inputThroughput);
            map.put("outputRecordNum", result.outputRecordNum);
            map.put("outputThroughput", result.outputThroughput);
            resultsMap.add(map);
        }
        return ReadWriteUtils.OBJECT_MAPPER
                .writerWithDefaultPrettyPrinter()
                .writeValueAsString(resultsMap);
    }

    /**
     * A stream sink that counts the number of all elements. The counting result is stored in an
     * {@link org.apache.flink.api.common.accumulators.Accumulator} specified by {@link
     * #COUNTER_NAME} and can be acquired by {@link
     * org.apache.flink.api.common.JobExecutionResult#getAccumulatorResult(String)}.
     *
     * @param <T> The type of elements received by the sink.
     */
    private static class CountingAndDiscardingSink<T> extends RichSinkFunction<T> {
        public static final String COUNTER_NAME = "numElements";

        private static final long serialVersionUID = 1L;

        private final LongCounter numElementsCounter = new LongCounter();

        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            getRuntimeContext().addAccumulator(COUNTER_NAME, numElementsCounter);
        }

        @Override
        public void invoke(T value, Context context) {
            numElementsCounter.add(1L);
        }
    }
}
