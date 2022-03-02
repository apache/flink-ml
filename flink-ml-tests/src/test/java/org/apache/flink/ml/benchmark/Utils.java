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

import org.apache.flink.ml.api.Model;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.expressions.Expression;
import org.apache.flink.table.expressions.ExpressionParser;

import org.apache.flink.shaded.jackson2.org.yaml.snakeyaml.Yaml;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Utility class for benchmarks. */
public class Utils {

    /**
     * Uses a given model data to transform the input tables, and assigns each data point with a
     * label. The schema of the output table is "feature, label".
     *
     * @param model The true model.
     * @param inputs The input tables to be transformed.
     * @return The train data with labels assigned.
     */
    public static Table[] getTrainDataFromTrueModelAndData(Model<?> model, Table[] inputs) {
        Table[] transformedResult = model.transform(inputs);
        Table[] trainData = new Table[transformedResult.length];
        for (int i = 0; i < transformedResult.length; i++) {
            trainData[i] =
                    transformedResult[i]
                            .select(
                                    ExpressionParser.parseExpressionList("feature, prediction")
                                            .toArray(new Expression[0]))
                            .as("feature", "label");
        }
        return trainData;
    }

    /**
     * Parses the config of benchmarks from a yaml file.
     *
     * @param configFile The path of the yaml config file.
     * @return The list of benchmarks to run.
     * @throws IOException Throws exception if the config file does not exist.
     */
    @SuppressWarnings("unchecked")
    public static List<Benchmark> getMLBenchmarksFromFile(String configFile) throws IOException {
        String configString =
                FileUtils.readFileToString(new File(configFile), StandardCharsets.UTF_8);
        Map<?, ?> map = new Yaml().load(configString);

        Map<String, Object> commonParams = (Map<String, Object>) map.get("common");
        List<Map<String, String>> classNameAndParams =
                (List<Map<String, String>>) map.get("benchmarks");
        List<Benchmark> benchmarks = new ArrayList<>();
        for (Map<?, ?> classNameAndParam : classNameAndParams) {
            String name = classNameAndParam.get("name").toString();
            Map<String, Object> algoParams = (Map<String, Object>) classNameAndParam.get("params");
            for (Map.Entry<String, Object> kv : commonParams.entrySet()) {
                if (!algoParams.containsKey(kv.getKey())) {
                    algoParams.put(kv.getKey(), kv.getValue());
                }
            }
            BenchmarkStage<?> benchmarkStage;
            String benchmarkName = "org.apache.flink.ml.benchmark." + name + "Benchmark";
            try {
                benchmarkStage =
                        (BenchmarkStage<?>)
                                Class.forName(benchmarkName).getConstructor().newInstance();
            } catch (IllegalAccessException
                    | InstantiationException
                    | ClassNotFoundException
                    | NoSuchMethodException
                    | InvocationTargetException e) {
                throw new RuntimeException("Class not found: " + benchmarkName);
            }
            benchmarks.add(new Benchmark(benchmarkStage, BenchmarkParams.fromParams(algoParams)));
        }

        return benchmarks;
    }
}
