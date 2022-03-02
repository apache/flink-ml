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

import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.api.Stage;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.DiscardingSink;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/** The main entrance of benchmark algorithms. */
public class Main {
    private static final Logger LOG = LoggerFactory.getLogger(Main.class);
    private static final String exampleConfig = "mlbench-demo.yaml";

    public static void main(String[] args) throws Exception {
        String configFile =
                args.length > 0 ? args[0] : Main.class.getResource("/").getPath() + exampleConfig;
        List<Benchmark> benchmarks = Utils.getMLBenchmarksFromFile(configFile);
        for (Benchmark benchmark : benchmarks) {
            runBenchmark(benchmark);
        }
    }

    @SuppressWarnings("rawtypes")
    private static void runBenchmark(Benchmark benchmark) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        env.setParallelism(benchmark.params.numPartitions);
        BenchmarkContext context = new BenchmarkContext(benchmark.params, env);

        long startTime = System.currentTimeMillis();
        LOG.info("Starts running " + benchmark + ", time: " + startTime);
        BenchmarkStage<?> benchmarkStage = benchmark.stage;
        Stage<?> mlStage = benchmarkStage.getStage(context);
        Table[] trainData = benchmarkStage.getTrainData(context);
        Table[] testData = benchmarkStage.getTestData(context);

        if (mlStage instanceof Estimator) {
            Model<?> m = ((Estimator) mlStage).fit(trainData);
            Table[] predictions = m.transform(testData);
            tEnv.toDataStream(predictions[0]).addSink(new DiscardingSink<>());
            env.execute();
        } else if (mlStage instanceof AlgoOperator) {
            AlgoOperator<?> algoOperator = (AlgoOperator) mlStage;
            Table[] predictions = algoOperator.transform(testData);
            tEnv.toDataStream(predictions[0]).addSink(new DiscardingSink<>());
            env.execute();
        }
        LOG.info(
                "Finished running "
                        + benchmark
                        + ", duration(s): "
                        + (System.currentTimeMillis() - startTime) / 1000);
    }
}
