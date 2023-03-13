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

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.ml.util.FileUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

/** Entry class for benchmark execution. */
public class Benchmark {
    static final String VERSION_KEY = "version";

    private static final Logger LOG = LoggerFactory.getLogger(Benchmark.class);

    private static final Option HELP_OPTION =
            Option.builder("h")
                    .longOpt("help")
                    .desc("Show the help message for the command line interface.")
                    .build();

    private static final Option OUTPUT_FILE_OPTION =
            Option.builder()
                    .longOpt("output-file")
                    .desc("The output file name to save benchmark results.")
                    .hasArg()
                    .build();

    private static final Options OPTIONS =
            new Options().addOption(HELP_OPTION).addOption(OUTPUT_FILE_OPTION);

    private static void printHelp() {
        HelpFormatter formatter = new HelpFormatter();
        formatter.setLeftPadding(5);
        formatter.setWidth(80);

        System.out.println("./benchmark-run.sh <config-file-path> [OPTIONS]\n");
        formatter.setSyntaxPrefix("The following options are available:");
        formatter.printHelp(" ", OPTIONS);

        System.out.println();
    }

    private static void executeBenchmarks(CommandLine commandLine) throws Exception {
        String configFile = commandLine.getArgs()[0];
        Map<String, Map<String, Map<String, ?>>> benchmarks =
                BenchmarkUtils.parseJsonFile(configFile);
        System.out.printf("Found %d benchmarks.\n", benchmarks.keySet().size());
        String saveFile = commandLine.getOptionValue(OUTPUT_FILE_OPTION.getLongOpt());

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.getConfig().disableGenericTypes();
        env.setRestartStrategy(RestartStrategies.noRestart());
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        int index = 0;
        for (Map.Entry<String, Map<String, Map<String, ?>>> entry : benchmarks.entrySet()) {
            String benchmarkName = entry.getKey();
            Map<String, Map<String, ?>> benchmarkMap = entry.getValue();

            LOG.info(
                    String.format(
                            "Running benchmark %d/%d: %s",
                            index++, benchmarks.keySet().size(), benchmarkName));

            try {
                BenchmarkResult result =
                        BenchmarkUtils.runBenchmark(tEnv, benchmarkName, benchmarkMap, false);
                benchmarkMap.put("results", result.toMap());
                LOG.info(String.format("Benchmark %s finished.\n%s", benchmarkName, benchmarkMap));
            } catch (Exception e) {
                benchmarkMap.put(
                        "results",
                        Collections.singletonMap(
                                "exception",
                                String.format(
                                        "%s(%s:%s)",
                                        e,
                                        e.getStackTrace()[0].getFileName(),
                                        e.getStackTrace()[0].getLineNumber())));
                LOG.error(String.format("Benchmark %s failed.\n%s", benchmarkName, e));
            }
        }
        System.out.println("Benchmarks execution completed.");

        String benchmarkResultsJson =
                ReadWriteUtils.OBJECT_MAPPER
                        .writerWithDefaultPrettyPrinter()
                        .writeValueAsString(benchmarks);
        if (commandLine.hasOption(OUTPUT_FILE_OPTION.getLongOpt())) {
            FileUtils.saveToFile(saveFile, benchmarkResultsJson, true);
            System.out.printf("Benchmark results saved as json in %s.\n", saveFile);
        } else {
            System.out.printf("Benchmark results summary:\n%s\n", benchmarkResultsJson);
        }
    }

    public static void main(String[] args) throws Exception {
        CommandLineParser parser = new DefaultParser();
        CommandLine commandLine = parser.parse(OPTIONS, args);

        if (commandLine.hasOption(HELP_OPTION.getLongOpt())) {
            printHelp();
        } else if (commandLine.getArgs().length == 1) {
            executeBenchmarks(commandLine);
        } else {
            System.out.printf("Invalid command line arguments %s\n\n", Arrays.toString(args));
            System.out.println(
                    "Specify the help option (-h or --help) to get help on the command.");
        }
    }
}
