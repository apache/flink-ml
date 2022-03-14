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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/** Entry class for benchmark execution. */
public class Benchmark {
    private static final Logger LOG = LoggerFactory.getLogger(Benchmark.class);

    static final String VERSION_KEY = "version";

    static final Option HELP_OPTION =
            Option.builder("h")
                    .longOpt("help")
                    .desc("Show the help message for the command line interface.")
                    .build();

    static final Option OUTPUT_FILE_OPTION =
            Option.builder()
                    .longOpt("output-file")
                    .desc("The output file name to save benchmark results.")
                    .hasArg()
                    .build();

    static final Options OPTIONS =
            new Options().addOption(HELP_OPTION).addOption(OUTPUT_FILE_OPTION);

    public static void printHelp() {
        HelpFormatter formatter = new HelpFormatter();
        formatter.setLeftPadding(5);
        formatter.setWidth(80);

        System.out.println("./flink-ml-benchmark.sh <config-file-path> [OPTIONS]");
        System.out.println();
        formatter.setSyntaxPrefix("The following options are available:");
        formatter.printHelp(" ", OPTIONS);

        System.out.println();
    }

    @SuppressWarnings("unchecked")
    public static void executeBenchmarks(CommandLine commandLine) throws Exception {
        String configFile = commandLine.getArgs()[0];
        Map<String, ?> benchmarks = BenchmarkUtils.parseJsonFile(configFile);
        System.out.println("Found benchmarks " + benchmarks.keySet());

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        List<BenchmarkResult> results = new ArrayList<>();
        for (Map.Entry<String, ?> benchmark : benchmarks.entrySet()) {
            LOG.info("Running benchmark " + benchmark.getKey() + ".");

            BenchmarkResult result =
                    BenchmarkUtils.runBenchmark(
                            tEnv, benchmark.getKey(), (Map<String, ?>) benchmark.getValue());

            results.add(result);
            LOG.info(BenchmarkUtils.getResultsMapAsJson(result));
        }

        String benchmarkResultsJson =
                BenchmarkUtils.getResultsMapAsJson(results.toArray(new BenchmarkResult[0]));
        System.out.println("Benchmarks execution completed.");
        System.out.println("Benchmark results summary:");
        System.out.println(benchmarkResultsJson);

        if (commandLine.hasOption(OUTPUT_FILE_OPTION.getLongOpt())) {
            String saveFile = commandLine.getOptionValue(OUTPUT_FILE_OPTION.getLongOpt());
            ReadWriteUtils.saveToFile(saveFile, benchmarkResultsJson, true);
            System.out.println("Benchmark results saved as json in " + saveFile + ".");
        }
    }

    public static void printInvalidError(String[] args) {
        System.out.println("Invalid command line arguments " + Arrays.toString(args));
        System.out.println();
        System.out.println("Specify the help option (-h or --help) to get help on the command.");
    }

    public static void main(String[] args) throws Exception {
        CommandLineParser parser = new DefaultParser();
        CommandLine commandLine = parser.parse(OPTIONS, args);

        if (commandLine.hasOption(HELP_OPTION.getLongOpt())) {
            printHelp();
        } else if (commandLine.getArgs().length == 1) {
            executeBenchmarks(commandLine);
        } else {
            printInvalidError(args);
        }
    }
}
