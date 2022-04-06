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

import org.apache.flink.util.Preconditions;

/** The result of executing a benchmark. */
public class BenchmarkResult {
    /** The benchmark name. */
    public final String name;

    /** The total execution time of the benchmark in milliseconds. */
    public final Double totalTimeMs;

    /** The total number of input records. */
    public final Long inputRecordNum;

    /** The average input throughput in number of records per second. */
    public final Double inputThroughput;

    /** The total number of output records. */
    public final Long outputRecordNum;

    /** The average output throughput in number of records per second. */
    public final Double outputThroughput;

    public BenchmarkResult(
            String name,
            Double totalTimeMs,
            Long inputRecordNum,
            Double inputThroughput,
            Long outputRecordNum,
            Double outputThroughput) {
        Preconditions.checkNotNull(name);
        this.name = name;
        this.totalTimeMs = totalTimeMs;
        this.inputRecordNum = inputRecordNum;
        this.inputThroughput = inputThroughput;
        this.outputRecordNum = outputRecordNum;
        this.outputThroughput = outputThroughput;
    }
}
