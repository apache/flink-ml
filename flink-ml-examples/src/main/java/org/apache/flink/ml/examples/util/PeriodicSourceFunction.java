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

package org.apache.flink.ml.examples.util;

import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.types.Row;

import java.util.List;

/** A source function that collects provided data periodically at a fixed interval. */
public class PeriodicSourceFunction implements SourceFunction<Row> {
    private final long interval;

    private final List<List<Row>> data;

    private int index = 0;

    private boolean isRunning = true;

    /**
     * @param interval The time interval in milliseconds to collect data into sourceContext.
     * @param data The data to be collected. Each element is a list of records to be collected
     *     between two adjacent intervals.
     */
    public PeriodicSourceFunction(long interval, List<List<Row>> data) {
        this.interval = interval;
        this.data = data;
    }

    @Override
    public void run(SourceFunction.SourceContext<Row> sourceContext) throws Exception {
        while (isRunning) {
            for (Row data : this.data.get(index)) {
                sourceContext.collect(data);
            }
            Thread.sleep(interval);
            index = (index + 1) % this.data.size();
        }
    }

    @Override
    public void cancel() {
        isRunning = false;
    }
}
