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

package org.apache.flink.ml.common.iteration;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.List;

/**
 * A FlatMapFunction which forwards input values in the last round of the iteration to the output.
 * Other input values will be dropped.
 *
 * @param <T> The class type of the input element.
 */
public class ForwardInputsOfLastRound<T> implements FlatMapFunction<T, T>, IterationListener<T> {
    private List<T> valuesInLastEpoch = new ArrayList<>();
    private List<T> valuesInCurrentEpoch = new ArrayList<>();

    @Override
    public void flatMap(T value, Collector<T> out) {
        valuesInCurrentEpoch.add(value);
    }

    @Override
    public void onEpochWatermarkIncremented(int epochWatermark, Context context, Collector<T> out) {
        valuesInLastEpoch = valuesInCurrentEpoch;
        valuesInCurrentEpoch = new ArrayList<>();
    }

    @Override
    public void onIterationTerminated(Context context, Collector<T> out) {
        for (T value : valuesInLastEpoch) {
            out.collect(value);
        }
        if (!valuesInCurrentEpoch.isEmpty()) {
            throw new IllegalStateException(
                    "flatMap() is invoked since the last onEpochWatermarkIncremented callback");
        }
        valuesInLastEpoch.clear();
    }
}
