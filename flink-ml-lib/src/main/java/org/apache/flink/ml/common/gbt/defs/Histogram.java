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

package org.apache.flink.ml.common.gbt.defs;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.util.Preconditions;

import java.io.Serializable;

/**
 * This class stores values of histogram bins, and necessary information of reducing and scattering.
 */
public class Histogram implements Serializable {

    // Stores source subtask ID when reducing or target subtask ID when scattering.
    public int subtaskId;
    // Stores values of histogram bins.
    public double[] hists;
    // Stores the number of elements received by subtasks in scattering.
    public int[] recvcnts;

    public Histogram(int subtaskId, double[] hists, int[] recvcnts) {
        this.subtaskId = subtaskId;
        this.hists = hists;
        this.recvcnts = recvcnts;
    }

    private Histogram accumulate(Histogram other) {
        Preconditions.checkArgument(hists.length == other.hists.length);
        for (int i = 0; i < hists.length; i += 1) {
            hists[i] += other.hists[i];
        }
        return this;
    }

    /** Aggregator for Histogram. */
    public static class Aggregator
            implements AggregateFunction<Histogram, Histogram, Histogram>, Serializable {
        @Override
        public Histogram createAccumulator() {
            return null;
        }

        @Override
        public Histogram add(Histogram value, Histogram accumulator) {
            if (null == accumulator) {
                return value;
            }
            return accumulator.accumulate(value);
        }

        @Override
        public Histogram getResult(Histogram accumulator) {
            return accumulator;
        }

        @Override
        public Histogram merge(Histogram a, Histogram b) {
            return a.accumulate(b);
        }
    }
}
