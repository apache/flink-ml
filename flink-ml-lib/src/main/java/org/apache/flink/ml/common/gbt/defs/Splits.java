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

/**
 * This class stores splits of nodes in the current layer, and necessary information of
 * all-reducing..
 */
public class Splits {

    // Stores source subtask ID when reducing or target subtask ID when scattering.
    public int subtaskId;
    // Stores splits of nodes in the current layer.
    public Split[] splits;

    public Splits(int subtaskId, Split[] splits) {
        this.subtaskId = subtaskId;
        this.splits = splits;
    }

    private Splits accumulate(Splits other) {
        for (int i = 0; i < splits.length; ++i) {
            if (splits[i] == null && other.splits[i] != null) {
                splits[i] = other.splits[i];
            } else if (splits[i] != null && other.splits[i] != null) {
                if (splits[i].gain < other.splits[i].gain) {
                    splits[i] = other.splits[i];
                } else if (splits[i].gain == other.splits[i].gain) {
                    if (splits[i].featureId < other.splits[i].featureId) {
                        splits[i] = other.splits[i];
                    }
                }
            }
        }
        return this;
    }

    /** Aggregator for Splits. */
    public static class Aggregator implements AggregateFunction<Splits, Splits, Splits> {
        @Override
        public Splits createAccumulator() {
            return null;
        }

        @Override
        public Splits add(Splits value, Splits accumulator) {
            if (null == accumulator) {
                return value;
            }
            return accumulator.accumulate(value);
        }

        @Override
        public Splits getResult(Splits accumulator) {
            return accumulator;
        }

        @Override
        public Splits merge(Splits a, Splits b) {
            return a.accumulate(b);
        }
    }
}
