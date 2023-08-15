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

package org.apache.flink.iteration.datacache.nonkeyed;

import org.apache.flink.util.Preconditions;

import java.util.HashMap;
import java.util.Map;

/**
 * A manager for operator-scope managed memory.
 *
 * <p>Every operator must create one (and only one) instance of this class to manage usages of
 * operator-scope managed memery. In the operator, for every usage, {@link #register} is called to
 * declare its weight of memory usage with a key. Then, the fraction of memory can be obtained by
 * calling {@link #getFraction}.
 *
 * <p>Note that all calls of {@link #register} must be before those of {@link #getFraction}.
 */
public class OperatorScopeManagedMemoryManager {

    protected Map<String, Double> weights = new HashMap<>();
    protected boolean frozen = false;
    protected double sum;

    public void register(String key, double weight) {
        Preconditions.checkState(!frozen, "Cannot call register after getFraction is called.");
        Preconditions.checkArgument(weight >= 0, "The weight must be non-negative.");
        weights.put(key, weight);
    }

    public double getFraction(String key) {
        Preconditions.checkArgument(weights.containsKey(key));
        if (!frozen) {
            frozen = true;
            sum = weights.values().stream().mapToDouble(d -> d).sum();
        }
        return sum > 0 ? weights.get(key) / sum : 0;
    }
}
