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

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.util.Preconditions;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.WeakHashMap;

/**
 * A manager for operator-scope managed memory.
 *
 * <p>Every operator must call {@link #getOrCreate} to get an instance to manage usages of
 * operator-scope managed memory.
 *
 * <p>In the operator, for every usage of operator-scope managed memory, {@link #register} is called
 * to declare its weight of memory usage with a key. Then, the fraction of memory can be obtained by
 * calling {@link #getFraction}. Note that all calls of {@link #register} must be before those of
 * {@link #getFraction}.
 */
@PublicEvolving
public class OperatorScopeManagedMemoryManager {

    /**
     * Stores instances corresponding to operators. The instance is expected to be released after
     * some point after the corresponding operator ID is unused.
     */
    private static final Map<OperatorID, OperatorScopeManagedMemoryManager> managers =
            Collections.synchronizedMap(new WeakHashMap<>());

    /** Stores keys and weights of memory usages. */
    protected Map<String, Double> weights = new HashMap<>();
    /** Indicates whether the `weights` is frozen. */
    protected boolean frozen = false;
    /** Stores sum of weights of all usages. */
    protected double sum;

    OperatorScopeManagedMemoryManager() {}

    /**
     * Gets or creates an instance identified by the operator ID.
     *
     * @param operatorID The operator ID.
     * @return An instance of {@link OperatorScopeManagedMemoryManager}.
     */
    public static OperatorScopeManagedMemoryManager getOrCreate(OperatorID operatorID) {
        return managers.computeIfAbsent(
                operatorID, (key) -> new OperatorScopeManagedMemoryManager());
    }

    /**
     * Registers a usage of operator-scope managed memory with memory weight declared.
     *
     * @param key The key to identify the usage.
     * @param weight The weight of memory usage.
     */
    public void register(String key, double weight) {
        Preconditions.checkState(!frozen, "Cannot call register after getFraction is called.");
        Preconditions.checkState(!weights.containsKey(key), "Cannot set a same key {} twice.", key);
        Preconditions.checkArgument(weight >= 0, "The weight must be non-negative.");
        weights.put(key, weight);
    }

    /**
     * Gets the fraction of operator-scope managed memory for the usage.
     *
     * @param key The key to identify the usage.
     * @return The fraction.
     */
    public double getFraction(String key) {
        Preconditions.checkArgument(weights.containsKey(key));
        if (!frozen) {
            frozen = true;
            sum = weights.values().stream().mapToDouble(d -> d).sum();
        }
        return sum > 0 ? weights.get(key) / sum : 0;
    }
}
