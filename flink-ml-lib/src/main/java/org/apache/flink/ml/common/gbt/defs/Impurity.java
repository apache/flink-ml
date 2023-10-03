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

import java.io.Serializable;

/** The base class for calculating information gain from statistics. */
public abstract class Impurity implements Cloneable, Serializable {

    // Number of instances.
    protected int numInstances;
    // Total of instance weights.
    protected double totalWeights;

    public Impurity(int numInstances, double totalWeights) {
        this.numInstances = numInstances;
        this.totalWeights = totalWeights;
    }

    /**
     * Calculates the prediction.
     *
     * @return The prediction.
     */
    public abstract double prediction();

    /**
     * Calculates the impurity.
     *
     * @return The impurity score.
     */
    public abstract double impurity();

    /**
     * Calculate the information gain over other impurity instances, usually coming from splitting
     * nodes.
     *
     * @param others Other impurity instances.
     * @return The value of information gain.
     */
    public abstract double gain(Impurity... others);

    /**
     * Add statistics from other impurity instance.
     *
     * @param other The other impurity instance.
     * @return The result after adding.
     */
    public abstract Impurity add(Impurity other);

    /**
     * Subtract statistics from other impurity instance.
     *
     * @param other The other impurity instance.
     * @return The result after subtracting.
     */
    public abstract Impurity subtract(Impurity other);

    /**
     * Get the total of instance weights.
     *
     * @return The total of instance weights.
     */
    public double getTotalWeights() {
        return totalWeights;
    }

    /**
     * Get the number of instances.
     *
     * @return The number of instances.
     */
    public int getNumInstances() {
        return numInstances;
    }

    @Override
    public Impurity clone() {
        try {
            return (Impurity) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new IllegalStateException("Can not clone the impurity instance.", e);
        }
    }
}
