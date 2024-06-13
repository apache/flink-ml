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

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.lossfunc.LossFunc;

import java.io.Serializable;
import java.util.Random;

/**
 * Stores necessary static context information for training. Subtasks of co-located operators
 * scheduled in a same TaskManager share a same context.
 */
public class TrainContext implements Serializable {
    /** Subtask ID of co-located operators. */
    public int subtaskId;

    /** Number of subtasks of co-located operators. */
    public int numSubtasks;

    /** Configurations for the boosting. */
    public BoostingStrategy strategy;

    /** Number of instances. */
    public int numInstances;

    /** Number of bagging instances used for training one tree. */
    public int numBaggingInstances;

    /** Randomizer for sampling instances. */
    public Random instanceRandomizer;

    /** Number of features. */
    public int numFeatures;

    /** Number of bagging features tested for splitting one node. */
    public int numBaggingFeatures;

    /** Randomizer for sampling features. */
    public Random featureRandomizer;

    /** Meta information of every feature. */
    public FeatureMeta[] featureMetas;

    /** Number of bins for every feature. */
    public int[] numFeatureBins;

    /** Sum and count of labels of all samples. */
    public Tuple2<Double, Long> labelSumCount;

    /** The prior value for prediction. */
    public double prior;

    /** The loss function. */
    public LossFunc loss;
}
