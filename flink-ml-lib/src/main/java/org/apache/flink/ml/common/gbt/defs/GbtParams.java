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

/** Internal parameters of a gradient boosting trees algorithm. */
public class GbtParams implements Serializable {
    public TaskType taskType;

    // Parameters related with input data.
    public String[] featuresCols;
    public boolean isInputVector;
    public String labelCol;
    public String weightCol;
    public String[] categoricalCols;

    // Parameters related with algorithms.
    public int maxDepth;
    public int maxBins;
    public int minInstancesPerNode;
    public double minWeightFractionPerNode;
    public double minInfoGain;
    public int maxIter;
    public double stepSize;
    public long seed;
    public double subsamplingRate;
    public String featureSubsetStrategy;
    public double validationTol;
    public double lambda;
    public double gamma;

    // Derived parameters.
    public String lossType;
    public int maxNumLeaves;
    // useMissing is always true right now.
    public boolean useMissing;

    public GbtParams() {}
}
