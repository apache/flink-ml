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

/** Configurations for {@link org.apache.flink.ml.common.gbt.GBTRunner}. */
public class BoostingStrategy implements Serializable {

    /** Indicates the task is classification or regression. */
    public TaskType taskType;

    /**
     * Indicates whether the features are in one column of vector type or multiple columns of
     * non-vector types.
     */
    public boolean isInputVector;

    /**
     * Names of features columns used for training. Contains only 1 column name when `isInputVector`
     * is `true`.
     */
    public String[] featuresCols;

    /** Name of label column. */
    public String labelCol;

    /**
     * Names of columns which should be treated as categorical features, when `isInputVector` is
     * `false`.
     */
    public String[] categoricalCols;

    /**
     * Max depth of the tree (root node is the 1st level). Depth 1 means 1 leaf node, i.e., the root
     * node; Depth 2 means 1 internal node + 2 leaf nodes; etc.
     */
    public int maxDepth;

    /** Maximum number of bins used for discretizing continuous features. */
    public int maxBins;

    /**
     * Minimum number of instances each node must have. If a split causes the left or right child to
     * have fewer instances than minInstancesPerNode, the split is invalid.
     */
    public int minInstancesPerNode;

    /**
     * Minimum fraction of the weighted sample count that each node must have. If a split causes the
     * left or right child to have a smaller fraction of the total weight than
     * minWeightFractionPerNode, the split is invalid.
     *
     * <p>NOTE: Weight column is not supported right now, so all samples have equal weights.
     */
    public double minWeightFractionPerNode;

    /** Minimum information gain for a split to be considered valid. */
    public double minInfoGain;

    /** Maximum number of iterations of boosting, i.e. the number of trees in the final model. */
    public int maxIter;

    /** Step size for shrinking the contribution of each estimator. */
    public double stepSize;

    /** The random seed used in samples/features subsampling. */
    public long seed;

    /** Fraction of the training data used for learning one tree. */
    public double subsamplingRate;

    /**
     * Fraction of the training data used for learning one tree. Supports "auto", "all", "onethird",
     * "sqrt", "log2", (0.0 - 1.0], and [1 - n].
     */
    public String featureSubsetStrategy;

    /** Regularization term for the number of leaves. */
    public double regLambda;

    /** L2 regularization term for the weights of leaves. */
    public double regGamma;

    /** The type of loss used in boosting. */
    public LossType lossType;

    // Derived parameters.
    /** Maximum number leaves. */
    public int maxNumLeaves;
    /** Whether to consider missing values in the model. Always `true` right now. */
    public boolean useMissing;

    public BoostingStrategy() {}
}
