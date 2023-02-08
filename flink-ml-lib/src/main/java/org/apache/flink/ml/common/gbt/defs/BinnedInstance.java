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

import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizer;
import org.apache.flink.ml.feature.stringindexer.StringIndexer;
import org.apache.flink.ml.linalg.SparseVector;

import org.eclipse.collections.impl.map.mutable.primitive.IntIntHashMap;

/**
 * Represents an instance including binned values of all features, weight, and label.
 *
 * <p>Categorical and continuous features are mapped to integers by {@link StringIndexer} and {@link
 * KBinsDiscretizer}, respectively. Null values (`null` or `Double.NaN`) are also mapped to certain
 * integers.
 *
 * <p>NOTE: When the input features are sparse, i.e., from {@link SparseVector}s, unseen indices are
 * not stored in `features`. They should be handled separately.
 */
public class BinnedInstance {

    public IntIntHashMap features;
    public double weight;
    public double label;

    public BinnedInstance() {}

    public BinnedInstance(IntIntHashMap features, double weight, double label) {
        this.weight = weight;
        this.label = label;
        this.features = features;
    }

    @Override
    public String toString() {
        return String.format(
                "BinnedInstance{features=%s, weight=%s, label=%s}", features, weight, label);
    }
}
