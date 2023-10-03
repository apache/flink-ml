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

/** A node used in learning procedure. */
public class LearningNode implements Serializable {

    // The node index in `currentTreeNodes` used in `PostSplitsOperator`.
    public int nodeIndex;
    // Slice of indices of bagging instances.
    public Slice slice = new Slice();
    // Slice of indices of non-bagging instances.
    public Slice oob = new Slice();
    // Depth of corresponding tree node.
    public int depth;

    public LearningNode() {}

    public LearningNode(int nodeIndex, Slice slice, Slice oob, int depth) {
        this.nodeIndex = nodeIndex;
        this.slice = slice;
        this.oob = oob;
        this.depth = depth;
    }

    @Override
    public String toString() {
        return String.format(
                "LearningNode{nodeIndex=%s, slice=%s, oob=%s, depth=%d}",
                nodeIndex, slice, oob, depth);
    }
}
