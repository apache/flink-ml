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

package org.apache.flink.ml.common.feature;

import org.apache.flink.api.java.tuple.Tuple2;

/** A data point to represent values that use long as index and double as values. */
public class LabeledLargePointWithWeight {
    public Tuple2<long[], double[]> features;

    public double label;

    public double weight;

    public LabeledLargePointWithWeight(
            Tuple2<long[], double[]> features, double label, double weight) {
        this.features = features;
        this.label = label;
        this.weight = weight;
    }

    /** Makes it pojo to use flink serializer. */
    public LabeledLargePointWithWeight() {}
}
