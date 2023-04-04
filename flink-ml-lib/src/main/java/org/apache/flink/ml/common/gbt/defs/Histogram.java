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

import org.apache.flink.api.common.typeinfo.TypeInfo;
import org.apache.flink.ml.common.gbt.typeinfo.HistogramTypeInfoFactory;
import org.apache.flink.util.Preconditions;

import java.io.Serializable;

/**
 * This class stores values of histogram bins.
 *
 * <p>Note that only the part of {@link Histogram#hists} specified by {@link Histogram#slice} is
 * valid.
 */
@TypeInfo(HistogramTypeInfoFactory.class)
public class Histogram implements Serializable {
    // Stores values of histogram bins.
    public double[] hists;
    // Stores the valid slice of `hists`.
    public Slice slice = new Slice();

    public Histogram() {}

    public Histogram(double[] hists, Slice slice) {
        this.hists = hists;
        this.slice = slice;
    }

    public Histogram accumulate(Histogram other) {
        Preconditions.checkArgument(slice.size() == other.slice.size());
        for (int i = 0; i < slice.size(); i += 1) {
            hists[slice.start + i] += other.hists[other.slice.start + i];
        }
        return this;
    }
}
