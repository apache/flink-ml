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

package org.apache.flink.ml.common.iteration;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

/**
 * A FlatMapFunction that emits values iff the iteration's epochWatermark does not exceed a certain
 * threshold and the loss exceeds a certain tolerance.
 *
 * <p>When the output of this FlatMapFunction is used as the termination criteria of an iteration
 * body, the iteration will be executed for at most the given `maxIter` iterations. And the
 * iteration will terminate once any input value is smaller than or equal to the given `tol`.
 */
public class TerminateOnMaxIterOrTol
        implements IterationListener<Integer>, FlatMapFunction<Double, Integer> {

    private final int maxIter;

    private final double tol;

    private double loss = Double.MAX_VALUE;

    public TerminateOnMaxIterOrTol(Integer maxIter, Double tol) {
        this.maxIter = maxIter;
        this.tol = tol;
    }

    public TerminateOnMaxIterOrTol(Double tol) {
        this.maxIter = Integer.MAX_VALUE;
        this.tol = tol;
    }

    @Override
    public void flatMap(Double value, Collector<Integer> out) {
        Preconditions.checkArgument(
                Double.compare(loss, Double.MAX_VALUE) == 0,
                "Each epoch should contain only one loss value.");
        loss = value;
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<Integer> collector) {
        if ((epochWatermark + 1) < maxIter && loss > tol) {
            collector.collect(0);
        }
        loss = Double.MAX_VALUE;
    }

    @Override
    public void onIterationTerminated(Context context, Collector<Integer> collector) {}
}
