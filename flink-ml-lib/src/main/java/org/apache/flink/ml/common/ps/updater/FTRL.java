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

package org.apache.flink.ml.common.ps.updater;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * The FTRL (Follow-the-Regularized-Leader) algorithm is an optimization algorithm used for
 * large-scale linear models. It aims to minimize the sum of a loss function over all training
 * examples, subject to a regularization constraint.
 *
 * <p>FTRL is well-suited for sparse data and can handle problems with billions of features.
 */
public class FTRL implements ModelUpdater<Tuple3<Long, Long, double[]>> {
    private final double alpha;
    private final double beta;
    private final double lambda1;
    private final double lambda2;

    // ------ Model data of FTRL optimizer. -----
    private long startIndex;
    private long endIndex;
    private double[] weight;
    private double[] sigma;
    private double[] z;
    private double[] n;

    private ListState<Long> boundaryState;
    private ListState<double[]> modelDataState;

    public FTRL(double alpha, double beta, double lambda1, double lambda2) {
        this.alpha = alpha;
        this.beta = beta;
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
    }

    @Override
    public void open(long startKeyIndex, long endKeyIndex) {
        this.startIndex = startKeyIndex;
        this.endIndex = endKeyIndex;
        int modelShardSize = (int) (endIndex - startIndex);
        weight = new double[modelShardSize];
        sigma = new double[modelShardSize];
        z = new double[modelShardSize];
        n = new double[modelShardSize];
    }

    @Override
    public void update(long[] keys, double[] values) {
        for (int i = 0; i < keys.length; i++) {
            int index = (int) (keys[i] - startIndex);
            double gi = values[i];
            updateModelOnOneDim(gi, index, weight);
        }
    }

    private void updateModelOnOneDim(double gi, int index, double[] weight) {
        double gigi = gi * gi;
        sigma[index] = 1 / alpha * (Math.sqrt(n[index] + gigi) - Math.sqrt(n[index]));
        z[index] += gi - sigma[index] * weight[index];
        n[index] += gigi;

        if (Math.abs(z[index]) <= lambda1) {
            weight[index] = 0;
        } else {
            weight[index] =
                    -(z[index] - Math.signum(z[index]) * lambda1)
                            / ((beta + Math.sqrt(n[index])) / alpha + lambda2);
        }
    }

    @Override
    public double[] get(long[] keys) {
        double[] values = new double[keys.length];
        for (int i = 0; i < keys.length; i++) {
            values[i] = weight[(int) (keys[i] - startIndex)];
        }
        return values;
    }

    @Override
    public Iterator<Tuple3<Long, Long, double[]>> getModelSegments() {
        List<Tuple3<Long, Long, double[]>> modelSegments = new ArrayList<>();
        modelSegments.add(Tuple3.of(startIndex, endIndex, weight));
        return modelSegments.iterator();
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        boundaryState =
                context.getOperatorStateStore()
                        .getListState(new ListStateDescriptor<>("BoundaryState", Types.LONG));

        Iterator<Long> iterator = boundaryState.get().iterator();
        if (iterator.hasNext()) {
            startIndex = iterator.next();
            endIndex = iterator.next();
        }

        modelDataState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "modelDataState",
                                        PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO));
        Iterator<double[]> modelData = modelDataState.get().iterator();
        if (modelData.hasNext()) {
            weight = modelData.next();
            sigma = modelData.next();
            z = modelData.next();
            n = modelData.next();
        }
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        if (weight != null) {
            boundaryState.clear();
            boundaryState.add(startIndex);
            boundaryState.add(endIndex);

            modelDataState.clear();
            modelDataState.add(weight);
            modelDataState.add(sigma);
            modelDataState.add(z);
            modelDataState.add(n);
        }
    }
}
