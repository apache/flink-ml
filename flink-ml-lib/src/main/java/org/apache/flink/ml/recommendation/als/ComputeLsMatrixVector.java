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

package org.apache.flink.ml.recommendation.als;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.ps.iterations.ProcessStage;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.NormalEquationSolver;
import org.apache.flink.ml.recommendation.als.Als.Ratings;

import it.unimi.dsi.fastutil.longs.LongOpenHashSet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * An iteration stage that uses the pulled model values and batch data to compute the least square
 * matrices and vectors.
 */
public class ComputeLsMatrixVector extends ProcessStage<AlsMLSession> {
    private final int rank;
    private final int matrixOffset;
    private final boolean implicit;
    private final double lambda;
    private final double alpha;

    public ComputeLsMatrixVector(int rank, boolean implicit, double lambda, double alpha) {
        this.rank = rank;
        this.matrixOffset = rank * rank + rank;
        this.implicit = implicit;
        this.lambda = lambda;
        this.alpha = alpha;
    }

    @Override
    public void process(AlsMLSession session) throws IOException {
        session.log(getClass().getSimpleName(), true);
        if (session.allReduceBuffer != null
                && BLAS.norm2(new DenseVector(session.allReduceBuffer[0])) != 0.0) {
            if (session.yty == null) {
                session.yty = new double[rank * rank];
            }
            System.arraycopy(session.allReduceBuffer[0], 0, session.yty, 0, rank * rank);
        }

        List<Tuple2<Long, double[]>> matrices =
                session.batchData.hasHotPoint ? computeMatrices(session, rank) : new ArrayList<>();

        session.pushIndices.clear();
        session.pushValues.clear();
        session.pullIndices.clear();
        session.pullValues.clear();

        if (matrices.size() == 0) {
            session.pushIndices.add(Long.MIN_VALUE + 1);
            session.pushValues.size(matrixOffset);

            session.pullIndices.add(Long.MIN_VALUE + 1);
            session.pullValues.size(matrixOffset);
        } else {
            for (Tuple2<Long, double[]> matrix : matrices) {
                session.pushIndices.add(matrix.f0);
                session.pushValues.addAll(matrix.f1);
            }
            LongOpenHashSet pullSet = new LongOpenHashSet();
            for (Ratings r : session.batchData.ratingsList) {
                if (r.isMainNode) {
                    pullSet.add(r.nodeId);
                }
            }
            session.pullIndices.size(pullSet.size());
            Iterator<Long> iter = pullSet.iterator();
            for (int i = 0; i < pullSet.size(); ++i) {
                session.pullIndices.elements()[i] = -iter.next() - 1;
            }
            session.pullValues.size(session.pullIndices.size() * matrixOffset);
        }

        AlsMLSession.LOG.info(
                String.format(
                        "Worker-%d mat vec pull size %d%n",
                        session.workerId, session.pushIndices.elements()[0]));

        session.log(getClass().getSimpleName(), false);
    }

    private List<Tuple2<Long, double[]>> computeMatrices(AlsMLSession session, int rank) {
        NormalEquationSolver ls = new NormalEquationSolver(rank);
        List<Tuple2<Long, double[]>> matvec = new ArrayList<>();
        double[] tmp = new double[rank];
        /* loops over local nodes. */
        for (Ratings ele : session.batchData.ratingsList) {
            if (!ele.isSplit) {
                continue;
            }
            double[] ret = new double[matrixOffset];
            /* solves an lease square problem. */
            ls.reset();

            if (!implicit) {
                long[] nb = ele.neighbors;
                double[] rating = ele.scores;
                for (int i = 0; i < nb.length; i++) {
                    long index = nb[i];
                    int realIndex = session.reusedNeighborIndexMapping.get(index);
                    System.arraycopy(session.pullValues.elements(), realIndex * rank, tmp, 0, rank);

                    ls.add(new DenseVector(tmp), rating[i], 1.0);
                }
                ls.regularize(nb.length * lambda);
            } else {
                if (ele.isMainNode) {
                    ls.merge(new DenseMatrix(rank, rank, session.yty));
                }
                int numExplicit = 0;
                long[] nb = ele.neighbors;
                double[] rating = ele.scores;

                for (int i = 0; i < nb.length; i++) {
                    long index = nb[i];
                    double r = rating[i];
                    double c1 = 0.;

                    if (r > 0) {
                        numExplicit++;
                        c1 = alpha * r;
                    }
                    int realIndex = session.reusedNeighborIndexMapping.get(index);
                    System.arraycopy(session.pullValues.elements(), realIndex * rank, tmp, 0, rank);

                    ls.add(new DenseVector(tmp), ((r > 0.0) ? (1.0 + c1) : 0.0), c1);
                }

                numExplicit = Math.max(numExplicit, 1);
                ls.regularize(numExplicit * lambda);
            }
            System.arraycopy(ls.getAta().values, 0, ret, 0, rank * rank);
            System.arraycopy(ls.getAtb().values, 0, ret, rank * rank, rank);
            matvec.add(Tuple2.of(-1 - ele.nodeId, ret));
        }
        return matvec;
    }
}
