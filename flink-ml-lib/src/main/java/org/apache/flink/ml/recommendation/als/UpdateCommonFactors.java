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

import org.apache.flink.ml.common.ps.iterations.ProcessStage;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.NormalEquationSolver;
import org.apache.flink.ml.recommendation.als.Als.Ratings;

import java.io.IOException;
import java.util.Arrays;

/** An iteration stage that uses the pulled model values and batch data to compute the factors. */
public class UpdateCommonFactors extends ProcessStage<AlsMLSession> {
    private final int rank;
    private final boolean implicit;
    private final boolean nonNegative;
    private final double lambda;
    private final double alpha;

    public UpdateCommonFactors(
            int rank, boolean implicit, boolean nonNegative, double lambda, double alpha) {
        this.rank = rank;
        this.implicit = implicit;
        this.nonNegative = nonNegative;
        this.lambda = lambda;
        this.alpha = alpha;
    }

    @Override
    public void process(AlsMLSession session) throws IOException {
        session.log(this.getClass().getSimpleName(), true);
        if (session.allReduceBuffer != null
                && BLAS.norm2(new DenseVector(session.allReduceBuffer[0])) > 0.0) {
            if (session.yty == null) {
                session.yty = new double[rank * rank];
            }
            System.arraycopy(session.allReduceBuffer[0], 0, session.yty, 0, rank * rank);
        }
        session.pushIndices.clear();
        session.pushValues.clear();
        if (session.batchData.numCommonNodeIds == 0) {
            session.pushIndices.add(Long.MIN_VALUE);
            session.pushValues.size(rank);
            return;
        } else {
            session.pushIndices.size(session.batchData.numCommonNodeIds);
            session.pushValues.size(rank * session.pushIndices.size());
            updatedFactorsWithNeighbors(session, rank);
        }

        AlsMLSession.LOG.info(
                String.format(
                        "Worker-%d push size %d", session.workerId, session.pushIndices.size()));
        session.log(this.getClass().getSimpleName(), false);
    }

    private void updatedFactorsWithNeighbors(AlsMLSession session, int rank) {

        NormalEquationSolver ls = new NormalEquationSolver(rank);

        double[] tmpVec = new double[rank];
        DenseVector x = new DenseVector(rank);

        int nonSplitId = 0;

        for (Ratings ele : session.batchData.ratingsList) {
            if (ele.isSplit) {
                continue;
            }

            ls.reset();
            Arrays.fill(x.values, 0);

            if (!implicit) {
                long[] nb = ele.neighbors;
                double[] rating = ele.scores;
                for (int i = 0; i < nb.length; i++) {
                    long index = nb[i];
                    int realIndex = session.reusedNeighborIndexMapping.get(index);
                    System.arraycopy(
                            session.pullValues.elements(), realIndex * rank, tmpVec, 0, rank);
                    ls.add(new DenseVector(tmpVec), rating[i], 1.0);
                }
                ls.regularize(nb.length * lambda);
                ls.solve(x, true);
            } else {
                ls.merge(new DenseMatrix(rank, rank, session.yty));

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
                    System.arraycopy(
                            session.pullValues.elements(), realIndex * rank, tmpVec, 0, rank);

                    ls.add(new DenseVector(tmpVec), ((r > 0.0) ? (1.0 + c1) : 0.0), c1);
                }

                numExplicit = Math.max(numExplicit, 1);
                ls.regularize(numExplicit * lambda);
                ls.solve(x, nonNegative);
            }
            session.pushIndices.elements()[nonSplitId] = ele.nodeId;
            System.arraycopy(x.values, 0, session.pushValues.elements(), nonSplitId * rank, rank);
            nonSplitId++;
        }
    }
}
