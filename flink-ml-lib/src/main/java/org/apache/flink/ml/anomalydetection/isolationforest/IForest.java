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

package org.apache.flink.ml.anomalydetection.isolationforest;

import org.apache.flink.ml.linalg.DenseVector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Construct isolation forest. */
public class IForest implements Serializable {
    public final int numTrees;
    public List<ITree> iTreeList;
    public Double center0;
    public Double center1;
    public int subSamplesSize;

    public IForest(int numTrees) {
        this.numTrees = numTrees;
        this.iTreeList = new ArrayList<>(256);
        this.center0 = null;
        this.center1 = null;
    }

    public void generateIsolationForest(DenseVector[] samplesData, int[] featureIndices) {
        int n = samplesData.length;
        subSamplesSize = Math.min(256, n);
        int limitHeight = (int) Math.ceil(Math.log(Math.max(subSamplesSize, 2)) / Math.log(2));

        Random randomState = new Random(System.nanoTime());
        for (int i = 0; i < numTrees; i++) {
            DenseVector[] subSamples = new DenseVector[subSamplesSize];
            for (int j = 0; j < subSamplesSize; j++) {
                int r = randomState.nextInt(n);
                subSamples[j] = samplesData[r];
            }
            ITree isolationTree =
                    ITree.generateIsolationTree(
                            subSamples, 0, limitHeight, randomState, featureIndices);
            this.iTreeList.add(isolationTree);
        }
    }

    public DenseVector calculateScore(DenseVector[] samplesData) throws Exception {
        DenseVector score = new DenseVector(samplesData.length);

        for (int i = 0; i < samplesData.length; i++) {
            double pathLengthSum = 0;
            for (ITree isolationTree : iTreeList) {
                pathLengthSum += ITree.calculatePathLength(samplesData[i], isolationTree);
            }

            double pathLengthAvg = pathLengthSum / iTreeList.size();
            double cn = ITree.calculateCn(subSamplesSize);
            double index = pathLengthAvg / cn;
            score.set(i, Math.pow(2, -index));
        }

        return score;
    }

    public DenseVector classifyByCluster(DenseVector score, int iters) {
        int scoresSize = score.size();

        center0 = score.get(0); // Cluster center of abnormal
        center1 = score.get(0); // Cluster center of normal

        for (double s : score.values) {
            if (s > center0) {
                center0 = s;
            }

            if (s < center1) {
                center1 = s;
            }
        }

        int cnt0;
        int cnt1;
        double diff0;
        double diff1;
        int[] labels = new int[scoresSize];

        for (int i = 0; i < iters; i++) {
            cnt0 = 0;
            cnt1 = 0;

            for (int j = 0; j < scoresSize; j++) {
                diff0 = Math.abs(score.get(j) - center0);
                diff1 = Math.abs(score.get(j) - center1);

                if (diff0 < diff1) {
                    labels[j] = 0;
                    cnt0++;
                } else {
                    labels[j] = 1;
                    cnt1++;
                }
            }

            diff0 = center0;
            diff1 = center1;

            center0 = 0.0;
            center1 = 0.0;

            for (int k = 0; k < scoresSize; k++) {
                if (labels[k] == 0) {
                    center0 += score.get(k);
                } else {
                    center1 += score.get(k);
                }
            }

            center0 /= cnt0;
            center1 /= cnt1;

            if (center0 - diff0 <= 1e-6 && center1 - diff1 <= 1e-6) {
                break;
            }
        }

        return new DenseVector(new double[] {center0, center1});
    }
}
