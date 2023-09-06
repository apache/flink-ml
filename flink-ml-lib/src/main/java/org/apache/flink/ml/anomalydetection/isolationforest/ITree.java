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

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.linalg.DenseVector;

import java.io.Serializable;
import java.util.Random;

/** Construct isolation tree. */
public class ITree implements Serializable {
    public final int attributeIndex;
    public final double splitAttributeValue;
    public ITree leftTree;
    public ITree rightTree;
    public int currentHeight;
    public int leafNodesNum;

    public ITree(int attributeIndex, double splitAttributeValue) {
        this.attributeIndex = attributeIndex;
        this.splitAttributeValue = splitAttributeValue;
        this.leftTree = null;
        this.rightTree = null;
        this.currentHeight = 0;
        this.leafNodesNum = 1;
    }

    public static ITree generateIsolationTree(
            DenseVector[] samplesData,
            int currentHeight,
            int limitHeight,
            Random randomState,
            int[] featureIndices) {
        int n = samplesData.length;
        ITree isolationTree;
        if (samplesData.length == 0) {
            return null;
        } else if (samplesData.length == 1 || currentHeight >= limitHeight) {
            isolationTree = new ITree(0, samplesData[0].get(0));
            isolationTree.currentHeight = currentHeight;
            isolationTree.leafNodesNum = samplesData.length;
            return isolationTree;
        }
        boolean flag = true;
        for (int i = 1; i < n; i++) {
            if (!samplesData[i].equals(samplesData[i - 1])) {
                flag = false;
                break;
            }
        }
        if (flag) {
            isolationTree = new ITree(0, samplesData[0].get(0));
            isolationTree.currentHeight = currentHeight;
            isolationTree.leafNodesNum = samplesData.length;
            return isolationTree;
        }

        Tuple2<Integer, Double> tuple2 =
                getRandomFeatureToSplit(samplesData, randomState, featureIndices);
        int attributeIndex = tuple2.f0;
        double splitAttributeValue = tuple2.f1;

        int leftNodesNum = 0;
        int rightNodesNum = 0;
        for (DenseVector datum : samplesData) {
            if (datum.get(attributeIndex) < splitAttributeValue) {
                leftNodesNum++;
            } else {
                rightNodesNum++;
            }
        }

        DenseVector[] leftSamples = new DenseVector[leftNodesNum];
        DenseVector[] rightSamples = new DenseVector[rightNodesNum];
        int l = 0, r = 0;
        for (DenseVector samplesDatum : samplesData) {
            if (samplesDatum.get(attributeIndex) < splitAttributeValue) {
                leftSamples[l++] = samplesDatum;
            } else {
                rightSamples[r++] = samplesDatum;
            }
        }

        ITree root = new ITree(attributeIndex, splitAttributeValue);
        root.currentHeight = currentHeight;
        root.leafNodesNum = samplesData.length;
        root.leftTree =
                generateIsolationTree(
                        leftSamples, currentHeight + 1, limitHeight, randomState, featureIndices);
        root.rightTree =
                generateIsolationTree(
                        rightSamples, currentHeight + 1, limitHeight, randomState, featureIndices);

        return root;
    }

    private static Tuple2<Integer, Double> getRandomFeatureToSplit(
            DenseVector[] samplesData, Random randomState, int[] featureIndices) {
        int attributeIndex = featureIndices[randomState.nextInt(featureIndices.length)];

        double maxValue = samplesData[0].get(attributeIndex);
        double minValue = samplesData[0].get(attributeIndex);
        for (int i = 1; i < samplesData.length; i++) {
            minValue = Math.min(minValue, samplesData[i].get(attributeIndex));
            maxValue = Math.max(maxValue, samplesData[i].get(attributeIndex));
        }
        double splitAttributeValue = (maxValue - minValue) * randomState.nextDouble() + minValue;

        return Tuple2.of(attributeIndex, splitAttributeValue);
    }

    public static double calculatePathLength(DenseVector sampleData, ITree isolationTree)
            throws Exception {
        double pathLength = -1;
        ITree tmpITree = isolationTree;
        while (tmpITree != null) {
            pathLength += 1;
            if (tmpITree.leftTree == null
                    || tmpITree.rightTree == null
                    || sampleData.get(tmpITree.attributeIndex) == tmpITree.splitAttributeValue) {
                break;
            } else if (sampleData.get(tmpITree.attributeIndex) < tmpITree.splitAttributeValue) {
                tmpITree = tmpITree.leftTree;
            } else {
                tmpITree = tmpITree.rightTree;
            }
        }

        assert tmpITree != null;
        return pathLength + calculateCn(tmpITree.leafNodesNum);
    }

    public static double calculateCn(double n) {
        if (n <= 1) {
            return 0;
        }
        return 2.0 * (Math.log(n - 1.0) + 0.5772156649015329) - 2.0 * (n - 1.0) / n;
    }
}
