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

package org.apache.flink.ml.classification.naivebayes;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.flink.api.java.tuple.Tuple3;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * The model data of {@link NaiveBayesModel}.
 */
public class NaiveBayesModelData implements Serializable {
    private static final long serialVersionUID = 3919917903722286395L;
    public String[] featureNames;
    public Number[][][] theta;
    public double[] piArray;
    public double[] labelWeights;
    public Object[] label;
    public boolean[] isCate;
    public double[][] weightSum;

    public void generateWeightAndNumbers(List <Tuple3 <Object, Double[], Map <Integer, Double>[]>> arrayData) {
        int arrayLength = arrayData.size();
        int featureNumber = arrayData.get(0).f1.length;
        weightSum = new double[arrayLength][featureNumber];
        for (int i = 0; i < arrayLength; i++) {
            weightSum[i] = ArrayUtils.toPrimitive(arrayData.get(i).f1);
        }
    }
}
