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

import java.io.Serializable;
import java.util.Map;

/**
 * The model data of {@link NaiveBayesModel}.
 */
public class NaiveBayesModelData implements Serializable {
    private static final long serialVersionUID = 3919917903722286395L;
    public final String[] featureNames;
    public final Map<Object, Double>[][] theta;
    public final double[] piArray;
    public final Object[] label;

    public NaiveBayesModelData(String[] featureNames, Map<Object, Double>[][] theta, double[] piArray, Object[] label) {
        this.featureNames = featureNames;
        this.theta = theta;
        this.piArray = piArray;
        this.label = label;
    }
}
