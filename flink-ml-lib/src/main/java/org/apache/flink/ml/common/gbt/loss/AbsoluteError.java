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

package org.apache.flink.ml.common.gbt.loss;

/**
 * Squared error loss function defined as |y - pred| where y and pred are label and predictions for
 * the instance respectively.
 */
public class AbsoluteError implements Loss {

    public static final AbsoluteError INSTANCE = new AbsoluteError();

    private AbsoluteError() {}

    @Override
    public double loss(double pred, double y) {
        double error = y - pred;
        return Math.abs(error);
    }

    @Override
    public double gradient(double pred, double y) {
        return y > pred ? -1. : 1;
    }

    @Override
    public double hessian(double pred, double y) {
        return 0.;
    }
}
