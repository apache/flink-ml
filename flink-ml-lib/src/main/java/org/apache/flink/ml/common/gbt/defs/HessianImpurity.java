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

/**
 * The impurity introduced in XGBoost.
 *
 * <p>See: <a href="https://xgboost.readthedocs.io/en/stable/tutorials/model.html">Introduction to
 * Boosted Trees</a>.
 */
public class HessianImpurity extends Impurity {

    // Regularization of the leaf number.
    protected final double lambda;
    // Regularization of leaf scores.
    protected final double gamma;
    // Total of instance gradients.
    protected double totalGradients;
    // Total of instance hessians.
    protected double totalHessians;

    public HessianImpurity(
            double lambda,
            double gamma,
            int numInstances,
            double totalWeights,
            double totalGradients,
            double totalHessians) {
        super(numInstances, totalWeights);
        this.lambda = lambda;
        this.gamma = gamma;
        this.totalGradients = totalGradients;
        this.totalHessians = totalHessians;
    }

    @Override
    public double prediction() {
        return -totalGradients / (totalHessians + gamma);
    }

    @Override
    public double impurity() {
        if (totalHessians + lambda == 0) {
            return 0.;
        }
        return totalGradients * totalGradients / (totalHessians + lambda);
    }

    @Override
    public double gain(Impurity... others) {
        double sum = 0.;
        for (Impurity other : others) {
            sum += other.impurity();
        }
        return .5 * (sum - impurity()) - gamma;
    }

    @Override
    public HessianImpurity add(Impurity other) {
        HessianImpurity impurity = (HessianImpurity) other;
        this.numInstances += impurity.numInstances;
        this.totalWeights += impurity.totalWeights;
        this.totalGradients += impurity.totalGradients;
        this.totalHessians += impurity.totalHessians;
        return this;
    }

    @Override
    public HessianImpurity subtract(Impurity other) {
        HessianImpurity impurity = (HessianImpurity) other;
        this.numInstances -= impurity.numInstances;
        this.totalWeights -= impurity.totalWeights;
        this.totalGradients -= impurity.totalGradients;
        this.totalHessians -= impurity.totalHessians;
        return this;
    }

    public void add(int numInstances, double weights, double gradients, double hessians) {
        this.numInstances += numInstances;
        this.totalWeights += weights;
        this.totalGradients += gradients;
        this.totalHessians += hessians;
    }

    public void subtract(int numInstances, double weights, double gradients, double hessians) {
        this.numInstances -= numInstances;
        this.totalWeights -= weights;
        this.totalGradients -= gradients;
        this.totalHessians -= hessians;
    }
}
