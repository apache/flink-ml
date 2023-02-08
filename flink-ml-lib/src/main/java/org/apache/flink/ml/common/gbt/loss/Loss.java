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

import java.io.Serializable;

/** Loss functions for gradient boosting algorithms. */
public interface Loss extends Serializable {

    /**
     * Calculates loss given pred and y.
     *
     * @param pred prediction value.
     * @param y label value.
     * @return loss value.
     */
    double loss(double pred, double y);

    /**
     * Calculates value of gradient given prediction and label.
     *
     * @param pred prediction value.
     * @param y label value.
     * @return the value of gradient.
     */
    double gradient(double pred, double y);

    /**
     * Calculates value of second derivative, i.e. hessian, given prediction and label.
     *
     * @param pred prediction value.
     * @param y label value.
     * @return the value of second derivative, i.e. hessian.
     */
    double hessian(double pred, double y);
}
