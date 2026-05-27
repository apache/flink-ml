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

package org.apache.flink.ml.linalg;

import org.apache.flink.annotation.PublicEvolving;

import java.io.Serializable;

/** A matrix of double values. */
@PublicEvolving
public interface Matrix extends Serializable {

    /** Gets number of rows. */
    int numRows();

    /** Gets number of columns. */
    int numCols();

    /** Gets value of the (i,j) element. */
    double get(int i, int j);

    /** Adds value to the (i,j) element. */
    double add(int i, int j, double value);

    /** Sets value of the (i,j) element. */
    double set(int i, int j, double value);

    /** Converts the instance to a dense matrix. */
    DenseMatrix toDense();
}
