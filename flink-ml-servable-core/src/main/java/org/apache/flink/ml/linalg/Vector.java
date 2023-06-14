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
import org.apache.flink.api.common.typeinfo.TypeInfo;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfoFactory;

import java.io.Serializable;

/** A vector of numerical values. */
@TypeInfo(VectorTypeInfoFactory.class)
@PublicEvolving
public interface Vector<I extends Number, V extends Number, IArray, VArray> extends Serializable {

    /** Gets the size of the vector. */
    long size();

    /** Gets the value of the ith element. */
    V get(I i);

    /** Sets the value of the ith element. */
    void set(I i, V value);

    /** Converts the instance to a double array. */
    VArray toArray();

    /** Converts the instance to a dense vector. */
    DenseVector<I, V, IArray, VArray> toDense();

    /** Converts the instance to a sparse vector. */
    SparseVector<I, V, IArray, VArray> toSparse();

    /** Makes a deep copy of the vector. */
    Vector<I, V, IArray, VArray> clone();
}
