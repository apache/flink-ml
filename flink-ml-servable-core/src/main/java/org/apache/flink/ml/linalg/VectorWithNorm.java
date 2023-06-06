/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.flink.ml.linalg;

import org.apache.flink.api.common.typeinfo.TypeInfo;
import org.apache.flink.ml.linalg.typeinfo.VectorWithNormTypeInfoFactory;

/** A vector with its norm. */
@TypeInfo(VectorWithNormTypeInfoFactory.class)
public class VectorWithNorm {
    public final IntDoubleVector vector;

    public final double l2Norm;

    public VectorWithNorm(IntDoubleVector vector) {
        this(vector, BLAS.norm2(vector));
    }

    public VectorWithNorm(IntDoubleVector vector, double l2Norm) {
        this.vector = vector;
        this.l2Norm = l2Norm;
    }
}
