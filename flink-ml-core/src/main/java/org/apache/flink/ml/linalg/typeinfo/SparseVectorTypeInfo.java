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

package org.apache.flink.ml.linalg.typeinfo;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.ml.linalg.SparseVector;

/** A {@link TypeInformation} for the {@link SparseVector} type. */
public class SparseVectorTypeInfo extends TypeInformation<SparseVector> {
    public static final SparseVectorTypeInfo INSTANCE = new SparseVectorTypeInfo();

    @Override
    public boolean isBasicType() {
        return false;
    }

    @Override
    public boolean isTupleType() {
        return false;
    }

    @Override
    public int getArity() {
        return 3;
    }

    @Override
    public int getTotalFields() {
        return 3;
    }

    @Override
    public Class<SparseVector> getTypeClass() {
        return SparseVector.class;
    }

    @Override
    public boolean isKeyType() {
        return false;
    }

    @Override
    public TypeSerializer<SparseVector> createSerializer(ExecutionConfig executionConfig) {
        return SparseVectorSerializer.INSTANCE;
    }

    @Override
    public String toString() {
        return "SparseVectorType";
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof SparseVectorTypeInfo;
    }

    @Override
    public int hashCode() {
        return getClass().hashCode();
    }

    @Override
    public boolean canEqual(Object obj) {
        return obj instanceof SparseVectorTypeInfo;
    }
}
