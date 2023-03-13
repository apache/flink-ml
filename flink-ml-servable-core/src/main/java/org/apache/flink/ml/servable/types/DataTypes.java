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

package org.apache.flink.ml.servable.types;

/** This class gives access to the most common types that are used to define DataFrames. */
public class DataTypes {

    public static final ScalarType BOOLEAN = new ScalarType(BasicType.BOOLEAN);

    public static final ScalarType BYTE = new ScalarType(BasicType.BYTE);

    public static final ScalarType SHORT = new ScalarType(BasicType.SHORT);

    public static final ScalarType INT = new ScalarType(BasicType.INT);

    public static final ScalarType LONG = new ScalarType(BasicType.LONG);

    public static final ScalarType FLOAT = new ScalarType(BasicType.FLOAT);

    public static final ScalarType DOUBLE = new ScalarType(BasicType.DOUBLE);

    public static final ScalarType STRING = new ScalarType(BasicType.STRING);

    public static final ScalarType BYTE_STRING = new ScalarType(BasicType.BYTE_STRING);

    public static VectorType VECTOR(BasicType elementType) {
        return new VectorType(elementType);
    }

    public static MatrixType MATRIX(BasicType elementType) {
        return new MatrixType(elementType);
    }
}
