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

package org.apache.flink.ml.servable.api;

import org.apache.flink.annotation.PublicEvolving;

import java.io.IOException;
import java.io.InputStream;

/**
 * A ModelServable is a TransformerServable with the extra API to set model data.
 *
 * @param <T> The class type of the ModelServable implementation itself.
 */
@PublicEvolving
public interface ModelServable<T extends ModelServable<T>> extends TransformerServable<T> {

    /** Sets model data using the serialized model data from the given input streams. */
    default T setModelData(InputStream... modelDataInputs) throws IOException {
        throw new UnsupportedOperationException("This operation is not supported.");
    }
}
