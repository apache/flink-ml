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
import org.apache.flink.ml.param.WithParams;

/**
 * A TransformerServable takes a DataFrame as input and produces a DataFrame as the result. It can
 * be used to encode online inference computation logic.
 *
 * <p>NOTE: Every TransformerServable subclass should have a no-arg constructor.
 *
 * <p>NOTE: Every TransformerServable subclass should implement a static method with signature
 * {@code static T load(String path) throws IOException;}, where {@code T} refers to the concrete
 * subclass. This static method should instantiate a new TransformerServable instance based on the
 * data read from the given path.
 *
 * @param <T> The class type of the TransformerServable implementation itself.
 */
@PublicEvolving
public interface TransformerServable<T extends TransformerServable<T>> extends WithParams<T> {
    /**
     * Applies the TransformerServable on the given input DataFrame and returns the result
     * DataFrame.
     *
     * @param input the input data
     * @return the result data
     */
    DataFrame transform(DataFrame input);
}
