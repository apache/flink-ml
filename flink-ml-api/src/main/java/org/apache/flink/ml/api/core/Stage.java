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

package org.apache.flink.ml.api.core;

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.ml.param.WithParams;

import java.io.IOException;
import java.io.Serializable;

/**
 * Base class for a node in a Pipeline or Graph. The interface is only a concept, and does not have
 * any actual functionality. Its subclasses could be Estimator, Model, Transformer or AlgoOperator.
 * No other classes should inherit this interface directly.
 *
 * <p>Each stage is with parameters, and requires a public empty constructor for restoration.
 *
 * @param <T> The class type of the Stage implementation itself.
 */
@PublicEvolving
public interface Stage<T extends Stage<T>> extends WithParams<T>, Serializable {
    /** Saves this stage to the given path. */
    void save(String path) throws IOException;

    // NOTE: every Stage subclass should implement a static method with signature "static T
    // load(String path)", where T refers to the concrete subclass. This static method should
    // instantiate a new stage instance based on the data from the given path.
}
