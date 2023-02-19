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

package org.apache.flink.ml.api;

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.ml.servable.api.TransformerServable;

/**
 * A Transformer is an AlgoOperator with the semantic difference that it encodes the Transformation
 * logic, such that a record in the output typically corresponds to one record in the input. In
 * contrast, an AlgoOperator is a better fit to express aggregation logic where a record in the
 * output could be computed from an arbitrary number of records in the input.
 *
 * <p>NOTE: If a Transformer has a corresponding {@link TransformerServable}, it should implement a
 * static method with the signature {@code static T loadServable(String path)}, where {@code T}
 * refers to the concrete subclass of {@link TransformerServable}. This static method should
 * instantiate a new {@link TransformerServable} instance based on the data read from the given
 * path.
 *
 * @param <T> The class type of the Transformer implementation itself.
 */
@PublicEvolving
public interface Transformer<T extends Transformer<T>> extends AlgoOperator<T> {}
