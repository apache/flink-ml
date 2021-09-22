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
import org.apache.flink.table.api.Table;

/**
 * An AlgoOperator takes a list of tables as inputs and produces a list of tables as results. It can
 * be used to encode generic multi-input multi-output computation logic.
 *
 * @param <T> The class type of the AlgoOperator implementation itself.
 */
@PublicEvolving
public interface AlgoOperator<T extends AlgoOperator<T>> extends Stage<T> {
    /**
     * Applies the AlgoOperator on the given input tables and returns the result tables.
     *
     * @param inputs a list of tables
     * @return a list of tables
     */
    Table[] transform(Table... inputs);
}
