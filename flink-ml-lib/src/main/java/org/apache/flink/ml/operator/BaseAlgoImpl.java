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

package org.apache.flink.ml.operator;

import org.apache.flink.ml.api.core.AlgoOperator;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.WithParams;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.table.api.Table;

import java.util.HashMap;
import java.util.Map;

/**
 * Base class for algorithm operators.
 *
 * <p>Base class for the algorithm operators. It hosts the parameters and output tables of an
 * algorithm operator. Each BaseAlgoImpl may have one or more output tables. One of the output table
 * is the primary output table which can be obtained by calling {@link #getOutput}. The other output
 * tables are side output tables that can be obtained by calling {@link #getSideOutputs()}.
 *
 * <p>The input of an BaseAlgoImpl is defined in the subclasses of the BaseAlgoImpl.
 *
 * @param <T> The class type of the {@link BaseAlgoImpl} implementation itself
 */
public abstract class BaseAlgoImpl<T extends BaseAlgoImpl<T>>
        implements AlgoOperator<T>, WithParams<T> {

    /** Params for algorithms. */
    private Map<Param<?>, Object> params;

    /** The table held by operator. */
    private transient Table output = null;

    /** The side outputs of operator that be similar to the stream's side outputs. */
    private transient Table[] sideOutputs = null;

    /** Construct the operator with the initial Params. */
    protected BaseAlgoImpl(Map<Param<?>, Object> params) {
        this.params = new HashMap<>();
        if (null != params) {
            for (Map.Entry<Param<?>, Object> entry : params.entrySet()) {
                this.params.put(entry.getKey(), entry.getValue());
            }
        }
        ParamUtils.initializeMapWithDefaultValues(this.params, this);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return this.params;
    }

    /** Returns the table held by operator. */
    public Table getOutput() {
        return this.output;
    }

    /** Returns the side outputs. */
    public Table[] getSideOutputs() {
        return this.sideOutputs;
    }

    /**
     * Set the table held by operator.
     *
     * @param output the output table.
     */
    protected void setOutput(Table output) {
        this.output = output;
    }

    /**
     * Set the side outputs.
     *
     * @param sideOutputs the side outputs set the operator.
     */
    protected void setSideOutputs(Table[] sideOutputs) {
        this.sideOutputs = sideOutputs;
    }
}
