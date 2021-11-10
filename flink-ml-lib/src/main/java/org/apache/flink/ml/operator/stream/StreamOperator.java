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

package org.apache.flink.ml.operator.stream;

import org.apache.flink.ml.operator.BaseAlgoImpl;
import org.apache.flink.ml.param.Param;
import org.apache.flink.table.api.Table;

import java.util.Arrays;
import java.util.Map;

/**
 * Base class of online learning algorithm operators.
 *
 * <p>This class extends {@link BaseAlgoImpl} to support data transmission between StreamOperators.
 */
public abstract class StreamOperator<T extends StreamOperator<T>> extends BaseAlgoImpl<T> {

    /**
     * The constructor of StreamOperator.
     *
     * @param params the initial Params.
     */
    public StreamOperator(Map<Param<?>, Object> params) {
        super(params);
    }

    /**
     * Link to another {@link StreamOperator}.
     *
     * <p>Link the <code>next</code> StreamOperator using this StreamOperator as its input.
     *
     * <p>For example:
     *
     * <pre>{@code
     * StreamOperator a = ...;
     * StreamOperator b = ...;
     * StreamOperator c = a.link(b)
     * }</pre>
     *
     * <p>The StreamOperator <code>c</code> in the above code is the same instance as <code>b
     * </code> which takes <code>a</code> as its input. Note that StreamOperator <code>b
     * </code> will be changed to link from StreamOperator <code>a</code>.
     *
     * @param next The operator that will be modified to add this operator to its input.
     * @param <OP> type of StreamOperator returned
     * @return the linked next
     */
    public final <OP extends StreamOperator<?>> OP link(OP next) {
        next.linkFrom(this);
        return next;
    }

    /**
     * Link from one {@link StreamOperator} and other {@link BaseAlgoImpl}.
     *
     * <p>Link this object to StreamOperator using at least one StreamOperator as its input.
     *
     * <p>For example:
     *
     * <pre>{@code
     * StreamOperator a = ...;
     * StreamOperator b = ...;
     * StreamOperator c = ...;
     *
     * StreamOperator d = c.linkFrom(a, b)
     * }</pre>
     *
     * <p>The <code>d</code> in the above code is the same instance as StreamOperator <code>c</code>
     * which takes both <code>a</code> and <code>b</code> as its input.
     *
     * <p>note: It is not recommended to linkFrom itself or linkFrom the same group inputs twice.
     *
     * @param streamInput the stream input.
     * @param otherInputs the other stream inputs.
     * @return the linked this object
     */
    @SuppressWarnings("unchecked")
    public final T linkFrom(StreamOperator<?> streamInput, BaseAlgoImpl<?>... otherInputs) {
        Table[] inputTables = new Table[1 + otherInputs.length];
        inputTables[0] = streamInput.getOutput();
        int idx = 1;
        for (BaseAlgoImpl<?> otherInput : otherInputs) {
            inputTables[idx++] = otherInput.getOutput();
        }
        Table[] outputTables = transform(inputTables);
        if (outputTables.length >= 1) {
            this.setOutput(outputTables[0]);
        }
        if (outputTables.length > 1) {
            this.setSideOutputs(Arrays.copyOfRange(outputTables, 1, outputTables.length));
        }
        return (T) this;
    }
}
