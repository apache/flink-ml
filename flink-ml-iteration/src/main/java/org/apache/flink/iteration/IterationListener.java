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

package org.apache.flink.iteration;

import org.apache.flink.annotation.Experimental;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

/**
 * The callbacks which will be invoked if the operator or UDF inside iteration implement this
 * interface.
 */
@Experimental
public interface IterationListener<T> {

    /**
     * This callback is invoked every time the epoch watermark of this operator increments. The
     * initial epoch watermark of an operator is 0.
     *
     * <p>The epochWatermark is the maximum integer that meets this requirement: every record that
     * arrives at the operator going forward should have an epoch larger than the epochWatermark.
     * See Java docs in IterationUtils for how epoch is determined for records ingested into the
     * iteration body and for records emitted by operators within the iteration body.
     *
     * <p>If all inputs are bounded, the maximum epoch of all records ingested into this operator is
     * used as the epochWatermark parameter for the last invocation of this callback.
     *
     * @param epochWatermark The incremented epoch watermark.
     * @param context A context that allows emitting side output. The context is only valid during
     *     the invocation of this method.
     * @param collector The collector for returning result values.
     */
    void onEpochWatermarkIncremented(int epochWatermark, Context context, Collector<T> collector);

    /**
     * This callback is invoked after the execution of the iteration body has terminated.
     *
     * @see Iterations
     * @param context A context that allows emitting side output. The context is only valid during
     *     the invocation of this method.
     * @param collector The collector for returning result values.
     */
    void onIterationTerminated(Context context, Collector<T> collector);

    /**
     * Information available in an invocation of the callbacks defined in the
     * IterationProgressListener.
     */
    interface Context {
        /**
         * Emits a record to the side output identified by the {@link OutputTag}.
         *
         * @param outputTag the {@code OutputTag} that identifies the side output to emit to.
         * @param value The record to emit.
         */
        <X> void output(OutputTag<X> outputTag, X value);
    }
}
