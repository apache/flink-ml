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

package org.apache.flink.iteration.operator;

import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.ChainingStrategy;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

/** Input operator that wraps the user record into {@link IterationRecord}. */
public class InputOperator<T> extends AbstractStreamOperator<IterationRecord<T>>
        implements OneInputStreamOperator<T, IterationRecord<T>> {

    private transient StreamRecord<IterationRecord<T>> reusable;

    public InputOperator() {
        this.chainingStrategy = ChainingStrategy.ALWAYS;
    }

    @Override
    public void open() throws Exception {
        super.open();
        this.reusable = new StreamRecord<>(IterationRecord.newRecord(null, 0));
    }

    @Override
    public void processElement(StreamRecord<T> streamRecord) throws Exception {
        reusable.setTimestamp(streamRecord.getTimestamp());
        reusable.getValue().setValue(streamRecord.getValue());
        output.collect(reusable);
    }

    @Override
    public void processWatermark(Watermark mark) {
        // TODO: FLINK-31373 Support processing watermarks in iterations.
    }
}
