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
import org.apache.flink.iteration.IterationRecord.Type;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.ChainingStrategy;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

/**
 * Output operator that unwraps {@link IterationRecord} into user record, and desert all the event
 * records.
 */
public class OutputOperator<T> extends AbstractStreamOperator<T>
        implements OneInputStreamOperator<IterationRecord<T>, T> {

    private transient StreamRecord<T> reusable;

    public OutputOperator() {
        this.chainingStrategy = ChainingStrategy.ALWAYS;
    }

    @Override
    public void open() throws Exception {
        super.open();
        this.reusable = new StreamRecord<>(null);
    }

    @Override
    public void processElement(StreamRecord<IterationRecord<T>> streamRecord) throws Exception {
        if (streamRecord.getValue().getType() == IterationRecord.Type.RECORD) {
            reusable.replace(streamRecord.getValue().getValue(), streamRecord.getTimestamp());
            output.collect(reusable);
        } else if (streamRecord.getValue().getType() == Type.EPOCH_WATERMARK
                && streamRecord.getValue().getEpoch() == Integer.MAX_VALUE) {
            output.emitWatermark(new Watermark(Long.MAX_VALUE));
        }
    }
}
