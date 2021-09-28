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

package org.apache.flink.ml.iteration.itcases.operators;

/** The output record type */
public class OutputRecord<T> {

    public enum Event {
        PROCESS_ELEMENT,
        EPOCH_WATERMARK_INCREMENTED,
        TERMINATED
    }

    private Event event;

    private int round;

    private T value;

    public OutputRecord() {}

    public OutputRecord(Event event, int round, T value) {
        this.event = event;
        this.round = round;
        this.value = value;
    }

    public Event getEvent() {
        return event;
    }

    public void setEvent(Event event) {
        this.event = event;
    }

    public int getRound() {
        return round;
    }

    public void setRound(int round) {
        this.round = round;
    }

    public T getValue() {
        return value;
    }

    public void setValue(T value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return "OutputRecord{" + "event=" + event + ", round=" + round + ", value=" + value + '}';
    }
}
