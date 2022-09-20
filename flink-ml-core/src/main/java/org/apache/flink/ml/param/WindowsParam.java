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

package org.apache.flink.ml.param;

import org.apache.flink.api.common.time.Time;
import org.apache.flink.ml.common.window.CountTumblingWindows;
import org.apache.flink.ml.common.window.EventTimeSessionWindows;
import org.apache.flink.ml.common.window.EventTimeTumblingWindows;
import org.apache.flink.ml.common.window.GlobalWindows;
import org.apache.flink.ml.common.window.ProcessingTimeSessionWindows;
import org.apache.flink.ml.common.window.ProcessingTimeTumblingWindows;
import org.apache.flink.ml.common.window.Windows;

import java.util.HashMap;
import java.util.Map;

/** Class for the {@link Windows} parameter. */
public class WindowsParam extends Param<Windows> {
    public WindowsParam(
            String name,
            String description,
            Windows defaultValue,
            ParamValidator<Windows> validator) {
        super(name, Windows.class, description, defaultValue, validator);
    }

    @Override
    public Object jsonEncode(Windows value) {
        Map<String, Object> map = new HashMap<>();

        map.put("class", value.getClass().getName());
        if (value instanceof GlobalWindows) {
            return map;
        } else if (value instanceof CountTumblingWindows) {
            map.put("size", ((CountTumblingWindows) value).getSize());
        } else if (value instanceof ProcessingTimeTumblingWindows) {
            map.put("size", ((ProcessingTimeTumblingWindows) value).getSize().toMilliseconds());
        } else if (value instanceof EventTimeTumblingWindows) {
            map.put("size", ((EventTimeTumblingWindows) value).getSize().toMilliseconds());
        } else if (value instanceof ProcessingTimeSessionWindows) {
            map.put("gap", ((ProcessingTimeSessionWindows) value).getGap().toMilliseconds());
        } else if (value instanceof EventTimeSessionWindows) {
            map.put("gap", ((EventTimeSessionWindows) value).getGap().toMilliseconds());
        } else {
            throw new UnsupportedOperationException(
                    String.format("Unsupported %s subclass: %s", Windows.class, value.getClass()));
        }
        return map;
    }

    @Override
    @SuppressWarnings("unchecked")
    public Windows jsonDecode(Object json) {
        Map<String, Object> map = (Map<String, Object>) json;

        String classString = (String) map.get("class");
        if (classString.equals(GlobalWindows.class.getName())) {
            return GlobalWindows.getInstance();
        } else if (classString.equals(CountTumblingWindows.class.getName())) {
            long size = ((Number) map.get("size")).longValue();
            return CountTumblingWindows.of(size);
        } else if (classString.equals(ProcessingTimeTumblingWindows.class.getName())) {
            Time size = Time.milliseconds(((Number) map.get("size")).longValue());
            return ProcessingTimeTumblingWindows.of(size);
        } else if (classString.equals(EventTimeTumblingWindows.class.getName())) {
            Time size = Time.milliseconds(((Number) map.get("size")).longValue());
            return EventTimeTumblingWindows.of(size);
        } else if (classString.equals(ProcessingTimeSessionWindows.class.getName())) {
            Time gap = Time.milliseconds(((Number) map.get("gap")).longValue());
            return ProcessingTimeSessionWindows.withGap(gap);
        } else if (classString.equals(EventTimeSessionWindows.class.getName())) {
            Time gap = Time.milliseconds(((Number) map.get("gap")).longValue());
            return EventTimeSessionWindows.withGap(gap);
        } else {
            throw new UnsupportedOperationException(
                    String.format("Unsupported %s subclass: %s", Windows.class, classString));
        }
    }
}
