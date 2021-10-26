package org.apache.flink.ml.common;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.tuple.Tuple;

import java.util.ArrayList;
import java.util.List;

/** Helper class to build the reduce function. */
public class ReduceFunctionBuilder<T extends Tuple> {

    enum Aggregation {
        SUM,
        MIN,
        MAX;
    }

    List<Aggregation> aggregations = new ArrayList<>();
    List<Integer> fields = new ArrayList<>();

    public ReduceFunctionBuilder<T> sum(int field) {
        aggregations.add(Aggregation.SUM);
        fields.add(field);
        return this;
    }

    public ReduceFunctionBuilder<T> min(int field) {
        aggregations.add(Aggregation.MIN);
        fields.add(field);
        return this;
    }

    public ReduceFunctionBuilder<T> max(int field) {
        aggregations.add(Aggregation.MAX);
        fields.add(field);
        return this;
    }

    public ReduceFunction<T> build() {
        return new CompositeReduceFunction<T>(aggregations, fields);
    }

    /** A reduce function. */
    private static class CompositeReduceFunction<T extends Tuple> implements ReduceFunction<T> {
        private final List<Aggregation> aggregations;
        private final List<Integer> fields;

        public CompositeReduceFunction(List<Aggregation> aggregations, List<Integer> fields) {
            this.aggregations = aggregations;
            this.fields = fields;
        }

        @Override
        public T reduce(T value1, T value2) {
            for (int i = 0; i < fields.size(); i++) {
                int field = fields.get(i);
                Number fieldValue1 = value1.getField(field);
                Number fieldValue2 = value2.getField(field);
                Number newValue = 0;

                switch (aggregations.get(i)) {
                    case SUM:
                        newValue = add(fieldValue1, fieldValue2);
                        break;
                    case MIN:
                        newValue = min(fieldValue1, fieldValue2);
                        break;
                    case MAX:
                        newValue = max(fieldValue1, fieldValue2);
                        break;
                }
                value2.setField(newValue, field);
            }
            return value2;
        }
    }

    // TODO: benchmark an actual algorithm and evaluate whether we should optimize the overhead of
    // class comparison here.
    private static Number add(Number x, Number y) {
        if (Byte.class.equals(x.getClass())) {
            return x.byteValue() + y.byteValue();
        } else if (Short.class.equals(x.getClass())) {
            return x.shortValue() + y.shortValue();
        } else if (Integer.class.equals(x.getClass())) {
            return x.intValue() + y.intValue();
        } else if (Long.class.equals(x.getClass())) {
            return x.longValue() + y.longValue();
        } else if (Float.class.equals(x.getClass())) {
            return x.floatValue() + y.floatValue();
        } else if (Double.class.equals(x.getClass())) {
            return x.doubleValue() + y.doubleValue();
        } else {
            throw new IllegalArgumentException();
        }
    }

    private static Number min(Number x, Number y) {
        if (Byte.class.equals(x.getClass())) {
            return Math.min(x.byteValue(), y.byteValue());
        } else if (Short.class.equals(x.getClass())) {
            return Math.min(x.shortValue(), y.shortValue());
        } else if (Integer.class.equals(x.getClass())) {
            return Math.min(x.intValue(), y.intValue());
        } else if (Long.class.equals(x.getClass())) {
            return Math.min(x.longValue(), y.longValue());
        } else if (Float.class.equals(x.getClass())) {
            return Math.min(x.floatValue(), y.floatValue());
        } else if (Double.class.equals(x.getClass())) {
            return Math.min(x.doubleValue(), y.doubleValue());
        } else {
            throw new IllegalArgumentException();
        }
    }

    private static Number max(Number x, Number y) {
        if (Byte.class.equals(x.getClass())) {
            return Math.max(x.byteValue(), y.byteValue());
        } else if (Short.class.equals(x.getClass())) {
            return Math.max(x.shortValue(), y.shortValue());
        } else if (Integer.class.equals(x.getClass())) {
            return Math.max(x.intValue(), y.intValue());
        } else if (Long.class.equals(x.getClass())) {
            return Math.max(x.longValue(), y.longValue());
        } else if (Float.class.equals(x.getClass())) {
            return Math.max(x.floatValue(), y.floatValue());
        } else if (Double.class.equals(x.getClass())) {
            return Math.max(x.doubleValue(), y.doubleValue());
        } else {
            throw new IllegalArgumentException();
        }
    }
}
