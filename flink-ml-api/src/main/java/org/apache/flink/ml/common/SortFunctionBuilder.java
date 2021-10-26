package org.apache.flink.ml.common;

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.operators.Order;
import org.apache.flink.api.java.tuple.Tuple;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** Helper class to build the sort partition function. */
public class SortFunctionBuilder<T extends Tuple> {

    List<Integer> fields = new ArrayList<>();
    List<Order> orders = new ArrayList<>();

    public SortFunctionBuilder<T> sort(int field, Order order) {
        fields.add(field);
        orders.add(order);
        return this;
    }

    public MapPartitionFunction<T, T> build() {
        return new CompositeSortFunction<T>(fields, orders);
    }

    /** A sort function. */
    private static class CompositeSortFunction<T extends Tuple>
            implements MapPartitionFunction<T, T> {
        private final List<Integer> fields;
        private final List<Order> orders;

        public CompositeSortFunction(List<Integer> fields, List<Order> orders) {
            this.fields = fields;
            this.orders = orders;
        }

        @Override
        public void mapPartition(Iterable<T> iterable, Collector<T> out) {
            List<T> values = new ArrayList<>();
            for (T value : iterable) {
                values.add(value);
            }
            values.sort(
                    new Comparator<T>() {
                        @Override
                        public int compare(T value1, T value2) {
                            for (int i = 0; i < fields.size(); i++) {
                                Number fieldValue1 = value1.getField(fields.get(i));
                                Number fieldValue2 = value2.getField(fields.get(i));

                                int comp = compareTo(fieldValue1, fieldValue2);
                                if (comp != 0) {
                                    if (orders.get(i) == Order.ASCENDING) {
                                        return comp;
                                    } else {
                                        return -comp;
                                    }
                                }
                            }
                            return 0;
                        }
                    });

            for (T value : values) {
                out.collect(value);
            }
        }
    }

    private static int compareTo(Number x, Number y) {
        if (Byte.class.equals(x.getClass())) {
            return ((Byte) x).compareTo((Byte) y);
        } else if (Short.class.equals(x.getClass())) {
            return ((Short) x).compareTo((Short) y);
        } else if (Integer.class.equals(x.getClass())) {
            return ((Integer) x).compareTo((Integer) y);
        } else if (Long.class.equals(x.getClass())) {
            return ((Long) x).compareTo((Long) y);
        } else if (Float.class.equals(x.getClass())) {
            return ((Float) x).compareTo((Float) y);
        } else if (Double.class.equals(x.getClass())) {
            return ((Double) x).compareTo((Double) y);
        } else {
            throw new IllegalArgumentException();
        }
    }
}
