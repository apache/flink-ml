package org.apache.flink.ml.stats.chisqtest;

import org.apache.flink.ml.common.param.HasInputCols;
import org.apache.flink.ml.common.param.HasLabelCol;

/**
 * Params for {@link ChiSqTest}.
 *
 * @param <T> The class type of this instance.
 */
public interface ChiSqTestParams<T> extends HasInputCols<T>, HasLabelCol<T> {}
