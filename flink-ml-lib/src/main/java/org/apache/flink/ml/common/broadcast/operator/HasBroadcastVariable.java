package org.apache.flink.ml.common.broadcast.operator;

import java.util.List;

/** interface for operator that has broadcast variables. */
public interface HasBroadcastVariable {

    /**
     * set broadcast variable.
     *
     * @param name name of the broadcast variable.
     * @param broadcastVariable list of the broadcast variable.
     */
    void setBroadcastVariable(String name, List<?> broadcastVariable);
}
