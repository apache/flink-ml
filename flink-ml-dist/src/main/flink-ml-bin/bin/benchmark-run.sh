#!/usr/bin/env bash
################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

current_path=$(pwd)
flink_ml_bin_path="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
flink_ml_root_path="$(dirname "$flink_ml_bin_path")"

# Checks FLINK_HOME.
if [ "$FLINK_HOME" == "" ]; then
  echo "FLINK_HOME not found. Please make sure you have installed Flink and configured FLINK_HOME in your environment path."
    exit 1
fi

# Checks flink version.
expected_version="1.14"
actual_version=`$FLINK_HOME/bin/flink --version | cut -d" " -f2 | tr -d ","`
unsorted_versions="${expected_version}\n${actual_version}\n"
sorted_versions=`printf ${unsorted_versions} | sort -V`
unsorted_versions=`printf ${unsorted_versions}`
if [ "${unsorted_versions}" != "${sorted_versions}" ]; then
    echo "$flink_cmd $expected_version or a higher version is required, but found $actual_version"
    exit 1
fi

# Submits benchmark flink job.
$FLINK_HOME/bin/flink run -c org.apache.flink.ml.benchmark.Benchmark $FLINK_HOME/lib/flink-ml-uber*.jar ${@:1}
