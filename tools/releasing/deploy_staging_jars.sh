#!/usr/bin/env bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

##
## Variables with defaults (if not overwritten by environment)
##
MVN=${MVN:-mvn}
CUSTOM_OPTIONS=${CUSTOM_OPTIONS:-}
SUPPORTED_FLINK_VERSIONS=${SUPPORTED_FLINK_VERSIONS:-}

# The variable SUPPORTED_FLINK_VERSIONS must be set, which contains all supported flink
# versions with comma separated, e.g. "1.15,1.16.1.17".
if [ "${SUPPORTED_FLINK_VERSIONS}" = "" ]; then
    echo "Variable SUPPORTED_FLINK_VERSIONS is not set, stop deploying."
    exit 1
fi

# fail immediately
set -o errexit
set -o nounset

CURR_DIR=`pwd`
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
PROJECT_ROOT="${BASE_DIR}/../../"

# Sanity check to ensure that resolved paths are valid; a LICENSE file should aways exist in project root
if [ ! -f ${PROJECT_ROOT}/LICENSE ]; then
    echo "Project root path ${PROJECT_ROOT} is not valid; script may be in the wrong directory."
    exit 1
fi

###########################

cd ${PROJECT_ROOT}

IFS=',' read -r -a arr <<< "${SUPPORTED_FLINK_VERSIONS}"
for v in ${arr[@]}
do
    echo "Deploying Flink ML with flink-${v} to repository.apache.org."
    ${MVN} clean deploy -Papache-release -Pflink-${v} -DskipTests -DretryFailedDeploymentCount=10 $CUSTOM_OPTIONS
done

cd ${CURR_DIR}
