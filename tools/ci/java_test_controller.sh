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

#
# This file contains generic control over the execution of Flink ML's java tests.
#

HERE="`dirname \"$0\"`"             # relative
HERE="`( cd \"$HERE\" && pwd )`"    # absolutized and normalized
if [ -z "$HERE" ] ; then
	exit 1
fi

source "${HERE}/controller_utils.sh"
source "${HERE}/stage.sh"

STAGE=$1

if [ -z "${STAGE:-}" ] ; then
	echo "ERROR: Environment variable 'STAGE' is not set but expected by java_test_controller.sh. The variable refers to the stage being executed."
	exit 1
fi

# =============================================================================
# Step 1: Rebuild jars and install Flink ML to local maven repository
# =============================================================================

MVN_COMMON_OPTIONS="--no-transfer-progress"
MVN_COMPILE_OPTIONS="-DskipTests"
MVN_COMPILE_MODULES=$(get_compile_modules_for_stage ${STAGE})
exit_if_error $? "Error: Unexpected STAGE value ${STAGE}"

mvn clean install $MVN_COMMON_OPTIONS $MVN_COMPILE_OPTIONS $MVN_COMPILE_MODULES
exit_if_error $? "Compilation failure detected, skipping test execution."

# =============================================================================
# Step 2: Run tests
# =============================================================================

MVN_TEST_MODULES=$(get_test_modules_for_stage ${STAGE})
exit_if_error $? "Error: Unexpected STAGE value ${STAGE}"

mvn verify $MVN_COMMON_OPTIONS $MVN_TEST_MODULES
