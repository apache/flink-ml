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
from pyflink.java_gateway import get_gateway
from pyflink.util import java_utils
from pyflink.util.java_utils import to_jarray, load_java_class


def add_jars_to_context_class_loader(jar_urls):
    """
    Add jars to Python gateway server for local compilation and local execution (i.e. minicluster).
    There are many component in Flink which won't be added to classpath by default. e.g. Kafka
    connector, JDBC connector, CSV format etc. This utility function can be used to hot load the
    jars.

    :param jar_urls: The list of jar urls.
    """
    gateway = get_gateway()
    # validate and normalize
    jar_urls = [gateway.jvm.java.net.URL(url) for url in jar_urls]
    context_classloader = gateway.jvm.Thread.currentThread().getContextClassLoader()
    existing_urls = []
    class_loader_name = context_classloader.getClass().getName()
    if class_loader_name == "java.net.URLClassLoader":
        existing_urls = set([url.toString() for url in context_classloader.getURLs()])
    if all([url.toString() in existing_urls for url in jar_urls]):
        # if urls all existed, no need to create new class loader.
        return
    URLClassLoaderClass = load_java_class("java.net.URLClassLoader")
    addURL = URLClassLoaderClass.getDeclaredMethod(
        "addURL",
        to_jarray(
            gateway.jvm.Class,
            [load_java_class("java.net.URL")]))
    addURL.setAccessible(True)
    if class_loader_name == "org.apache.flink.runtime.execution.librarycache." \
                            "FlinkUserCodeClassLoaders$SafetyNetWrapperClassLoader":
        ensureInner = context_classloader.getClass().getDeclaredMethod("ensureInner", None)
        ensureInner.setAccessible(True)
        loader = ensureInner.invoke(context_classloader, None)
    else:
        loader = context_classloader
    for url in jar_urls:
        addURL.invoke(loader, to_jarray(get_gateway().jvm.Object, [url]))


java_utils.add_jars_to_context_class_loader = add_jars_to_context_class_loader
