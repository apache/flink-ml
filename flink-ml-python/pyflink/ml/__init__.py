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
from py4j.java_gateway import JavaClass, get_java_class, JavaObject
from pyflink.java_gateway import get_gateway
from pyflink.util import java_utils
from pyflink.util.java_utils import to_jarray, load_java_class


# TODO: Remove custom jar loader after FLINK-15635 and FLINK-28002 are fixed and released.
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
    if is_instance_of(context_classloader, URLClassLoaderClass):
        if class_loader_name == "org.apache.flink.runtime.execution.librarycache." \
                                "FlinkUserCodeClassLoaders$SafetyNetWrapperClassLoader":
            ensureInner = context_classloader.getClass().getDeclaredMethod("ensureInner", None)
            ensureInner.setAccessible(True)
            context_classloader = ensureInner.invoke(context_classloader, None)

        addURL = URLClassLoaderClass.getDeclaredMethod(
            "addURL",
            to_jarray(
                gateway.jvm.Class,
                [load_java_class("java.net.URL")]))
        addURL.setAccessible(True)

        for url in jar_urls:
            addURL.invoke(context_classloader, to_jarray(get_gateway().jvm.Object, [url]))

    else:
        context_classloader = create_url_class_loader(jar_urls, context_classloader)
        gateway.jvm.Thread.currentThread().setContextClassLoader(context_classloader)


def is_instance_of(java_object, java_class):
    gateway = get_gateway()
    if isinstance(java_class, str):
        param = java_class
    elif isinstance(java_class, JavaClass):
        param = get_java_class(java_class)
    elif isinstance(java_class, JavaObject):
        if not is_instance_of(java_class, gateway.jvm.Class):
            param = java_class.getClass()
        else:
            param = java_class
    else:
        raise TypeError(
            "java_class must be a string, a JavaClass, or a JavaObject")

    return gateway.jvm.org.apache.flink.api.python.shaded.py4j.reflection.TypeUtil.isInstanceOf(
        param, java_object)


def create_url_class_loader(urls, parent_class_loader):
    gateway = get_gateway()
    url_class_loader = gateway.jvm.java.net.URLClassLoader(
        to_jarray(gateway.jvm.java.net.URL, urls), parent_class_loader)
    return url_class_loader


java_utils.add_jars_to_context_class_loader = add_jars_to_context_class_loader
