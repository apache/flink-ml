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
import io
import os
import sys
from platform import python_version
from shutil import copytree, rmtree

from setuptools import setup

if sys.version_info < (3, 6) or sys.version_info >= (3, 9):
    print("Only Python versions between 3.6 and 3.8 (inclusive) are supported for Flink ML. "
          "The current Python version is %s." % python_version(), file=sys.stderr)
    sys.exit(-1)


def remove_if_exists(file_path):
    if os.path.exists(file_path):
        if os.path.islink(file_path) or os.path.isfile(file_path):
            os.remove(file_path)
        else:
            assert os.path.isdir(file_path)
            rmtree(file_path)


this_directory = os.path.abspath(os.path.dirname(__file__))
version_file = os.path.join(this_directory, 'pyflink/ml/version.py')

try:
    exec(open(version_file).read())
except IOError:
    print("Failed to load Flink ML version file for packaging. " +
          "'%s' not found!" % version_file,
          file=sys.stderr)
    sys.exit(-1)
VERSION = __version__  # noqa

with io.open(os.path.join(this_directory, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

TEMP_PATH = "deps"

LIB_TEMP_PATH = os.path.join(TEMP_PATH, "lib")
EXAMPLES_TEMP_PATH = os.path.join(TEMP_PATH, "examples")

in_flink_ml_source = os.path.isfile("../flink-ml-core/src/main/java/org/apache/flink/ml/api/"
                                    "Stage.java")
try:
    if in_flink_ml_source:

        try:
            os.mkdir(TEMP_PATH)
        except:
            print("Temp path for symlink to parent already exists {0}".format(TEMP_PATH),
                  file=sys.stderr)
            sys.exit(-1)
        flink_ml_version = VERSION.replace(".dev0", "-SNAPSHOT")
        FLINK_ML_HOME = os.path.abspath(
            "../flink-ml-dist/target/flink-ml-%s-bin/flink-ml-%s"
            % (flink_ml_version, flink_ml_version))
        FLINK_ML_ROOT = os.path.abspath("..")

        LIB_PATH = os.path.join(FLINK_ML_HOME, "lib")
        EXAMPLES_PATH = os.path.join(this_directory, "pyflink/examples")

        if getattr(os, "symlink", None) is not None:
            os.symlink(LIB_PATH, LIB_TEMP_PATH)
            os.symlink(EXAMPLES_PATH, EXAMPLES_TEMP_PATH)
        else:
            copytree(LIB_PATH, LIB_TEMP_PATH)
            copytree(EXAMPLES_PATH, EXAMPLES_TEMP_PATH)

    PACKAGES = ['pyflink',
                'pyflink.ml',
                'pyflink.ml.classification',
                'pyflink.ml.clustering',
                'pyflink.ml.evaluation',
                'pyflink.ml.feature',
                'pyflink.ml.recommendation',
                'pyflink.ml.regression',
                'pyflink.ml.stats',
                'pyflink.ml.util',
                'pyflink.ml.common',
                'pyflink.lib',
                'pyflink.examples']

    PACKAGE_DIR = {
        'pyflink.lib': TEMP_PATH + '/lib',
        'pyflink.examples': TEMP_PATH + '/examples'}

    PACKAGE_DATA = {
        'pyflink.lib': ['*.jar'],
        'pyflink.examples': ['*.py', '*/*.py']}

    setup(
        name='apache-flink-ml',
        version=VERSION,
        packages=PACKAGES,
        include_package_data=True,
        package_dir=PACKAGE_DIR,
        package_data=PACKAGE_DATA,
        url='https://flink.apache.org',
        license='https://www.apache.org/licenses/LICENSE-2.0',
        author='Apache Software Foundation',
        author_email='dev@flink.apache.org',
        python_requires='>=3.6',
        install_requires=['apache-flink==1.15.1', 'pandas>=1.0,<1.2.0', 'jsonpickle==2.0.0',
                          'cloudpickle==1.2.2', 'numpy>=1.14.3,<1.20'],
        tests_require=['pytest==4.4.1'],
        description='Apache Flink ML Python API',
        long_description=long_description,
        long_description_content_type='text/markdown',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8'],
    )
finally:
    if in_flink_ml_source:
        remove_if_exists(TEMP_PATH)
