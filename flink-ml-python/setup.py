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
from shutil import copytree, copy, rmtree

from setuptools import setup

if sys.version_info < (3, 6):
    print("Python versions prior to 3.6 are not supported for Flink ML.",
          file=sys.stderr)
    sys.exit(-1)


def remove_if_exists(file_path):
    if os.path.exists(file_path):
        if os.path.islink(file_path) or os.path.isfile(file_path):
            os.remove(file_path)
        else:
            assert os.path.isdir(file_path)
            rmtree(file_path)


this_directory = os.path.abspath(os.path.dirname(__file__))
version_file = os.path.join(this_directory, 'apache_flink_ml/version.py')

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

EXAMPLES_TEMP_PATH = os.path.join(TEMP_PATH, "examples")
SCRIPTS_TEMP_PATH = os.path.join(TEMP_PATH, "bin")

LICENSE_FILE_TEMP_PATH = os.path.join('apache_flink_ml', "LICENSE")
README_FILE_TEMP_PATH = os.path.join("apache_flink_ml", "README.txt")

in_flink_ml_source = os.path.isfile("../flink-ml-api/src/main/java/org/apache/flink/ml/api/core/"
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
        FLINK_ML_ROOT = os.path.abspath("..")

        EXAMPLES_PATH = os.path.join(this_directory, "apache_flink_ml/examples")

        LICENSE_FILE_PATH = os.path.join(FLINK_ML_ROOT, "LICENSE")
        README_FILE_PATH = os.path.join(this_directory, "README.md")

        try:
            os.symlink(EXAMPLES_PATH, EXAMPLES_TEMP_PATH)
            os.symlink(LICENSE_FILE_PATH, LICENSE_FILE_TEMP_PATH)
            os.symlink(README_FILE_PATH, README_FILE_TEMP_PATH)
        except BaseException:  # pylint: disable=broad-except
            copytree(EXAMPLES_PATH, EXAMPLES_TEMP_PATH)
            copy(LICENSE_FILE_PATH, LICENSE_FILE_TEMP_PATH)
            copy(README_FILE_PATH, README_FILE_TEMP_PATH)

    PACKAGES = ['apache_flink_ml',
                'apache_flink_ml.ml',
                'apache_flink_ml.ml.api',
                'apache_flink_ml.ml.lib',
                'apache_flink_ml.ml.param',
                'apache_flink_ml.ml.util',
                'apache_flink_ml.examples']

    PACKAGE_DIR = {
        'apache_flink_ml.examples': TEMP_PATH + '/examples'}

    PACKAGE_DATA = {
        'apache_flink_ml': ['README.txt', 'LICENSE'],
        'apache_flink_ml.examples': ['*.py', '*/*.py']}

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
        install_requires=['apache-flink==1.14.0', 'pandas>=1.0,<1.2.0', 'jsonpickle==2.0.0',
                          'cloudpickle==1.2.2'],
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
        remove_if_exists(LICENSE_FILE_TEMP_PATH)
        remove_if_exists(README_FILE_TEMP_PATH)
