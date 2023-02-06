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
import typing
from typing import Tuple

from pyflink.java_gateway import get_gateway
from pyflink.ml.param import Param, StringArrayParam, BooleanParam, StringParam
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureTransformer
from pyflink.ml.common.param import HasInputCols, HasOutputCols


def _load_default_stop_words(language: str) -> Tuple[str, ...]:
    return tuple(*[get_gateway().jvm.org.apache.flink.ml.feature.
                 stopwordsremover.StopWordsRemover.loadDefaultStopWords(language)])


def _get_default_or_us() -> str:
    return get_gateway().jvm.org.apache.flink.ml.feature. \
        stopwordsremover.StopWordsRemover.getDefaultOrUS()


def _get_available_locales() -> set:
    return {*get_gateway().jvm.org.apache.flink.ml.feature.
            stopwordsremover.StopWordsRemover.getAvailableLocales()}


class _StopWordsRemoverParams(
    JavaWithParams,
    HasInputCols,
    HasOutputCols
):
    """
    Params for :class:`StopWordsRemover`.
    """

    STOP_WORDS: Param[Tuple[str, ...]] = StringArrayParam(
        "stop_words",
        "The words to be filtered out.",
        _load_default_stop_words('english'))

    CASE_SENSITIVE: Param[bool] = BooleanParam(
        "case_sensitive",
        "Whether to do a case-sensitive comparison over the stop words.",
        False
    )

    LOCALE: Param[str] = StringParam(
        "locale",
        "Locale of the input for case insensitive matching. Ignored when caseSensitive is true.",
        _get_default_or_us())

    def __init__(self, java_params):
        super(_StopWordsRemoverParams, self).__init__(java_params)

    def set_stop_words(self, *value: str):
        return typing.cast(_StopWordsRemoverParams, self.set(self.STOP_WORDS, value))

    def set_case_sensitive(self, value: bool):
        return typing.cast(_StopWordsRemoverParams, self.set(self.CASE_SENSITIVE, value))

    def set_locale(self, value: str):
        return typing.cast(_StopWordsRemoverParams, self.set(self.LOCALE, value))

    def get_stop_words(self) -> Tuple[str, ...]:
        return self.get(self.STOP_WORDS)

    def get_case_sensitive(self) -> bool:
        return self.get(self.CASE_SENSITIVE)

    def get_locale(self) -> str:
        return self.get(self.LOCALE)

    @property
    def stop_words(self):
        return self.get_stop_words()

    @property
    def case_sensitive(self):
        return self.get_case_sensitive()

    @property
    def locale(self):
        return self.get_locale()


class StopWordsRemover(JavaFeatureTransformer, _StopWordsRemoverParams):
    """
    A feature transformer that filters out stop words from input.

    Note: null values from input array are preserved unless adding null to stopWords explicitly.

    See Also: http://en.wikipedia.org/wiki/Stop_words
    """

    def __init__(self, java_model=None):
        super(StopWordsRemover, self).__init__(java_model)

    @classmethod
    def _java_transformer_package_name(cls) -> str:
        return "stopwordsremover"

    @classmethod
    def _java_transformer_class_name(cls) -> str:
        return "StopWordsRemover"

    @classmethod
    def load_default_stop_words(cls, language: str):
        """
        Loads the default stop words for the given language.

        Supported languages: danish, dutch, english, finnish, french, german, hungarian, italian,
        norwegian, portuguese, russian, spanish, swedish, turkish

        See Also: http://anoncvs.postgresql.org/cvsweb.cgi/pgsql/src/backend/snowball/stopwords/
        """
        return _load_default_stop_words(language)

    @classmethod
    def get_default_or_us(cls):
        """
        Returns system default locale, or "en_US" if the default locale is not available. The locale
        is returned as a String.
        """
        return _get_default_or_us()

    @classmethod
    def get_available_locales(cls):
        """
        Returns a set of all installed locales. It must contain at least a Locale
        instance equal to "en_US". The locales are returned as Strings.
        """
        return _get_available_locales()
