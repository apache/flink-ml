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

from pyflink.common import Types
from pyflink.ml.feature.stopwordsremover import StopWordsRemover
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class StopWordsRemoverTest(PyFlinkMLTestCase):
    def setUp(self):
        super(StopWordsRemoverTest, self).setUp()
        self.input_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (["test", "test"], ["test", "test"]),
                (["a", "b", "c", "d"], ["b", "c", "d"]),
                (["a", "the", "an"], []),
                (["A", "The", "AN"], []),
                ([None], [None]),
                ([], []),
            ],
                type_info=Types.ROW_NAMED(
                    ['raw', 'expected'],
                    [Types.OBJECT_ARRAY(Types.STRING()), Types.OBJECT_ARRAY(Types.STRING())])))

    def test_param(self):
        remover = StopWordsRemover()

        self.assertTrue({'i', 'would'} <= set(remover.stop_words))
        self.assertEqual(remover.locale, StopWordsRemover.get_default_or_us())
        self.assertFalse(remover.case_sensitive)

        remover.set_input_cols('f1', 'f2') \
            .set_output_cols('o1', 'o2') \
            .set_stop_words(*StopWordsRemover.load_default_stop_words('turkish')) \
            .set_locale('en_US') \
            .set_case_sensitive(True)

        self.assertEqual(('f1', 'f2'), remover.input_cols)
        self.assertEqual(('o1', 'o2'), remover.output_cols)
        self.assertTrue({'acaba', 'yani'} <= set(remover.stop_words))
        self.assertEqual('en_US', remover.locale)
        self.assertTrue(remover.case_sensitive)

    def test_output_schema(self):
        remover = StopWordsRemover().set_input_cols('raw').set_output_cols('filtered')
        output_table = remover.transform(self.input_table)[0]

        self.assertEqual(
            ['raw', 'expected', 'filtered'],
            output_table.get_schema().get_field_names())

    def test_transform(self):
        remover = StopWordsRemover().set_input_cols('raw').set_output_cols('filtered')
        self.verify_output_result(remover, self.input_table)

    def test_save_load_and_transform(self):
        remover = StopWordsRemover().set_input_cols('raw').set_output_cols('filtered')
        loaded_remover = self.save_and_reload(remover)
        self.verify_output_result(loaded_remover, self.input_table)

    def test_available_locales(self):
        self.assertTrue('en_US' in StopWordsRemover.get_available_locales())

        remover = StopWordsRemover()
        for locale in StopWordsRemover.get_available_locales():
            remover.set_locale(locale)

    def test_default_language_stop_words_not_empty(self):
        supported_languages = {
            "danish",
            "dutch",
            "english",
            "finnish",
            "french",
            "german",
            "hungarian",
            "italian",
            "norwegian",
            "portuguese",
            "russian",
            "spanish",
            "swedish",
            "turkish"
        }

        for language in supported_languages:
            self.assertTrue(len(StopWordsRemover.load_default_stop_words(language)) > 0)

    def verify_output_result(self, remover, input_table):
        output_table = remover.transform(input_table)[0]
        results = [x for x in self.t_env.to_data_stream(output_table).execute_and_collect()]
        expected_length = len([_ for _ in
                               self.t_env.to_data_stream(input_table).execute_and_collect()])
        self.assertEqual(expected_length, len(results))

        field_names = output_table.get_schema().get_field_names()
        for result in results:
            expected = result[field_names.index('expected')]
            actual = result[field_names.index('filtered')]
            self.assertEqual(expected, actual)
