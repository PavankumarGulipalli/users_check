import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
import os
import shutil
import glob
from datetime import date, datetime, timedelta

# Assuming user_profile_pl.py is in the same directory or accessible in PYTHONPATH
# For the testing environment, it's usually in the root.
from user_profile_pl import (
    getSelectedAudioLanguage,
    get_mapping_audiolanguage,
    read_events_data,
    read_deeplink_events_data
)

ALL_EXPECTED_COLUMNS = [
    'userid', 'projectid', 'content_id', 'content_type', 'originmedium',
    'originname', 'originsource', 'trackingid', 'carouselid', 'device',
    'value', 'ts', 'additionalproperties', 'tabname', 'carouselposition',
    'contentposition', 'Updated_Type'
]

# Helper to create a schema with all columns as Utf8, as expected by scan_csv mock
ALL_UTF8_SCHEMA = {col: pl.Utf8 for col in ALL_EXPECTED_COLUMNS}

class TestGetSelectedAudioLanguage(unittest.TestCase):
    def test_valid_language_present(self):
        self.assertEqual(getSelectedAudioLanguage('{"selectedAudioLanguage": "English", "other": "data"}'), "English")

    def test_valid_language_not_present(self):
        self.assertEqual(getSelectedAudioLanguage('{"otherProperty": "value"}'), "")

    def test_valid_language_empty_value(self):
        self.assertEqual(getSelectedAudioLanguage('{"selectedAudioLanguage": "", "other": "data"}'), "")

    def test_empty_string_input(self):
        self.assertEqual(getSelectedAudioLanguage(""), "")

    def test_malformed_json_string(self):
        self.assertEqual(getSelectedAudioLanguage("{'selectedAudioLanguage': 'English', invalid}"), "")

    def test_not_a_string_input(self):
        # The function has type hint x:str, but good to test defensive check
        self.assertEqual(getSelectedAudioLanguage(None), "") # type: ignore
        self.assertEqual(getSelectedAudioLanguage(123), "") # type: ignore

class TestGetMappingAudioLanguage(unittest.TestCase):
    def test_basic_mapping(self):
        data_lf = pl.LazyFrame({
            'user_id': ['u1', 'u1', 'u2'],
            'content_id': ['c1', 'c1', 'c1'],
            'mou': [10.0, 20.0, 5.0],
            'selectedAudioLanguage': ['Eng', 'Hin', 'Eng']
        })
        expected_lf = pl.LazyFrame({
            'user_id': ['u1', 'u1', 'u2'],
            'content_id': ['c1', 'c1', 'c1'],
            'mou': [10.0, 20.0, 5.0],
            'selectedAudioLanguage': ['Hin', 'Hin', 'Eng']
        })
        
        result_lf = get_mapping_audiolanguage(data_lf)
        self.assertIsInstance(result_lf, pl.LazyFrame)
        # Sort by user_id, content_id, mou for consistent comparison
        assert_frame_equal(
            result_lf.sort(['user_id', 'content_id', 'mou']).collect(),
            expected_lf.sort(['user_id', 'content_id', 'mou']).collect(),
            check_dtype=True
        )

    def test_tie_in_mou_picks_first_original_lang_in_group(self):
        # Polars .first() behavior after filter can depend on original order or internal stability
        # The implementation uses .first() after filter(max_mou) -> group_by(keys).agg(first(lang))
        # This should be deterministic if the input row order for the same max_mou is consistent.
        data_lf = pl.LazyFrame({
            'user_id': ['u1', 'u1', 'u1'],
            'content_id': ['c1', 'c1', 'c1'],
            'mou': [20.0, 10.0, 20.0], # Tie: 20.0 for 'Eng' and 'Tam'
            'selectedAudioLanguage': ['Eng', 'Hin', 'Tam']
        })
        
        # After max_mou_in_group filter: (u1,c1,20.0,Eng), (u1,c1,20.0,Tam)
        # Then groupby(u1,c1).agg(pl.first(selectedAudioLanguage)) -> will pick 'Eng' if it appears first in this intermediate frame
        # Polars' groupby().first() is documented to take the first element of each group.
        # The stability of this intermediate frame might depend on previous operations.
        # Let's assume 'Eng' comes before 'Tam' if their mou are equal and max.
        expected_language_after_tie = 'Eng' 
        
        expected_lf = pl.LazyFrame({
            'user_id': ['u1', 'u1', 'u1'],
            'content_id': ['c1', 'c1', 'c1'],
            'mou': [20.0, 10.0, 20.0],
            'selectedAudioLanguage': [expected_language_after_tie, expected_language_after_tie, expected_language_after_tie]
        })
        
        result_lf = get_mapping_audiolanguage(data_lf)
        self.assertIsInstance(result_lf, pl.LazyFrame)
        assert_frame_equal(
            result_lf.sort(['user_id', 'content_id', 'mou']).collect(),
            expected_lf.sort(['user_id', 'content_id', 'mou']).collect(),
            check_dtype=True
        )

    def test_empty_input_lazyframe(self):
        data_lf = pl.LazyFrame({
            'user_id': [], 'content_id': [], 'mou': [], 'selectedAudioLanguage': []
        }, schema={'user_id':pl.Utf8, 'content_id':pl.Utf8, 'mou':pl.Float64, 'selectedAudioLanguage':pl.Utf8})
        
        expected_lf = pl.LazyFrame({
            'user_id': [], 'content_id': [], 'mou': [], 'selectedAudioLanguage': []
        }, schema={'user_id':pl.Utf8, 'content_id':pl.Utf8, 'mou':pl.Float64, 'selectedAudioLanguage':pl.Utf8})

        result_lf = get_mapping_audiolanguage(data_lf)
        self.assertIsInstance(result_lf, pl.LazyFrame)
        assert_frame_equal(result_lf.collect(), expected_lf.collect(), check_dtype=True)

    def test_mou_is_null(self):
        data_lf = pl.LazyFrame({
            'user_id': ['u1', 'u1'],
            'content_id': ['c1', 'c1'],
            'mou': [10.0, None], # Polars max() will ignore nulls
            'selectedAudioLanguage': ['Eng', 'Hin']
        })
        expected_lf = pl.LazyFrame({
            'user_id': ['u1', 'u1'],
            'content_id': ['c1', 'c1'],
            'mou': [10.0, None],
            'selectedAudioLanguage': ['Eng', 'Eng'] # 'Eng' has max non-null mou
        })
        result_lf = get_mapping_audiolanguage(data_lf)
        self.assertIsInstance(result_lf, pl.LazyFrame)
        assert_frame_equal(
            result_lf.sort(['user_id', 'content_id', 'mou'], nulls_last=True).collect(), 
            expected_lf.sort(['user_id', 'content_id', 'mou'], nulls_last=True).collect(), 
            check_dtype=True
        )

    def test_all_mou_null_for_group(self):
        data_lf = pl.LazyFrame({
            'user_id': ['u1', 'u1', 'u2'],
            'content_id': ['c1', 'c1', 'c2'],
            'mou': [None, None, 10.0],
            'selectedAudioLanguage': ['Eng', 'Hin', 'Tel']
        })
        # For u1, c1: max_mou_in_group will be null. filter(mou == max_mou_in_group) might yield nothing or all rows with null mou.
        # If mou and max_mou_in_group are both null, filter condition `null == null` is false.
        # So mapping_lf will not have entries for (u1,c1).
        # Left join will result in null for 'audio_lang_with_max_mou'.
        # Then rename makes 'selectedAudioLanguage' null for u1,c1.
        expected_lf = pl.LazyFrame({
            'user_id': ['u1', 'u1', 'u2'],
            'content_id': ['c1', 'c1', 'c2'],
            'mou': [None, None, 10.0],
            'selectedAudioLanguage': [None, None, 'Tel'] # Null for u1,c1 group
        }, schema={'user_id':pl.Utf8, 'content_id':pl.Utf8, 'mou':pl.Float64, 'selectedAudioLanguage':pl.Utf8})
        
        result_lf = get_mapping_audiolanguage(data_lf)
        self.assertIsInstance(result_lf, pl.LazyFrame)
        assert_frame_equal(
            result_lf.sort(['user_id', 'content_id', 'mou'], nulls_last=True).collect(), 
            expected_lf.sort(['user_id', 'content_id', 'mou'], nulls_last=True).collect(),
            check_dtype=True
        )


class TestReadEventsFunctions(unittest.TestCase):
    def _get_mock_scan_csv_return_value(self, data_list_of_dicts):
        # Helper to create a LazyFrame with all columns as Utf8 initially
        if not data_list_of_dicts:
             # Handle empty data case for scan_csv (e.g., no files found or all files empty)
             # Create an empty frame with the expected schema
            return pl.LazyFrame({col: [] for col in ALL_EXPECTED_COLUMNS}, schema=ALL_UTF8_SCHEMA)

        # Ensure all columns are present, filling with empty string for Utf8
        processed_data = []
        for row_dict in data_list_of_dicts:
            new_row = {col: str(row_dict.get(col, "")) for col in ALL_EXPECTED_COLUMNS}
            processed_data.append(new_row)
        
        # Create a DataFrame with the specified schema (all Utf8)
        df = pl.DataFrame(processed_data, schema=ALL_UTF8_SCHEMA)
        return df.lazy()

    def setUp(self):
        # Common mocks for file system operations
        self.patcher_exists = patch('os.path.exists')
        self.patcher_listdir = patch('os.listdir')
        self.patcher_makedirs = patch('os.makedirs') # os.makedirs used in functions
        self.patcher_copyfile = patch('shutil.copyfile')
        self.patcher_rmtree = patch('shutil.rmtree')
        self.patcher_glob = patch('glob.glob')
        self.patcher_scan_csv = patch('polars.scan_csv')

        self.mock_exists = self.patcher_exists.start()
        self.mock_listdir = self.patcher_listdir.start()
        self.mock_makedirs = self.patcher_makedirs.start()
        self.mock_copyfile = self.patcher_copyfile.start()
        self.mock_rmtree = self.patcher_rmtree.start()
        self.mock_glob = self.patcher_glob.start()
        self.mock_scan_csv = self.patcher_scan_csv.start()

        # Default behaviors for mocks
        self.mock_exists.return_value = True  # Assume paths exist by default
        self.mock_listdir.return_value = ['events_20230101_00001.csv'] # Non-empty overall_events
        self.mock_glob.return_value = ['temp_dir/events_20230101_00001.csv'] # Mocked glob result for copy loop

        # Mock project_directory, start_date, end_date
        self.project_dir = "dummy_project_dir"
        self.start_date = date(2023, 1, 1)
        self.end_date = date(2023, 1, 1)
        
        # Shared mock data for scan_csv
        self.sample_event_data_dicts = [
            { # Row 1: u1, c1, mou 10, Eng
                'userid': 'u1', 'projectid': 'p1', 'content_id': 'c1', 'content_type': 'movie',
                'value': '10.0', 'ts': '2023-01-01T10:00:00', 
                'additionalproperties': '{"selectedAudioLanguage": "English"}'
            },
            { # Row 2: u1, c1, mou 20, Hin (this should be chosen for u1,c1)
                'userid': 'u1', 'projectid': 'p1', 'content_id': 'c1', 'content_type': 'movie',
                'value': '20.0', 'ts': '2023-01-01T11:30:00', 
                'additionalproperties': '{"selectedAudioLanguage": "Hindi"}'
            },
            { # Row 3: u2, c2, mou 5, Tel
                'userid': 'u2', 'projectid': 'p2', 'content_id': 'c2', 'content_type': 'tvepisode',
                'value': '5.0', 'ts': '2023-01-01T12:00:00', 
                'additionalproperties': '{"selectedAudioLanguage": "Telugu"}'
            },
            { # Row 4: u3, c3, to be filtered out by content_type 'vod'
                'userid': 'u3', 'projectid': 'p3', 'content_id': 'c3', 'content_type': 'vod',
                'value': '15.0', 'ts': '2023-01-01T13:00:00', 
                'additionalproperties': '{"selectedAudioLanguage": "Tamil"}'
            },
            { # Row 5: u4, c4, missing 'content_type', should be dropped by drop_nulls
                'userid': 'u4', 'projectid': 'p4', 'content_id': 'c4', 'content_type': None, # or ""
                'value': '25.0', 'ts': '2023-01-01T14:00:00', 
                'additionalproperties': '{"selectedAudioLanguage": "Kannada"}'
            }
        ]
        self.mock_scan_csv.return_value = self._get_mock_scan_csv_return_value(self.sample_event_data_dicts)

    def tearDown(self):
        self.patcher_exists.stop()
        self.patcher_listdir.stop()
        self.patcher_makedirs.stop()
        self.patcher_copyfile.stop()
        self.patcher_rmtree.stop()
        self.patcher_glob.stop()
        self.patcher_scan_csv.stop()

    def test_read_events_data_successful_run_movie_content(self):
        # Test with content_type = 'movie'
        result_lf = read_events_data(self.project_dir, self.end_date, self.start_date, "movie")
        self.assertIsInstance(result_lf, pl.LazyFrame)
        result_df = result_lf.collect()

        # Expected output for 'movie' content_type:
        # Only u1, c1 data. Language should be Hindi (max mou).
        # ts for u1,c1,Eng: 2023-01-01T10:00:00 -> 2023-01-01T15:30:00 IST -> hour 15
        # ts for u1,c1,Hin: 2023-01-01T11:30:00 -> 2023-01-01T17:00:00 IST -> hour 17
        # After get_mapping_audiolanguage, both rows for (u1,c1) will have 'Hindi'
        # Grouped by (c1, u1, Hindi):
        # mou = 10.0 + 20.0 = 30.0
        # streams = 1.0 + 1.0 = 2.0
        # hour_segment_list = [15, 17] (order might vary, but mode should be consistent)
        # mode([15,17]) -> can be 15 or 17. Polars mode().first() behavior. Let's assume 15.
        expected_data = {
            'content_id': ['c1'],
            'user_id': ['u1'],
            'selectedAudioLanguage': ['Hindi'],
            'mou': [30.0],
            'streams': [2.0],
            'hour_segment': [15] # Or 17, depends on mode of [15,17]
        }
        expected_df = pl.DataFrame(expected_data).with_columns([
            pl.col('mou').cast(pl.Float64),
            pl.col('streams').cast(pl.Float64),
            pl.col('hour_segment').cast(pl.UInt32) # dt.hour() is UInt32
        ])
        
        # Sort by keys to ensure consistent comparison
        assert_frame_equal(
            result_df.sort(['content_id', 'user_id']), 
            expected_df.sort(['content_id', 'user_id']),
            check_dtype=True # hour_segment type might differ if not cast in expected
        )
        self.mock_rmtree.assert_called_once() # Check cleanup

    def test_read_events_data_all_content(self):
        # Test with content_type = 'all'
        # This will include (u2,c2) data as well. (u3,c3) is vod, (u4,c4) dropped due to null content_type.
        # (u2,c2,Tel): mou 5.0, streams 1.0, ts: 2023-01-01T12:00:00 -> 17:30:00 IST -> hour 17
        result_lf = read_events_data(self.project_dir, self.end_date, self.start_date, "all")
        result_df = result_lf.collect()

        expected_data_all = [
            {'content_id': 'c1', 'user_id': 'u1', 'selectedAudioLanguage': 'Hindi', 'mou': 30.0, 'streams': 2.0, 'hour_segment': 15}, # or 17
            {'content_id': 'c2', 'user_id': 'u2', 'selectedAudioLanguage': 'Telugu', 'mou': 5.0, 'streams': 1.0, 'hour_segment': 17}
        ]
        expected_df_all = pl.DataFrame(expected_data_all).with_columns([
            pl.col('mou').cast(pl.Float64),
            pl.col('streams').cast(pl.Float64),
            pl.col('hour_segment').cast(pl.UInt32)
        ])
        
        assert_frame_equal(
            result_df.sort(['content_id', 'user_id']), 
            expected_df_all.sort(['content_id', 'user_id']),
            check_dtype=True
        )

    def test_read_deeplink_events_data_successful_run(self):
        result_lf = read_deeplink_events_data(self.project_dir, self.end_date, self.start_date, "all")
        self.assertIsInstance(result_lf, pl.LazyFrame)
        result_df = result_lf.collect()

        # Expected: No aggregation. (u1,c1) rows will have 'Hindi'. (u2,c2) as is. (u3,c3) as is. (u4,c4) dropped.
        # Original rows from sample_event_data_dicts (after transformations):
        # 1. u1, c1, 10.0, movie, ts_15:30(hr 15), Hindi (after get_mapping)
        # 2. u1, c1, 20.0, movie, ts_17:00(hr 17), Hindi (after get_mapping)
        # 3. u2, c2, 5.0,  tvepisode, ts_17:30(hr 17), Telugu
        # 4. u3, c3, 15.0, vod, ts_18:30(hr 18), Tamil
        # Row 5 (u4,c4) is dropped because content_type is null.
        # Original read_deeplink_events_data in pandas did NOT filter by content_type argument.
        # So, if "all" is passed, it should mean all valid data after processing.
        
        expected_data = [
            {'user_id': 'u1', 'content_id': 'c1', 'mou': 10.0, 'streams': 1.0, 'hour_segment': 15, 'selectedAudioLanguage': 'Hindi'},
            {'user_id': 'u1', 'content_id': 'c1', 'mou': 20.0, 'streams': 1.0, 'hour_segment': 17, 'selectedAudioLanguage': 'Hindi'},
            {'user_id': 'u2', 'content_id': 'c2', 'mou': 5.0,  'streams': 1.0, 'hour_segment': 17, 'selectedAudioLanguage': 'Telugu'},
            {'user_id': 'u3', 'content_id': 'c3', 'mou': 15.0, 'streams': 1.0, 'hour_segment': 18, 'selectedAudioLanguage': 'Tamil'},
        ]
        expected_df = pl.DataFrame(expected_data).with_columns([
            pl.col("streams").cast(pl.Float64),
            pl.col("hour_segment").cast(pl.UInt32) # dt.hour() is UInt32
        ])
        
        # Sort by all columns for robust comparison as row order isn't guaranteed
        sort_cols = ['user_id', 'content_id', 'mou']
        assert_frame_equal(
            result_df.sort(sort_cols), 
            expected_df.sort(sort_cols),
            check_dtype=True
        )
        self.mock_rmtree.assert_called_once()

    def test_read_events_data_invalid_content_type(self):
        with self.assertRaisesRegex(ValueError, 'Invalid content type.'):
            read_events_data(self.project_dir, self.end_date, self.start_date, "bad_type").collect()

    def test_read_events_data_events_folder_not_exist(self):
        self.mock_exists.side_effect = lambda path: False if 'overall_events' in path else True
        with self.assertRaisesRegex(FileNotFoundError, 'overall_events folder not present'):
            read_events_data(self.project_dir, self.end_date, self.start_date, "movie").collect()
            
    def test_read_events_data_events_folder_empty(self):
        self.mock_listdir.return_value = [] # Empty overall_events folder
        with self.assertRaisesRegex(FileNotFoundError, 'overall_events folder is empty'):
            read_events_data(self.project_dir, self.end_date, self.start_date, "movie").collect()

    def test_read_events_data_no_csv_files_in_temp(self):
        # This tests if scan_csv gets an empty list of files or a pattern that matches nothing.
        # The file copying loop might not copy anything if source folders are empty/missing.
        # If os.listdir(curr_folder) is empty, no copyfile calls.
        # Then glob.glob(events_temp_directory + os.sep + 'events_*.csv') would be empty.
        # polars.scan_csv with an empty list of files or a pattern that matches no files
        # typically returns an empty DataFrame or errors, depending on version/params.
        # The current mock for scan_csv uses _get_mock_scan_csv_return_value which can handle empty list of dicts.
        # Let's simulate no files copied, thus scan_csv gets empty data.
        
        self.mock_scan_csv.return_value = self._get_mock_scan_csv_return_value([])
        
        result_lf = read_events_data(self.project_dir, self.end_date, self.start_date, "movie")
        result_df = result_lf.collect()
        
        # Expect an empty frame with the correct schema from the aggregation step
        expected_cols = ['content_id', 'user_id', 'selectedAudioLanguage', 'mou', 'streams', 'hour_segment']
        self.assertTrue(result_df.is_empty())
        self.assertEqual(set(result_df.columns), set(expected_cols))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
