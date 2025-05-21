import ast
import polars as pl
import os
import shutil
import glob
from datetime import datetime, timedelta, date

# getting user selected audio language 
def getSelectedAudioLanguage(x: str) -> str:
    if not isinstance(x, str) or len(x) <= 0:
        return ""  
    try:
        x_eval = ast.literal_eval(x) # x should be a string representation of a dictionary
        if isinstance(x_eval, dict) and 'selectedAudioLanguage' in x_eval.keys() and x_eval['selectedAudioLanguage'] != "":
            return x_eval['selectedAudioLanguage']
        else:
            return ""
    except (ValueError, SyntaxError): # Handles cases where x is not a valid dict string representation
        return ""

# for each userid and contentid map the selectedaudiolanguage with highest mou
def get_mapping_audiolanguage(data_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Updates the 'selectedAudioLanguage' column in a Polars LazyFrame.
    For each (user_id, content_id) pair, the selectedAudioLanguage is determined
    by the one associated with the maximum 'mou' (measure of usage).
    If there are multiple languages with the same maximum 'mou', the first one encountered
    after internal Polars grouping (which might involve sorting or hashing) is chosen.

    Args:
        data_lf: A Polars LazyFrame containing at least 'user_id', 'content_id',
                 'mou', and 'selectedAudioLanguage' columns. 'mou' should be numeric.

    Returns:
        A new Polars LazyFrame with the 'selectedAudioLanguage' column updated
        based on the highest 'mou' for each user-content pair.
    """
    mapping_lf = (
        data_lf
        .with_columns(
            pl.col("mou").max().over(["user_id", "content_id"]).alias("max_mou_in_group")
        )
        .filter(pl.col("mou") == pl.col("max_mou_in_group"))
        .group_by(["user_id", "content_id"], maintain_order=False)
        .agg(
            pl.col("selectedAudioLanguage").first().alias("audio_lang_with_max_mou")
        )
    )

    updated_lf = data_lf.join(
        mapping_lf,
        on=["user_id", "content_id"],
        how="left"
    ).with_columns(
        pl.col("audio_lang_with_max_mou").alias("new_selectedAudioLanguage")
    ).drop("selectedAudioLanguage", "max_mou_in_group", "audio_lang_with_max_mou") \
     .rename({"new_selectedAudioLanguage": "selectedAudioLanguage"})

    return updated_lf

def read_events_data(project_directory: str, end_date: date, start_date: date, content_type: str) -> pl.LazyFrame:
    supported_content_types = ['movie', 'tvepisode', 'vod', 'all']
    if content_type not in supported_content_types:
        raise ValueError('Invalid content type.')

    events_data_directory = os.path.join(project_directory, 'events', 'overall_events')
    if not os.path.exists(events_data_directory):
        print(f'Error: overall_events folder not present at {events_data_directory}')
        raise FileNotFoundError("overall_events folder not present")
    if not os.listdir(events_data_directory):
        print(f'Error: overall_events folder is empty at {events_data_directory}')
        raise FileNotFoundError("overall_events folder is empty")

    temp = 1
    temp_folder_name = f'temp_{str(temp).zfill(5)}'
    events_temp_directory = os.path.join(events_data_directory, temp_folder_name)
    while os.path.isdir(events_temp_directory):
        temp += 1
        temp_folder_name = f'temp_{str(temp).zfill(5)}'
        events_temp_directory = os.path.join(events_data_directory, temp_folder_name)
    
    os.makedirs(events_temp_directory, exist_ok=True)
    print(f"Temporary events directory created at: {events_temp_directory}")

    days = (end_date - start_date).days + 1
    for day_offset in range(days):
        curr_date_obj = start_date + timedelta(days=day_offset)
        curr_date_str = curr_date_obj.strftime('%Y%m%d')
        curr_folder = os.path.join(events_data_directory, f'temp_{curr_date_str}_{str(1).zfill(5)}')
        if os.path.exists(curr_folder):
            for f in os.listdir(curr_folder):
                if 'events_' in f and '.csv' in f:
                    source = os.path.join(curr_folder, f)
                    destination = os.path.join(events_temp_directory, f)
                    if os.path.isfile(destination):
                        os.remove(destination)
                    shutil.copyfile(source, destination)
    
    all_expected_columns = [
        'userid', 'projectid', 'content_id', 'content_type', 'originmedium', 
        'originname', 'originsource', 'trackingid', 'carouselid', 'device', 
        'value', 'ts', 'additionalproperties', 'tabname', 'carouselposition', 
        'contentposition', 'Updated_Type'
    ]
    schema = {col: pl.Utf8 for col in all_expected_columns}
    csv_files_pattern = os.path.join(events_temp_directory, 'events_*.csv')
    
    data_lf = pl.scan_csv(
        csv_files_pattern,
        separator='\t',
        schema=schema,
        infer_schema_length=0,
        ignore_errors=True
    )
    
    data_lf = data_lf.with_columns(pl.lit(1).cast(pl.Float64).alias('streams'))
    data_lf = data_lf.rename({'value': 'mou', 'userid': 'user_id'})
    data_lf = data_lf.with_columns(pl.col('mou').cast(pl.Float64, strict=False))
    data_lf = data_lf.with_columns(
        pl.col('ts').str.to_datetime(errors='coerce')
        .dt.offset_by("5h30m")
        .alias('ts')
    )
    data_lf = data_lf.with_columns(pl.col('ts').dt.hour().alias('hour_segment'))
    data_lf = data_lf.drop_nulls(subset=['content_type', 'user_id', 'hour_segment'])
    data_lf = data_lf.with_columns(
        pl.col('additionalproperties').apply(getSelectedAudioLanguage, return_dtype=pl.Utf8)
        .alias('selectedAudioLanguage')
    )
    
    # 'content_type' is needed for filtering if content_type != 'all'
    # It's not in the final select list for read_deeplink_events_data
    columns_to_select_intermediate = ['user_id', 'content_id', 'mou', 'streams', 'hour_segment', 'selectedAudioLanguage', 'content_type']
    data_lf = data_lf.select(columns_to_select_intermediate)
    
    data_lf = get_mapping_audiolanguage(data_lf)

    if content_type != "all":
        data_lf = data_lf.filter(pl.col('content_type') == content_type)
    
    # Final selection for read_events_data before aggregation
    final_columns = ['user_id', 'content_id', 'mou', 'streams', 'hour_segment','selectedAudioLanguage']
    data_lf = data_lf.select(final_columns) # content_type is dropped here if not explicitly selected

    # Aggregation and mode calculation for read_events_data
    data_lf_agg = data_lf.group_by(['content_id', 'user_id', 'selectedAudioLanguage'], maintain_order=False).agg([
        pl.sum('mou').alias('mou'),
        pl.sum('streams').alias('streams'),
        pl.col('hour_segment').implode().alias('hour_segment_list')
    ])
    data_lf_agg = data_lf_agg.with_columns(
        pl.col('hour_segment_list').list.eval(pl.element().mode().first()).alias('hour_segment')
    ).drop('hour_segment_list')

    try:
        shutil.rmtree(events_temp_directory)
        print(f"Successfully removed temporary directory: {events_temp_directory}")
    except Exception as e:
        print(f"Error removing temporary directory {events_temp_directory}: {e}")

    return data_lf_agg.lazy()


def read_deeplink_events_data(project_directory: str, end_date: date, start_date: date, content_type: str) -> pl.LazyFrame:
    supported_content_types = ['movie', 'tvepisode', 'vod', 'all']
    if content_type not in supported_content_types:
        raise ValueError('Invalid content type.')

    events_data_directory = os.path.join(project_directory, 'events', 'overall_events')
    if not os.path.exists(events_data_directory):
        print(f'Error: overall_events folder not present at {events_data_directory}')
        raise FileNotFoundError("overall_events folder not present")
    if not os.listdir(events_data_directory):
        print(f'Error: overall_events folder is empty at {events_data_directory}')
        raise FileNotFoundError("overall_events folder is empty")

    temp = 1
    temp_folder_name = f'temp_{str(temp).zfill(5)}'
    events_temp_directory = os.path.join(events_data_directory, temp_folder_name)
    while os.path.isdir(events_temp_directory):
        temp += 1
        temp_folder_name = f'temp_{str(temp).zfill(5)}'
        events_temp_directory = os.path.join(events_data_directory, temp_folder_name)
    
    os.makedirs(events_temp_directory, exist_ok=True)
    print(f"Temporary events directory created at: {events_temp_directory}")

    days = (end_date - start_date).days + 1
    for day_offset in range(days):
        curr_date_obj = start_date + timedelta(days=day_offset)
        curr_date_str = curr_date_obj.strftime('%Y%m%d')
        curr_folder = os.path.join(events_data_directory, f'temp_{curr_date_str}_{str(1).zfill(5)}')
        if os.path.exists(curr_folder):
            for f in os.listdir(curr_folder):
                if 'events_' in f and '.csv' in f:
                    source = os.path.join(curr_folder, f)
                    destination = os.path.join(events_temp_directory, f)
                    if os.path.isfile(destination):
                        os.remove(destination)
                    shutil.copyfile(source, destination)
    
    all_expected_columns = [
        'userid', 'projectid', 'content_id', 'content_type', 'originmedium', 
        'originname', 'originsource', 'trackingid', 'carouselid', 'device', 
        'value', 'ts', 'additionalproperties', 'tabname', 'carouselposition', 
        'contentposition', 'Updated_Type'
    ]
    schema = {col: pl.Utf8 for col in all_expected_columns}
    csv_files_pattern = os.path.join(events_temp_directory, 'events_*.csv')
    
    # Step 2: CSV Reading
    data_lf = pl.scan_csv(
        csv_files_pattern,
        separator='\t',
        schema=schema,
        infer_schema_length=0,
        ignore_errors=True
    )
    
    # Step 3: Add 'streams' column and Step 5: Cast to Float64
    data_lf = data_lf.with_columns(pl.lit(1).cast(pl.Float64).alias('streams'))

    # Step 4: Rename columns
    data_lf = data_lf.rename({'value': 'mou', 'userid': 'user_id'})

    # Step 5: Cast 'mou' to Float64
    data_lf = data_lf.with_columns(pl.col('mou').cast(pl.Float64, strict=False))

    # Step 6: Datetime casting and offset
    data_lf = data_lf.with_columns(
        pl.col('ts').str.to_datetime(errors='coerce')
        .dt.offset_by("5h30m") # Using Polars duration string
        .alias('ts')
    )

    # Step 7: Extract 'hour_segment'
    data_lf = data_lf.with_columns(pl.col('ts').dt.hour().alias('hour_segment'))

    # Step 8: Drop nulls
    data_lf = data_lf.drop_nulls(subset=['content_type', 'user_id', 'hour_segment'])

    # Step 9: Apply 'getSelectedAudioLanguage'
    data_lf = data_lf.with_columns(
        pl.col('additionalproperties').apply(getSelectedAudioLanguage, return_dtype=pl.Utf8)
        .alias('selectedAudioLanguage')
    )

    # Select columns needed for get_mapping_audiolanguage and potential content_type filtering
    # This is before the final select specified in step 10
    columns_for_processing = ['user_id', 'content_id', 'mou', 'streams', 'hour_segment', 'selectedAudioLanguage', 'content_type']
    data_lf = data_lf.select(columns_for_processing)

    # Step 11: Apply get_mapping_audiolanguage
    data_lf = get_mapping_audiolanguage(data_lf) # Assumes this function handles Polars LazyFrames

    # Filter by 'content_type' (this was not explicitly in deeplink's pandas version, but good to keep consistent if logic applies)
    # The original pandas read_deeplink_events_data did NOT have this content_type filter.
    # Let's check the original user_profile.py for read_deeplink_events_data.
    # The pandas version of read_deeplink_events_data does *not* filter by content_type.
    # So, I will REMOVE this filter for read_deeplink_events_data.
    # if content_type != "all": # This was in read_events_data, but not in original read_deeplink_events_data
    #     data_lf = data_lf.filter(pl.col('content_type') == content_type)

    # Step 10: Select final columns
    # The original pandas version of read_deeplink_events_data selects:
    # data = data[['user_id', 'content_id', 'mou', 'streams', 'hour_segment','selectedAudioLanguage']].copy()
    final_columns_to_select = ['user_id', 'content_id', 'mou', 'streams', 'hour_segment', 'selectedAudioLanguage']
    data_lf = data_lf.select(final_columns_to_select)
    
    # Step 12: No Groupby/Agg/Mode - this is the key difference from read_events_data

    # Step 13: Temporary Directory Cleanup
    try:
        shutil.rmtree(events_temp_directory)
        print(f"Successfully removed temporary directory: {events_temp_directory}")
    except Exception as e:
        print(f"Error removing temporary directory {events_temp_directory}: {e}")

    # Step 14: Return Type
    return data_lf.lazy() # Ensure it's a LazyFrame

# Example usage (for testing, not part of the script typically)
if __name__ == '__main__':
    # This block would not run in the agent's environment but is useful for local testing.
    print("Running a dummy example for read_deeplink_events_data (if this were executed locally)...")
    
    test_proj_dir = "test_project_data_deeplink"
    overall_events_dir = os.path.join(test_proj_dir, "events", "overall_events")
    
    if os.path.exists(test_proj_dir):
        shutil.rmtree(test_proj_dir)
    
    start_date_obj = date(2023, 1, 1)
    end_date_obj = date(2023, 1, 1)
    
    date_str = start_date_obj.strftime('%Y%m%d')
    event_day_folder = os.path.join(overall_events_dir, f'temp_{date_str}_{str(1).zfill(5)}')
    os.makedirs(event_day_folder, exist_ok=True)
    
    dummy_csv_path = os.path.join(event_day_folder, "events_dl_01.csv")
    with open(dummy_csv_path, 'w') as f:
        header = "\t".join([
            'userid', 'projectid', 'content_id', 'content_type', 'originmedium', 
            'originname', 'originsource', 'trackingid', 'carouselid', 'device', 
            'value', 'ts', 'additionalproperties', 'tabname', 'carouselposition', 
            'contentposition', 'Updated_Type'
        ])
        f.write(header + "\n")
        row1 = "\t".join([
            "user1", "projA", "c001", "movie", "medium1", "name1", "source1", "track1", 
            "carouselA", "phone", "120.5", "2023-01-01T10:00:00", 
            "{'selectedAudioLanguage': 'English', 'otherProp': 'val'}", 
            "tab1", "1", "2", "typeA"
        ])
        f.write(row1 + "\n")
        row2 = "\t".join([ # Same user, same content, different mou and language initially
            "user1", "projA", "c001", "movie", "medium1", "name1", "source1", "track1", 
            "carouselA", "phone", "150.0", "2023-01-01T11:00:00", 
            "{'selectedAudioLanguage': 'Hindi'}", 
            "tab1", "1", "2", "typeA"
        ])
        f.write(row2 + "\n")
        row3_different_user = "\t".join([
            "user2", "projB", "c002", "tvepisode", "medium2", "name2", "source2", "track2",
            "carouselB", "tv", "100.0", "2023-01-01T12:00:00",
            "{'selectedAudioLanguage': 'English'}",
            "tab2", "2", "3", "typeB"
        ])
        f.write(row3_different_user + "\n")

    print(f"Dummy data created in {test_proj_dir}")

    try:
        lazy_frame_dl = read_deeplink_events_data(
            project_directory=test_proj_dir,
            start_date=start_date_obj,
            end_date=end_date_obj,
            content_type="all" # Original deeplink did not filter by content_type
        )
        
        result_df_dl = lazy_frame_dl.collect()
        print("\nResulting DataFrame (Deeplink):")
        print(result_df_dl)
        # Expected: 3 rows. For user1, c001, selectedAudioLanguage should be 'Hindi' for both rows.
        # Row for user2, c002 should be as is with 'English'.
        # Columns: user_id, content_id, mou, streams, hour_segment, selectedAudioLanguage
        
    except FileNotFoundError as e:
        print(f"Test execution failed (Deeplink): {e}")
    except Exception as e:
        print(f"An unexpected error occurred during test (Deeplink): {e}")
    finally:
        if os.path.exists(test_proj_dir):
            shutil.rmtree(test_proj_dir)
        print(f"Cleaned up dummy data directory: {test_proj_dir}")
# End of example usage
