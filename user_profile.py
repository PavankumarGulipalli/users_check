
def read_events_data(project_directory, end_date, start_date, content_type):
    supported_content_types = ['movie', 'tvepisode', 'vod', 'all']
    if content_type not in supported_content_types:
        raise ValueError('Invalid content type.')
    events_data_directory = project_directory + os.sep + 'events' + os.sep + 'overall_events'
    if os.path.exists(events_data_directory)==False:
        logger.error(f'overall_events folder not present')
        raise FileNotFoundError("overall_events folder not present")
    if len(os.listdir(events_data_directory))==0:
        logger.error(f'overall_events folder is empty')
        raise FileNotFoundError("overall_events folder is empty")
    temp = 1
    temp_folder_name = 'temp_' + str(temp).zfill(5)
    events_temp_directory = events_data_directory + os.sep + temp_folder_name
    while os.path.isdir(events_temp_directory):
        temp += 1
        temp_folder_name = 'temp_' + str(temp).zfill(5)
        events_temp_directory = events_data_directory + os.sep + temp_folder_name
    os.mkdir(events_temp_directory)
    logger.info(msg = f"events {events_temp_directory}")
    days = (end_date - start_date).days + 1  # as both are inclusive
    for day in range(days):
        curr_date = (start_date + datetime.timedelta(days=day)).strftime('%Y%m%d')
        curr_folder = events_data_directory + os.sep + 'temp_' + curr_date +'_'+str(1).zfill(5)
        if os.path.exists(curr_folder):
            for f in os.listdir(curr_folder):
                if 'events_' in f and '.csv' in f:
                    source = curr_folder + os.sep + f
                    destination = events_temp_directory + os.sep + f
                    if os.path.isfile(destination):
                        os.remove(destination)
                    shutil.copyfile(source, destination)
    # reading events data
    #data = dd.read_csv(events_temp_directory + os.sep + 'events_*.csv', **{'dtype': 'object', 'delimiter': '\t'}).compute()
    # List of all matching CSV files
    files1 = glob.glob(os.path.join(events_temp_directory, 'events_*.csv'))
    # Read and concatenate all files, ensuring all columns are of type string
    data = pd.concat(pd.read_csv(f, sep='\t', dtype=object) for f in files1)
    data['streams'] = 1
    data = data.rename(columns={'value': 'mou', 'userid': 'user_id'})
    data['mou'] = data['mou'].astype(float)
    data['streams'] = data['streams'].astype(float)
    data['ts'] = data['ts'].astype('datetime64[ns]')
    #data['ts'] = data['ts'].obj.tz_localize(None)
    data['ts'] = data['ts'] + pd.Timedelta(hours=5, minutes=30) 
    data['hour_segment'] = data['ts'].dt.hour
    data=data.dropna(subset=['content_type', 'user_id', 'hour_segment']).reset_index(drop=True) #change
    data['selectedAudioLanguage']=data['additionalproperties'].apply(getSelectedAudioLanguage)
    data = data[['user_id', 'content_id', 'mou', 'streams', 'hour_segment','selectedAudioLanguage']].copy()
    data = get_mapping_audiolanguage(data)

    if content_type != "all":
        data = data[data['content_type'] == content_type]

    data = data.groupby(['content_id', 'user_id','selectedAudioLanguage']).agg({'mou': np.sum, 'streams': np.sum, 'hour_segment': list}).reset_index()
    func = lambda x: mode(x)
    data['hour_segment'] = data['hour_segment'].apply(func)
    shutil.rmtree(events_temp_directory)
    return data


# Reading deep link events data.
def read_deeplink_events_data(project_directory, end_date, start_date, content_type):
    
    supported_content_types = ['movie', 'tvepisode', 'vod', 'all']
    if content_type not in supported_content_types:
        raise ValueError('Invalid content type.')
    events_data_directory = project_directory + os.sep + 'events' + os.sep + 'overall_events'
    temp = 1
    temp_folder_name = 'temp_' + str(temp).zfill(5)
    events_temp_directory = events_data_directory + os.sep + temp_folder_name
    while os.path.isdir(events_temp_directory):
        temp += 1
        temp_folder_name = 'temp_' + str(temp).zfill(5)
        events_temp_directory = events_data_directory + os.sep + temp_folder_name
    os.mkdir(events_temp_directory)
    logger.info(msg = f"events {events_temp_directory}")
    days = (end_date - start_date).days + 1  # as both are inclusive
    for day in range(days):
        curr_date = (start_date + datetime.timedelta(days=day)).strftime('%Y%m%d')
        curr_folder = events_data_directory + os.sep + 'temp_' + curr_date +'_'+str(1).zfill(5)
        if os.path.exists(curr_folder):
            for f in os.listdir(curr_folder):
                if 'events_' in f and '.csv' in f:
                    source = curr_folder + os.sep + f
                    destination = events_temp_directory + os.sep + f
                    if os.path.isfile(destination):
                        os.remove(destination)
                    shutil.copyfile(source, destination)
    # reading events data
    #data = dd.read_csv(events_temp_directory + os.sep + 'events_*.csv', **{'dtype': 'object', 'delimiter': '\t'}).compute()
    # List of all matching CSV files
    files1 = glob.glob(os.path.join(events_temp_directory, 'events_*.csv'))
    # Read and concatenate all files, ensuring all columns are of type string
    data = pd.concat(pd.read_csv(f, sep='\t', dtype=object) for f in files1)
    data['streams'] = 1
    data = data.rename(columns={'value': 'mou', 'userid': 'user_id'})
    data['mou'] = data['mou'].astype(float)
    data['streams'] = data['streams'].astype(float)
    data['ts'] = data['ts'].astype('datetime64[ns]')
    #data['ts'] = data['ts'].obj.tz_localize(None)
    data['ts'] = data['ts'] + pd.Timedelta(hours=5, minutes=30)
    data['hour_segment'] = data['ts'].dt.hour
    data=data.dropna(subset=['content_type', 'user_id', 'hour_segment']).reset_index(drop=True)
    data['selectedAudioLanguage']=data['additionalproperties'].apply(getSelectedAudioLanguage)
    data = data[['user_id', 'content_id', 'mou', 'streams', 'hour_segment','selectedAudioLanguage']].copy()
    data = get_mapping_audiolanguage(data)                                                                                      
    
    
    
    #getting user selected audio language 
def getSelectedAudioLanguage(x):
    if not isinstance(x,str) or len(x)<=0:
        return ""  
    x=ast.literal_eval(x)
    if 'selectedAudioLanguage' in x.keys() and x['selectedAudioLanguage']!="":
        return x['selectedAudioLanguage']
    else:
        return ""                                   
      
      
    #for each userid and contentid map the selectedauidolanguage with highest mou
def get_mapping_audiolanguage(data):
    # Dictionary to store the maximum 'mou' and corresponding 'selectedAudioLanguage' for each (user_id, content_id)
    max_mou_dict = {}
    
    for _, row in data.iterrows():
        key = (row['user_id'], row['content_id'])
        if key not in max_mou_dict or row['mou'] > max_mou_dict[key][0]:
            max_mou_dict[key] = (row['mou'], row['selectedAudioLanguage'])
    
    # Extracting 'selectedAudioLanguage' from the dictionary
    mapping_audiolanguage = {k: v[1] for k, v in max_mou_dict.items()}
    
    # Adding 'selectedAudioLanguage' column to data based on mapping_audiolanguage
    data['selectedAudioLanguage'] = data.apply(lambda row: mapping_audiolanguage[(row['user_id'], row['content_id'])], axis=1)
    
    return data