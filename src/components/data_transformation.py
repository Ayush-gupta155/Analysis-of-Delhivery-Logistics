import numpy as np
import pandas as pd
import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger  import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from src.components.target_encoding_pipeline import TargetEncoding

from src.utils import save_object



# print('line13')
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact','Preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    # df = pd.read_csv('artifact\data.csv')
    # print(df)
    logging.info("loaded raw df")

    # Convert data type to date_time columnns 
    def convert_datetime_columns(self, df):
        date_time_columns = ['trip_creation_time', 'od_start_time', 'od_end_time', 'cutoff_timestamp']
        for column in date_time_columns:
                df[column] = pd.to_datetime(df[column], format='mixed')
                df[column] = df[column].dt.floor('s')
        return df

    # Drop the null values
    def drop_null_rows(self, df):
        return df.dropna()
    
    def rename_columns(self,df):
        df.rename(columns = { 'actual_distance_to_destination':'actual_distance_to_destination_cumulative',
                     'actual_time':'actual_time_cumulative',
                   "osrm_time": "osrm_time_cumulative",
                    "osrm_distance":"osrm_distance_cumulative",'start_scan_to_end_scan' : 'Trip_duration(mins)'} , inplace = True)
        return df 
    
    def creating_new_features(self,df):
        trip_durations = df.groupby(['trip_uuid','source_center','destination_center'])['actual_time_cumulative'].max()
        Actual_trip_time = trip_durations.groupby('trip_uuid').sum()

        trip_distance = df.groupby(['trip_uuid','source_center','destination_center'])['actual_distance_to_destination_cumulative'].max()
        Total_trip_distance = trip_distance.groupby('trip_uuid').sum()
        
        trip_summary = pd.DataFrame({
        'Actual_trip_time': Actual_trip_time,
        'Total_trip_distance': Total_trip_distance})

        trip_summary['Actual_Average_Speed'] = trip_summary['Total_trip_distance']/trip_summary['Actual_trip_time']
        trip_summary = trip_summary.reset_index()

        df = df.merge(
        trip_summary[['trip_uuid','Actual_Average_Speed', 'Total_trip_distance', 'Actual_trip_time']],
        on=['trip_uuid'],
        how='left'
        )

        # Average OSRM Time taken to complete a delivery
        OSRM_trip_durations = df.groupby(['trip_uuid','source_center','destination_center'])['osrm_time_cumulative'].max()
        Total_OSRM_trip_duration = OSRM_trip_durations.groupby('trip_uuid').sum()

        # Average OSRM distance to complete a delivery
        OSRM_trip_distance = df.groupby(['trip_uuid','source_center','destination_center'])['osrm_distance_cumulative'].max()
        Total_OSRM_trip_distance = OSRM_trip_distance.groupby('trip_uuid').sum()

        OSRM_trip_summary = pd.DataFrame({
        'Total_osrm_trip_duration': Total_OSRM_trip_duration,
        'Total_osrm_trip_distance': Total_OSRM_trip_distance})

        OSRM_trip_summary['Osrm_Average_Speed'] = OSRM_trip_summary['Total_osrm_trip_distance']/OSRM_trip_summary['Total_osrm_trip_duration']

        #Reset the indexes and merging 'Total_osrm_trip_duration' and 'Total_osrm_trip_distance' into main df.
        OSRM_trip_summary = OSRM_trip_summary.reset_index()
        df = df.merge(
        OSRM_trip_summary[['trip_uuid','Osrm_Average_Speed','Total_osrm_trip_distance','Total_osrm_trip_duration']],
        on=['trip_uuid'],
        how='left')

        # Let's find the number of stops in each trip. This can be a prominent feature for predicting Actual Time.
        # This can be found out by taking the sum of all the unique occurrences of each trip_uuid.
        trip_stop_counts = df['trip_uuid'].value_counts()
        df['No_of_stops'] = df['trip_uuid'].map(trip_stop_counts)

        # Creating a new feature column Trip_Total_duration(mins), which will tell us how much time each trip takes from start to end 
        Trip_duration = df.groupby(['trip_uuid', 'source_center', 'destination_center'])['Trip_duration(mins)'].max()
        Trip_total_duration  = Trip_duration.groupby('trip_uuid').sum()
        df['Trip_total_duration(mins)'] = df['trip_uuid'].map(Trip_total_duration)

        return df
    
    def find_outlier(self,df):

        col = ['actual_time_cumulative', 'osrm_time_cumulative', 'actual_distance_to_destination_cumulative', 'segment_actual_time'
                 , 'osrm_distance_cumulative','segment_osrm_time','segment_osrm_distance','Actual_Average_Speed','Total_trip_distance',
                 'Actual_trip_time','Osrm_Average_Speed','Total_osrm_trip_distance','No_of_stops','Trip_total_duration(mins)']
        for column in col:
    
            percentile25 = df[column].quantile(.25)
            percentile75 = df[column].quantile(.75)
            IQR = percentile75 - percentile25
            upperlimit = percentile75 + 1.5*IQR
            lowerlimit = percentile25 - 1.5*IQR

            # Cap values above the upper limit
            df[column] = np.where(df[column] > upperlimit, upperlimit, df[column])
        
            # Drop rows with values below the lower limit
            df = df[df[column] >= lowerlimit]

        return df

    def drop_columns(self,df):
        df = df.drop(columns=['data', 'route_schedule_uuid','trip_uuid','source_name','destination_name'])
        return df
    def handling_categorical_features(self,df):

        le = LabelEncoder()
        df['route_type'] = le.fit_transform(df['route_type'])
        return df

    def date_time_features(self,df):
        datetime_cols= ['od_end_time','od_start_time','trip_creation_time','cutoff_timestamp']
        for col in datetime_cols:
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day

        # Dropping the original Datetime features from the dataset
        df.drop(columns = ['od_end_time','od_start_time','trip_creation_time','cutoff_timestamp'], inplace = True)

        return df
    
    def run_pipeline(self, df):
        df = self.convert_datetime_columns(df)
        df = self.drop_null_rows(df)
        df = self.rename_columns(df)
        df = self.creating_new_features(df)
        df = self.find_outlier(df)
        df = self.drop_columns(df)
        df = self.handling_categorical_features(df)
        df = self.date_time_features(df)
        return df

    
    def initiate_data_transformation(self,raw_path):
        try:
            # Read the raw data.
            raw_df = pd.read_csv(raw_path)

            logging.info("Cleaned dataframe is instantiated")

            cleaned_df=self.run_pipeline(raw_df)

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=cleaned_df)
            
            return (
                cleaned_df,
                    self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)      
    


