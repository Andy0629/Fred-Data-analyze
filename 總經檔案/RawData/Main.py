import os
import pandas as pd
import numpy as np
from urllib.parse import quote
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from full_fred.fred import Fred
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class DeleteData:   #刪除檔案中的原始檔案
    def __init__(self, output_path):
        self.directory = output_path
    
    def delete_csv_files(self):
        if os.path.isdir(self.directory):
            files = os.listdir(self.directory)
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(self.directory, file)
                    os.remove(file_path)
                    print(f'文件 {file_path} 已删除')
        else:
            print(f'{self.directory} 為錯誤路徑')

class CrawlData:
    def __init__(self, data=None, name=None, start_date=None, end_date=None, change_freq=None, FREQ=None, days=None, columns_name=None, observation_start=None, realtime_start=None, Fred_file=None, Fred_path=None):
        self.name = name
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.change_freq = change_freq
        self.FREQ = FREQ
        self.days = days
        self.columns_name = columns_name
        self.observation_date = observation_start
        self.realtime_start = observation_start
        self.Fred_file = Fred_file
        self.Fred_path = Fred_path
        self.error_data = []

        if Fred_path is None:
            self.Fred_path = os.path.join(os.path.join(os.getcwd(),'FRED_SImply') ,'FRED_Key.txt')

    def get_fred_data(self):
        with open(self.Fred_path, 'r') as file:
            api_key = file.readline().strip()
        FRED = Fred(self.Fred_path)
        try:
            data_df = FRED.get_series_df(self.name, observation_start="2005-01-01", realtime_start="1999-01-01")
        except Exception as e:
            print(f"Error occurred while fetching data for {self.name}: {e}")
            error_data = {'備註': [f"Error occurred while fetching data for {self.name}: {e}"]}
            return error_data

        if data_df is not None:
            if 'date' in data_df.columns:
                data_df = data_df.groupby('date').head(1)
            else:
                print("DataFrame中沒有名為 'date' 的欄位")
            if 'observations' not in data_df.columns:
                print("DataFrame中没有名為 'observations' 的列")
            data_df = data_df.groupby('date').head(1)
            data = data_df.rename(columns={'value': self.columns_name})
            data[self.columns_name] = pd.to_numeric(data[self.columns_name], errors='coerce').round(3)
            data['Frequency'] = self.FREQ
            return data
        else:
            print("get_series_df() 返回 None")

    def get_error_data(self):
        return self.error_data

class ChangeFrequency:
    def __init__(self, data=None, days=None, columns_name=None, change_freq=None, Frequency=None):
        self.data = data
        self.columns_name = columns_name
        self.change_freq_options = ['d', 'w', 'm', 'q']
        self.change_freq = change_freq
        self.FREQ = Frequency

    def convert_frequency(self):
        df1 = self.data.iloc[:, [0, 2, 3, 4]].copy()
        df2 = self.data.iloc[:, [0, 2, 3, 4]].copy()

        if self.FREQ == 'D':
            df1['date'] = pd.to_datetime(df1['date'])
            df1.set_index('date', inplace=True)
            resampled_data = df1.resample(self.change_freq).ffill()
            resampled_data.index.name = 'Date'
        else:
            if df2['realtime_start'].duplicated().any():
                df2['original_date'] = df2['date']
                non_duplicate_dates = df2[~df2.duplicated(subset=['realtime_start'], keep=False)]
                non_duplicate_dates['realtime_start'] = pd.to_datetime(non_duplicate_dates['realtime_start'])
                non_duplicate_dates['date'] = pd.to_datetime(non_duplicate_dates['date'])
                delta = non_duplicate_dates['realtime_start'] - non_duplicate_dates['date']
                duplicated_dates = df2[df2.duplicated(subset=['realtime_start'], keep=False)]
                delta_cleaned = delta.dropna()
                if not delta_cleaned.empty:
                    delta_mean = delta_cleaned.mean()
                    df2['date'] = pd.to_datetime(df2['date'])
                    for rt_start in duplicated_dates['realtime_start'].unique():
                        indices = df2['realtime_start'] == rt_start
                        df2.loc[indices, 'realtime_start_update'] = df2.loc[indices, 'date'] + delta_mean
                    non_duplicated_dates = df2[~df2.index.isin(duplicated_dates.index)]
                    df2.loc[non_duplicated_dates.index, 'realtime_start_update'] = non_duplicated_dates['realtime_start']
                    df2.drop(columns=['original_date'], inplace=True)
                    df2.set_index('realtime_start_update', inplace=True, drop=False)
                    df2['realtime_start_update'] = pd.to_datetime(df2['realtime_start_update'])
                    df2.set_index('realtime_start_update', inplace=True)
                    df2.index.name = 'Date'
            else:
                df2.set_index('realtime_start', inplace=True, drop=False)
                df2['realtime_start'] = pd.to_datetime(df2['realtime_start'])
                df2.set_index('realtime_start', inplace=True)
                df2.index.name = 'Date'

            if self.FREQ == self.change_freq:
                resampled_data = df2
                resampled_data.index = pd.to_datetime(resampled_data.index)
                resampled_data.index = resampled_data.index.date
                resampled_data.index.name = 'Date'
            else:
                resampled_data = df2.resample(self.change_freq).ffill()
                resampled_data.index = resampled_data.index.date
                resampled_data.index.name = 'Date'

        return resampled_data

class SignalPoints1:    #若本期數值>上期數值，為1；反之，為0。
    def __init__(self, data=None, days=None, columns_name=None, change_freq=None):
        self.data = data
        self.days = days
        self.columns_name = columns_name

    def signal_points_1(self):
        self.data.set_index('Date', inplace=True)
        self.data.drop(columns=['date', 'realtime_start', 'Frequency'], inplace=True, errors='ignore')
        z = self.data.copy()
        z.index = self.data.index.copy()
        new_zz = z.iloc[:, 0].values[1:] - z.iloc[:, 0].values[0:-1]
        new_zz = pd.DataFrame(new_zz, index=z.index[1:])
        z['signal_points'] = (new_zz > 0).astype(int)
        z.index = pd.to_datetime(z.index)
        z = z.resample('D').fillna(method='ffill')
        result_df = z.copy()
        return result_df

class SignalPoints2:    #若本期數值<上期數值，為1；反之，為0。
    def __init__(self, data=None, days=None, columns_name=None, change_freq=None):
        self.data = data
        self.days = days
        self.columns_name = columns_name

    def signal_points_2(self):
        self.data.drop(columns=['date', 'realtime_start', 'Frequency'], inplace=True, errors='ignore')
        z = self.data.copy()
        z.index = self.data.index.copy()
        new_zz = z.iloc[:, 0].values[1:] - z.iloc[:, 0].values[0:-1]
        new_zz = pd.DataFrame(new_zz, index=z.index[1:])
        z['signal_points'] = ((new_zz < 0)).astype(int)
        z.index = pd.to_datetime(z.index)
        z = z.resample('D').fillna(method='ffill')
        result_df = z.copy()
        return result_df

class SignalPoints3:    #若本期數值/上期數值>1，為1；反之，為0。
    def __init__(self, data=None, days=None, columns_name=None, change_freq=None):
        self.data = data
        self.days = days
        self.columns_name = columns_name

    def signal_points_3(self):
        self.data.drop(columns=['date', 'realtime_start', 'Frequency'], inplace=True, errors='ignore')
        z = self.data.copy()
        z.index = self.data.index.copy()
        new_zz = z.iloc[:, 0].values[1:] / z.iloc[:, 0].values[0:-1]
        new_zz = pd.DataFrame(new_zz, index=z.index[1:])
        z['signal_points'] = ((new_zz > 1)).astype(int)
        z.index = pd.to_datetime(z.index)
        z = z.resample('D').fillna(method='ffill')
        result_df = z.copy()
        return result_df

class SignalPoints4:    #若本期數值/上期數值<1，為1；反之，為0。
    def __init__(self, data=None, days=None, columns_name=None, change_freq=None):
        self.data = data
        self.days = days
        self.columns_name = columns_name

    def signal_points_4(self):
        self.data.drop(columns=['date', 'realtime_start', 'Frequency'], inplace=True, errors='ignore')
        z = self.data.copy()
        z.index = self.data.index.copy()
        new_zz = z.iloc[:, 0].values[1:] / z.iloc[:, 0].values[0:-1]
        new_zz = pd.DataFrame(new_zz, index=z.index[1:])
        z['signal_points'] = ((new_zz < 1)).astype(int)
        z.index = pd.to_datetime(z.index)
        z = z.resample('D').fillna(method='ffill')
        result_df = z.copy()
        return result_df
class merged_csv():     #將實際買賣點和預測買賣點合併
    def __init__(self, input_filename1=None, input_filename2=None, data=None):
        self.input_filename1 = input_filename1
        self.input_filename2 = input_filename2
        self.data = data

    def merged_csv(self):
        try:
            data1_df = pd.read_csv(self.input_filename1, encoding="utf-8-sig", index_col=0)
        except:
            data1_df = pd.read_csv(self.input_filename1, encoding="Big5", index_col=0)

        try:
            data2_df = pd.read_csv(self.input_filename2, encoding="utf-8-sig", index_col=0)
        except:
            data2_df = pd.read_csv(self.input_filename2, encoding="utf-8-sig", index_col=0)

        data1_df.index = (pd.to_datetime(data1_df.index)).strftime('%Y-%m-%d')
        data2_df.index = (pd.to_datetime(data2_df.index)).strftime('%Y-%m-%d')

        if self.data is not None and isinstance(self.data, (pd.DataFrame, pd.Series)):
            self.data.index = pd.to_datetime(self.data.index).strftime('%Y-%m-%d')
            data_df = self.data.fillna(method='ffill')
            data_df = pd.DataFrame(self.data)
            merged_data = data1_df.join(data_df)
        else:
            data1_df = data1_df.fillna(method='ffill')
            data2_df = data2_df.fillna(method='ffill')
            merged_data = data1_df.join(data2_df)

        data = merged_data.fillna(method='ffill')
        return data

class Confusion_Matrix():   #產出混淆矩陣和報表 #報表內容分別有id、Ticker、中文名稱、資料轉換方式、訊號判別式、多空、開始日期、結束日期、準確率、精確率、召回率、F1分數
    current_id = 0
    def __init__(self, data=None, columns_name=None, ID=None, Ticker=None, change_freq=None, signal_points=None, longshort=None, Datestart=None, Date_End=None, y_true=None, y_pred=None, counter=None, chinese_columns=None):
        self.data = data
        self.columns_name = columns_name
        self.ID = ID or self.generate_unique_id()
        self.Ticker = Ticker
        self.change_freq = change_freq
        self.signal_points = signal_points
        self.longshort = longshort
        self.Datestart = Datestart
        self.Date_End = Date_End
        self.counter = 1
        self.error_data = pd.DataFrame(columns=[self.columns_name])
        self.chinese_columns = chinese_columns


    @classmethod
    def generate_unique_id(cls):
        ID = cls.current_id
        cls.current_id += 1
        return ID
    
    def find_chinese_columns(self):
        for index, row in self.chinese_columns.iterrows():
            if row['Ticker'] == self.Ticker:
                return row['chinese_columns']
        return None

    def variable(self, filename):
        parts = filename.split("_")
        signal_points = "_".join(parts[0:3])
        Ticker = parts[3]
        change_freq = parts[-1].split('.')[0]
        return signal_points, Ticker, change_freq

    def time(self, data):
        DateStart = data.index.min()
        DateEnd = data.index.max()
        return DateStart, DateEnd

    def Confusion_Matrix(self, data):
        try:
            chinese_columns = self.find_chinese_columns()
            if self.longshort == 'long':
                self.y_true = np.array(self.data['波段低點區間'])
            elif self.longshort == 'short':
                self.y_true = np.array(self.data['波段高點區間'])
            else:
                raise ValueError("Invalid value for 'longshort' attribute")
            self.y_pred = np.array(self.data['signal_points'])
            self.y_pred[np.isnan(self.y_pred)] = 0

            Result = {}
            result = {}
            confusion_matrix_result = confusion_matrix(self.y_true, self.y_pred)
            accuracy = accuracy_score(self.y_true, self.y_pred)
            precision = precision_score(self.y_true, self.y_pred)
            recall = recall_score(self.y_true, self.y_pred)
            f1 = f1_score(self.y_true, self.y_pred)
            cross_tab = pd.crosstab(pd.Series(self.y_true, name='Actual'), pd.Series(self.y_pred, name='Predicted'), dropna=False)
            print("Confusion Matrix:")
            print(cross_tab)

            result['ID'] = self.ID
            result['Ticker'] = self.Ticker
            result['chinese_columns'] = chinese_columns
            result['資料轉換方式'] = self.change_freq
            result['signal_points'] = self.signal_points
            result['LongShort'] = self.longshort
            result['Datestart'] = self.Datestart
            result['Date_End'] = self.Date_End
            result['0_points'] = cross_tab.iloc[0, 0]
            if len(np.unique(self.y_pred)) > 1:
                result['1_points'] = cross_tab.iloc[1, 1]
            else:
                result['1_points'] = None
            result['accuracy'] = accuracy
            result['precision'] = precision
            result['recall'] = recall
            result['f1_score'] = f1
        except Exception as e:
            result = {
                'ID': self.ID,
                'Ticker': self.Ticker,
                'chinese_columns': self.chinese_columns,
                '資料轉換方式': self.change_freq,
                'signal_points': self.signal_points,
                'LongShort': self.longshort,
                'Datestart': self.Datestart,
                'Date_End': self.Date_End,
                '0_points': None,
                '1_points': None,
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1_score': None,
                'Error': str(e),
            }

        Result[self.columns_name] = result
        df_Result = pd.DataFrame(Result)
        custom_order = ['ID', 'Ticker', 'chinese_columns', '資料轉換方式', 'signal_points', 'LongShort', 'Datestart', 'Date_End', '0_points', '1_points', 'accuracy', 'precision', 'recall', 'f1_score']
        df_Result = df_Result.reindex(custom_order)
        return df_Result

    def GetResult(self):
        return pd.DataFrame(self.Result)
def execute_code():
    csv_file_path = os.path.join(os.getcwd(),"RawData")
    txt_file_path = os.path.join(os.getcwd(),'FRED_SImply')
    file_path = os.path.join(csv_file_path, 'Fred_Ticker.csv')
    Fred_path = os.path.join(txt_file_path, 'FRED_Key.txt')

    input_Ticker = []
    input_column_name = []
    FREQ = []

    df = pd.read_csv(file_path, encoding='utf-8-sig')
    input_Ticker = df['Ticker'].tolist()
    input_column_name = df['名稱'].tolist()
    FREQ = df['freq'].tolist()

    if True:
        for i in range(len(input_Ticker)):
            crawl_instance = CrawlData(
                name=input_Ticker[i],
                columns_name=input_column_name[i],
                observation_start="2005-01-01",
                realtime_start="1998-01-01",
                FREQ=FREQ[i],
                days=1
            )
            observation = []
            fred_data = crawl_instance.get_fred_data()
            if isinstance(fred_data, dict):
                error_data = fred_data
                df.loc[df['Ticker'] == input_Ticker[i], '備註'] = "Error"
                file_name = 'Data_output.csv'.replace(' ', " ")
                output_path = os.path.join(csv_file_path, file_name)
                df.to_csv(output_path, index=False, encoding="utf-8-sig", date_format='%Y/%m/%d')
            else:
                data = pd.DataFrame(fred_data)
                file_name = f"{input_Ticker[i]}_{input_column_name[i]}.csv".replace(' ', " ")
                output_path = os.path.join(os.path.join(csv_file_path,'Ticker'), file_name)
                data.to_csv(output_path, index=False, encoding="utf-8-sig", date_format='%Y/%m/%d')
    if True:
        directory_path = os.path.join(csv_file_path,'Ticker')
        all_files = os.listdir(directory_path)
        csv_files = [file for file in all_files if file.endswith('.csv')]
        freq_list = ['D', 'W', 'M', 'Q']

        for file_name in csv_files:
            data = pd.read_csv(os.path.join(directory_path, file_name), encoding='utf-8-sig')
            for freq in freq_list:
                ChangeFrequency_instance = ChangeFrequency(data, change_freq=freq)
                change_freq_data = ChangeFrequency_instance.convert_frequency()
                output_file_name = f'{file_name[:-4]}_{freq}.csv'
                output_path = os.path.join(os.path.join(csv_file_path,'change_freq'), output_file_name)
                change_freq_data.to_csv(output_path, index=True, encoding='utf-8-sig', date_format='%Y/%m/%d')
    if True:
        directory_path = os.path.join(csv_file_path,'change_freq')
        all_files = os.listdir(directory_path)
        csv_files = [file for file in all_files if file.endswith('.csv')]
        signal_function = [SignalPoints1,SignalPoints2,SignalPoints3,SignalPoints4]
        for file_name in csv_files:
            file_path = os.path.join(directory_path, file_name)
            data = pd.read_csv(file_path, encoding='utf-8-sig',date_format='%Y/%m/%d')

            for i, signal_func in enumerate(signal_function, start=1):
                # 產生訊號
                signal = signal_func(data)
                
                # 使用訊號點判別式並產出數據結果
                data_with_signals = getattr(signal, f'signal_points_{i}')()
                
                # 將數據結果儲存成CSV文件
                output_path = os.path.join(os.path.join(csv_file_path,'signal'),'signal_points')
                data_with_signals.to_csv(os.path.join(output_path,f'signal_points_{i}_{file_name}'), encoding='utf-8-sig')

    if True:
        # 設定路徑
        directory2_path = os.path.join(os.path.join(csv_file_path,'signal'),'signal_points')
        input_Ticker_path = csv_file_path
        output_directory =  os.path.join(csv_file_path,'Analysis')
        combined_data_output_path = os.path.join(directory2_path, '合併後檔案')
        confusion_matrix_output_dir = os.path.join(output_directory, 'Confusion_Matrix')
        merged_file_output_path = os.path.join(output_directory, 'merged_file.csv')

        # 列出路徑中的所有檔案
        all_files2 = os.listdir(directory2_path)

        # 篩選 CSV 檔案
        csv_files2 = [file for file in all_files2 if file.endswith('.csv')]

        # 合併檔案
        for file_name in csv_files2:
            input_filename_1 = os.path.join(input_Ticker_path, '波動標註檔.csv')
            input_filename_2 = os.path.join(directory2_path, file_name)
            merger = merged_csv(input_filename1=input_filename_1, input_filename2=input_filename_2)
            merged_data = merger.merged_csv()
            
            output_file_name = f'{file_name[:-4]}.csv'
            output_path = os.path.join(combined_data_output_path, output_file_name)
            merged_data.to_csv(output_path, encoding='utf-8-sig')

        # 執行混淆矩陣程式碼
        df = pd.read_csv(os.path.join(input_Ticker_path, 'Fred_Ticker.csv'))
        chinese_columns = df[['Ticker', 'chinese_columns']]
        deleter = DeleteData(confusion_matrix_output_dir)
        deleter.delete_csv_files()

        all_files = os.listdir(combined_data_output_path)
        csv_files = [file for file in all_files if file.endswith('.csv')]
        longshort_list = ['long', 'short']

        for file_name in csv_files:
            file_path = os.path.join(combined_data_output_path, file_name)
            data = pd.read_csv(file_path, index_col='Date', encoding='utf-8-sig')
            
            for longshort in longshort_list:
                cm = Confusion_Matrix()
                signal_points, Ticker, change_freq = cm.variable(file_name)
                cm.chinese_columns = chinese_columns
                cm.signal_points = signal_points
                cm.Ticker = Ticker
                cm.change_freq = change_freq
                Datestart, Date_End = cm.time(data)
                cm.Datestart = Datestart
                cm.Date_End = Date_End
                cm.ID = cm.generate_unique_id()
                cm.longshort = longshort
                cm.data = data
                
                result = cm.Confusion_Matrix(data)
                output_file_name = f'{cm.ID}{Ticker}.csv'
                output_path = os.path.join(confusion_matrix_output_dir, output_file_name)
                result.to_csv(output_path, encoding='utf-8-sig')

        # 合併大表
        dfs = []
        for file_name in os.listdir(confusion_matrix_output_dir):
            file_path = os.path.join(confusion_matrix_output_dir, file_name)
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_path)
                dfs.append(df.iloc[:, 1])

        merged_df = pd.concat(dfs, ignore_index=True, axis=1)
        merged_df.index = df.iloc[:, 0]
        merged_df = merged_df.T
        merged_df.to_csv(merged_file_output_path, encoding='utf-8-sig', index=False)\

if __name__ == '__main__':
    #執行程式
    execute_code()