import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import os
import glob
from scipy import signal

def data_start(dir: str) -> object:
    main_df = pd.DataFrame()
    
    filepaths = glob.glob(os.path.join(dir, "*.csv"))
    
    for files in filepaths:
        df = pd.read_csv(files, sep=';',decimal=',', index_col=False, skiprows=8, usecols=range(18), encoding='iso-8859-1')
        df['datetime'] = pd.to_datetime(df.iloc[:, 0] + ' ' + df.iloc[:, 1])
        # Drop the original date and time columns by index
        df.drop(df.columns[[0, 1]], axis=1, inplace=True)
        df.replace(-9999, np.nan, inplace=True)
        df = df.dropna()
        
        
        df['year_month'] = df['datetime'].dt.to_period('M')
        df.drop(df.columns[[-2]], axis=1, inplace=True)
        
        # Group by year and month and calculate the average of 'x' and 'y'

        monthly_avg = df.groupby('year_month').mean()
        monthly_avg.reset_index(inplace=True)
        monthly_avg['year_month'] = monthly_avg['year_month'].astype(str)
        if len(main_df) == 0:
            main_df = monthly_avg
        else:
            main_df = pd.concat([main_df, monthly_avg])
    main_df = main_df.sort_values('year_month', inplace=True).reset_index()
    
    return main_df


def plot(df: object) -> None:
    plt.figure(figsize=(10, 5))

    # Plot the 'x' column
    for i in df.columns:
        if i != 'year_month':
            plt.plot(df.index, df[i], linestyle='-', label=i)  

    # Adding titles and labels
    plt.title('Data x Months')
    plt.xlabel('DATA')
    plt.ylabel('YEAR_MOUNTH')

    # Adding a legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    data_dir = "../Data"
    df_main = data_start(data_dir)
    df_main['TEMPERATURA MÉDIA HORA (°C)'] = df_main[['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)', 'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)']].mean(axis=1)
    df_main['UMIDADE REL. MÉDIA HORA (%)'] = df_main[['UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)', 'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)']].mean(axis=1)
    df_tradted = df_main[['TEMPERATURA MÉDIA HORA (°C)', 'UMIDADE REL. MÉDIA HORA (%)', 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)', 'year_month']]
    
    lookback = 365*25
    df_tradted['TEMPERATURA MÉDIA HORA (°C)'][-lookback:]

    def apply_convolution(x, window):
        conv = np.repeat([0., 1., 0.], window)
        filtered = signal.convolve(x, conv, mode='same') / window
        return filtered

    denoised = df_tradted.apply(lambda x: apply_convolution(x, 90))
    plot(df_tradted)