import pandas as pd


def get_acoustic_spectrum(excelFile_fft):
    '''
    Function to read FFT file recorded with 831C sound level meter from Larson & Davis
    '''

    column_values_freqs = pd.read_excel(
        excelFile_fft,
        sheet_name="FFT Overall",       # Replace with your sheet name
        usecols=[0] # Or usecols="B" or usecols=[0]
    ).squeeze().dropna().tolist()

    column_values_LAavg = pd.read_excel(
        excelFile_fft,
        sheet_name="FFT Overall",       # Replace with your sheet name
        usecols=[1] # Or usecols="B" or usecols=[0]
    ).squeeze().dropna().tolist()

    # Find where list starts
    try:
        start_index = column_values_freqs.index('Frequency (Hz)')
    except ValueError:
        print("Start of measurements not found. No list element = Frequency (Hz)")

    freqs = column_values_freqs[start_index+1:]
    LAavg = column_values_LAavg[start_index-1:] ## off by two because of two dark blue headers in first column

    return freqs, LAavg