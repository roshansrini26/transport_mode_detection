import pandas as pd
import numpy as np
import os
from scipy import stats

#Module 1: Loading data
def load_data(data_dir):
    all_data = []  #Load data files
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            df['ride_id'] = int(file.split('_')[1].split('.')[0])
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Module 2: Data Cleaning
def clean_data(df):
    df = df.dropna() #clean data by removing outliers
    z_scores = np.abs(stats.zscore(df[['acceleration_x', 'acceleration_y', 'acceleration_z', 
                                       'gyro_x', 'gyro_y', 'gyro_z']]))
    df = df[(z_scores < 3).all(axis=1)]
    return df

# Module 3: Sensor Fusion (Simplified Complementary Filter)
def compute_orientation(df):
    # Initialize roll, pitch
    df['roll'] = 0.0
    df['pitch'] = 0.0

    # Compute initial roll and pitch from accelerometer (gravity)
    df['roll'] = np.arctan2(df['acceleration_y'], df['acceleration_z'])
    df['pitch'] = np.arctan2(-df['acceleration_x'], np.sqrt(df['acceleration_y']**2 + df['acceleration_z']**2))

    # Use gyroscope to update orientation (complementary filter)
    alpha = 0.98  # Weight for gyroscope (0-1)
    dt = 0.1      # Assume 10 Hz sampling rate (adjust based on your data)

    for i in range(1, len(df)):
        # Integrate gyroscope data
        roll_gyro = df['roll'].iloc[i-1] + df['gyro_x'].iloc[i] * dt
        pitch_gyro = df['pitch'].iloc[i-1] + df['gyro_y'].iloc[i] * dt

        # Accelerometer-based roll and pitch
        roll_acc = np.arctan2(df['acceleration_y'].iloc[i], df['acceleration_z'].iloc[i])
        pitch_acc = np.arctan2(-df['acceleration_x'].iloc[i], 
                               np.sqrt(df['acceleration_y'].iloc[i]**2 + df['acceleration_z'].iloc[i]**2))

        # Combine using complementary filter
        df.loc[i, 'roll'] = alpha * roll_gyro + (1 - alpha) * roll_acc
        df.loc[i, 'pitch'] = alpha * pitch_gyro + (1 - alpha) * pitch_acc

    df['game_rotation_vector'] = np.sqrt(df['roll']**2 + df['pitch']**2)
    df['rotation_vector'] = df['game_rotation_vector']  # Simplified, no magnetometer
    df['orientation'] = df['roll']  # Simplified, using roll as a proxy

    return df

# Module 4: Feature Engineering
def calculate_statistics(df):
    """Calculate statistics for each sensor type per ride."""
    df['accel_magnitude'] = np.sqrt(df['acceleration_x']**2 + df['acceleration_y']**2 + df['acceleration_z']**2)
    df['gyro_magnitude'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)

    # Compute orientation-related fields
    df = compute_orientation(df)

    # Group by ride_id and compute stats
    ride_stats = df.groupby('ride_id').agg({
        'timestamp': 'count',
        'accel_magnitude': ['mean', 'min', 'max', 'std'],
        'gyro_magnitude': ['mean', 'min', 'max', 'std'],
        'game_rotation_vector': ['mean', 'min', 'max', 'std'],
        'rotation_vector': ['mean', 'min', 'max', 'std'],
        'orientation': ['mean', 'min', 'max', 'std']
    }).reset_index()

    # Flatten columns
    ride_stats.columns = ['ride_id', 'time', 
                         'android.sensor.accelerometer#mean', 'android.sensor.accelerometer#min', 
                         'android.sensor.accelerometer#max', 'android.sensor.accelerometer#std',
                         'android.sensor.gyroscope#mean', 'android.sensor.gyroscope#min', 
                         'android.sensor.gyroscope#max', 'android.sensor.gyroscope#std',
                         'android.sensor.game_rotation_vector#mean', 'android.sensor.game_rotation_vector#min',
                         'android.sensor.game_rotation_vector#max', 'android.sensor.game_rotation_vector#std',
                         'android.sensor.rotation_vector#mean', 'android.sensor.rotation_vector#min',
                         'android.sensor.rotation_vector#max', 'android.sensor.rotation_vector#std',
                         'android.sensor.orientation#mean', 'android.sensor.orientation#min',
                         'android.sensor.orientation#max', 'android.sensor.orientation#std']

    # Add placeholders for sound
    ride_stats['sound#mean'] = 0.0
    ride_stats['sound#min'] = 0.0
    ride_stats['sound#max'] = 0.0
    ride_stats['sound#std'] = 0.0

    # Assign target
    ride_stats['target'] = ride_stats['ride_id'].apply(lambda x: 'Car' if x % 2 == 1 else 'Still')

    return ride_stats

# Module 5: Main Execution
def main():
    """Main function to process all CSV files into standardized format."""
    data_dir = "ride_data/"
    output_file = "standardized_ride_data.csv"

    # Load and clean data
    raw_data = load_data(data_dir)
    cleaned_data = clean_data(raw_data)

    # Calculate statistics
    standardized_data = calculate_statistics(cleaned_data)

    # Reorder columns to match desired format
    desired_columns = ['time', 
                      'android.sensor.accelerometer#mean', 'android.sensor.accelerometer#min', 
                      'android.sensor.accelerometer#max', 'android.sensor.accelerometer#std',
                      'android.sensor.game_rotation_vector#mean', 'android.sensor.game_rotation_vector#min',
                      'android.sensor.game_rotation_vector#max', 'android.sensor.game_rotation_vector#std',
                      'android.sensor.gyroscope#mean', 'android.sensor.gyroscope#min', 
                      'android.sensor.gyroscope#max', 'android.sensor.gyroscope#std',
                      'android.sensor.orientation#mean', 'android.sensor.orientation#min', 
                      'android.sensor.orientation#max', 'android.sensor.orientation#std',
                      'android.sensor.rotation_vector#mean', 'android.sensor.rotation_vector#min', 
                      'android.sensor.rotation_vector#max', 'android.sensor.rotation_vector#std',
                      'sound#mean', 'sound#min', 'sound#max', 'sound#std', 'target']
    
    standardized_data = standardized_data[desired_columns]

    # Save to CSV
    standardized_data.to_csv(output_file, index=False)
    print(f"Standardized data saved to {output_file}")

if __name__ == "__main__":
    main()