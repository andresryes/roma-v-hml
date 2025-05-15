import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

file_root = pathlib.Path(__file__).resolve().parent.parent

# Load data
detailed_path = file_root / 'output' / 'hpatches_detailed_results.csv'
summary_path = file_root / 'output' / 'hpatches_summary_results.csv'
detailed_df = pd.read_csv(detailed_path)

detailed_df['average_reprojection_error'].replace([np.inf, -np.inf], np.nan, inplace=True)

threshold = 100
filtered_df = detailed_df[~((detailed_df['matcher'] == 'HomoMatcherLite') &
                            (detailed_df['average_reprojection_error'] > threshold))]

q1_err = filtered_df['average_reprojection_error'].quantile(0.25)
q3_err = filtered_df['average_reprojection_error'].quantile(0.75)
iqr_err = q3_err - q1_err
lower_err = q1_err - 1.5 * iqr_err
upper_err = q3_err + 1.5 * iqr_err
non_outlier_df = filtered_df[(filtered_df['average_reprojection_error'] >= lower_err) &
                             (filtered_df['average_reprojection_error'] <= upper_err)]
non_outlier_sorted = non_outlier_df.sort_values(by=['matcher', 'sequence'])

q1_time = filtered_df['total_time_s'].quantile(0.25)
q3_time = filtered_df['total_time_s'].quantile(0.75)
iqr_time = q3_time - q1_time
lower_time = q1_time - 1.5 * iqr_time
upper_time = q3_time + 1.5 * iqr_time
time_filtered_df = filtered_df[(filtered_df['total_time_s'] >= lower_time) &
                               (filtered_df['total_time_s'] <= upper_time)]
time_filtered_sorted = time_filtered_df.sort_values(by=['matcher', 'sequence'])

custom_palette = sns.color_palette("Set2", n_colors=2)

plt.figure(figsize=(14, 6))
sns.lineplot(data=non_outlier_sorted, x='sequence', y='average_reprojection_error',
             hue='matcher', marker='o', palette=custom_palette)
plt.title("Average Reprojection Error per Sequence (Outliers Removed by IQR)")
plt.xticks(rotation=90)
plt.ylabel("Reprojection Error")
plt.xlabel("Sequence")
plt.tight_layout()
plt.savefig(file_root / 'output' / 'hpatches_detailed_reprojection_error.png')
plt.show()

plt.figure(figsize=(14, 6))
sns.lineplot(data=time_filtered_sorted, x='sequence', y='total_time_s',
             hue='matcher', marker='o', palette=custom_palette)
plt.title("Total Time per Sequence (Outliers Removed by IQR)")
plt.xticks(rotation=90)
plt.ylabel("Total Time (s)")
plt.xlabel("Sequence")
plt.tight_layout()
plt.savefig(file_root / 'output' / 'hpatches_detailed_total_time.png')
plt.show()
