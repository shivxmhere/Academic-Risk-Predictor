import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure we have paths correct depending on where script runs
data_path = 'data/raw_students.csv'
if not os.path.exists(data_path):
    data_path = '../data/raw_students.csv'

output_dir = 'outputs'
if not os.path.exists(output_dir) and os.path.exists('../outputs'):
    output_dir = '../outputs'
os.makedirs(output_dir, exist_ok=True)

# 1. Load data
df = pd.read_csv(data_path)

# 2. Print Basic Info
print("--- BASIC INFO ---")
print(f"Shape: {df.shape}")
print("\n--- DATA TYPES ---")
print(df.dtypes)
print("\n--- NULL COUNTS ---")
print(df.isnull().sum())
print("\n--- CLASS DISTRIBUTION (counts) ---")
print(df['risk_label'].value_counts())
print("\n--- CLASS DISTRIBUTION (%) ---")
print((df['risk_label'].value_counts(normalize=True) * 100).round(2))
print("\n--- DESCRIPTIVE STATISTICS ---")
print(df.describe().round(2))

# 3. Generate Charts
color_palette = {"Low": "#22C55E", "Medium": "#EAB308", "High": "#EF4444"}

# Chart 1 - Class Balance Bar
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='risk_label', order=['Low', 'Medium', 'High'], palette=color_palette)
plt.title("Class Balance of Risk Labels")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_class_balance.png"), dpi=150)
plt.close()

# Chart 2 - Correlation Heatmap
plt.figure(figsize=(10, 6))
# Encode risk label for correlation
df_encoded = df.copy()
df_encoded['risk_label_encoded'] = df_encoded['risk_label'].map({"Low": 0, "Medium": 1, "High": 2})
corr = df_encoded.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_correlation_heatmap.png"), dpi=150)
plt.close()

# Helper for histograms
def plot_histogram(feature, filename, title):
    plt.figure(figsize=(10, 6))
    # We use multiple layers with alpha so overlapping is visible
    sns.histplot(data=df, x=feature, hue='risk_label', hue_order=['Low', 'Medium', 'High'], 
                 palette=color_palette, element="step", alpha=0.6, common_norm=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()

# Chart 3 - Attendance Distribution
plot_histogram('attendance_pct', "03_attendance_dist.png", "Attendance Distribution by Risk")

# Chart 4 - Internal Score Distribution
plot_histogram('internal_score', "04_internal_score_dist.png", "Internal Score Distribution by Risk")

# Chart 5 - Assignment Marks Distribution
plot_histogram('assignment_marks', "05_assignment_marks_dist.png", "Assignment Marks Distribution by Risk")

# Chart 6 - Study Hours Distribution
plot_histogram('study_hours', "06_study_hours_dist.png", "Study Hours Distribution by Risk")

# Chart 7 - Prev Grade Distribution
plot_histogram('prev_grade', "07_prev_grade_dist.png", "Previous Grade Distribution by Risk")

# Chart 8 - Attendance vs Internal Score Scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='attendance_pct', y='internal_score', hue='risk_label', 
                hue_order=['Low', 'Medium', 'High'], palette=color_palette, alpha=0.7)
plt.title("Attendance vs Internal Score by Risk")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "08_scatter_attendance_internal.png"), dpi=150)
plt.close()

# Chart 9 - Pairplot
numeric_cols = ['attendance_pct', 'assignment_marks', 'internal_score', 'participation', 'prev_grade', 'study_hours']
pairplot_data = df[numeric_cols + ['risk_label']]

# Pairplot has its own figure size mechanism
g = sns.pairplot(pairplot_data, hue='risk_label', hue_order=['Low', 'Medium', 'High'], palette=color_palette,
                 plot_kws={'alpha': 0.6})
g.fig.set_size_inches(14, 12)
plt.tight_layout()
g.savefig(os.path.join(output_dir, "09_pairplot.png"), dpi=150)
plt.close()

print("\nAll 9 EDA charts saved to outputs/")
