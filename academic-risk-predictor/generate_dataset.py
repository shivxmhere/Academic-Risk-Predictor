import pandas as pd
import numpy as np
import os

def generate_data():
    np.random.seed(42)
    n_students = 850
    
    # Target distribution counts
    n_low = int(850 * 0.55)    # 467
    n_med = int(850 * 0.27)    # 229
    n_high = 850 - n_low - n_med # 154
    
    labels = ['Low'] * n_low + ['Medium'] * n_med + ['High'] * n_high
    np.random.shuffle(labels)
    
    # 1. Generate student IDs
    student_ids = [f"STU{str(i).zfill(3)}" for i in range(1, n_students + 1)]
    
    # 2. Features generation based on labels
    attendance = np.zeros(n_students)
    assignment_marks = np.zeros(n_students)
    internal_score = np.zeros(n_students)
    participation = np.zeros(n_students)
    prev_grade = np.zeros(n_students)
    study_hours = np.zeros(n_students)
    
    for i, label in enumerate(labels):
        if label == 'Low':
            # Low Risk: attendance_pct > 78 AND internal_score > 70 AND prev_grade > 68
            attendance[i] = np.random.uniform(79, 100)
            internal_score[i] = np.random.uniform(71, 100)
            prev_grade[i] = np.random.uniform(69, 100)
            assignment_marks[i] = np.random.uniform(60, 100)
            participation[i] = np.random.uniform(7.0, 10.0)
            study_hours[i] = np.random.uniform(3.0, 8.0)
        elif label == 'High':
            # High Risk: attendance_pct < 60 AND (internal_score < 50 OR assignment_marks < 45)
            attendance[i] = np.random.uniform(30, 59)
            if np.random.rand() > 0.5:
                internal_score[i] = np.random.uniform(20, 49)
                assignment_marks[i] = np.random.uniform(30, 70)
            else:
                internal_score[i] = np.random.uniform(30, 70)
                assignment_marks[i] = np.random.uniform(20, 44)
            prev_grade[i] = np.random.uniform(30, 60)
            participation[i] = np.random.uniform(1.0, 4.0)
            study_hours[i] = np.random.uniform(0.5, 3.0)
        else: # Medium
            # Everything else. Avoid the strict Low/High boundaries to keep it Medium.
            attendance[i] = np.random.uniform(60, 78)
            internal_score[i] = np.random.uniform(50, 70)
            prev_grade[i] = np.random.uniform(50, 68)
            assignment_marks[i] = np.random.uniform(45, 80)
            participation[i] = np.random.uniform(4.0, 7.0)
            study_hours[i] = np.random.uniform(2.0, 5.0)

    df = pd.DataFrame({
        'student_id': student_ids,
        'attendance_pct': np.round(attendance, 2),
        'assignment_marks': np.round(assignment_marks, 2),
        'internal_score': np.round(internal_score, 2),
        'participation': np.round(participation, 2),
        'prev_grade': np.round(prev_grade, 2),
        'study_hours': np.round(study_hours, 2),
        'risk_label': labels
    })
    
    # 3. Add slight random noise (5% chance to flip label one step)
    def add_noise(label):
        if np.random.rand() < 0.05:
            if label == 'High': return 'Medium'
            elif label == 'Low': return 'Medium'
            elif label == 'Medium': return np.random.choice(['Low', 'High'])
        return label
        
    df['risk_label'] = df['risk_label'].apply(add_noise)
    
    # 5. Add 12 rows with missing values (NaN)
    missing_indices = np.random.choice(n_students, 12, replace=False)
    cols_to_nan = ['attendance_pct', 'assignment_marks', 'internal_score', 'participation', 'prev_grade', 'study_hours']
    for idx in missing_indices:
        col = np.random.choice(cols_to_nan)
        df.loc[idx, col] = np.nan
        
    # 6. Print metrics
    print(f"Final shape: {df.shape}")
    print("\nClass distribution (%):")
    print((df['risk_label'].value_counts(normalize=True) * 100).round(2))
    print("\nNull counts:")
    print(df.isnull().sum())
    
    # Save
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_students.csv', index=False)
    print("\nSaved to data/raw_students.csv")

if __name__ == "__main__":
    generate_data()
