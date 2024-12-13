import pandas as pd
import json

# Load the JSON file
file_path = 'DataEngineeringQ2.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract relevant data
records = []
for record in data:
    patient_details = record.get('patientDetails', {})
    consultation_data = record.get('consultationData', {})
    medicines = consultation_data.get('medicines', [])
    
    records.append({
        'firstName': patient_details.get('firstName', ''),
        'lastName': patient_details.get('lastName', ''),
        'DOB': patient_details.get('birthDate', None),
        'gender': patient_details.get('gender', ''),
        'phoneNumber': record.get('phoneNumber', ''),
        'medicines': medicines
    })

# Create DataFrame
df = pd.DataFrame(records)

# Task 1: Missing Values
missing_values_percentage = df[['firstName', 'lastName', 'DOB']].apply(
    lambda col: ((col.isna().sum() + (col == '').sum()) / len(col) * 100).round(2)
)

# Task 2: Gender Imputation
df['gender'] = df['gender'].replace('', pd.NA)
mode_gender = df['gender'].mode()[0]
df['gender'] = df['gender'].fillna(mode_gender)
female_percentage = ((df['gender'] == 'F').sum() / len(df) * 100).round(2)

# Task 3: Age Groups
current_year = pd.Timestamp.now().year
df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
df['age'] = df['DOB'].apply(lambda dob: current_year - dob.year if pd.notna(dob) else None)
df['ageGroup'] = pd.cut(
    df['age'], bins=[-1, 12, 19, 59, float('inf')], labels=['Child', 'Teen', 'Adult', 'Senior']
)
adult_count = (df['ageGroup'] == 'Adult').sum()

# Task 4: Average Medicines
df['num_medicines'] = df['medicines'].apply(len)
avg_medicines = df['num_medicines'].mean().round(2)

# Task 5: 3rd Most Prescribed Medicine
all_medicines = [medicine['medicineName'] for sublist in df['medicines'] for medicine in sublist]
medicine_counts = pd.Series(all_medicines).value_counts()
third_most_prescribed = medicine_counts.index[2]

# Task 6: Active and Inactive Medicines
all_medicine_statuses = [medicine['isActive'] for sublist in df['medicines'] for medicine in sublist]
status_distribution = pd.Series(all_medicine_statuses).value_counts(normalize=True) * 100
active_percentage = status_distribution.get(True, 0.00).round(2)
inactive_percentage = status_distribution.get(False, 0.00).round(2)

# Task 7: Phone Number Validation
def is_valid_phone_number(phone):
    if isinstance(phone, str):
        if phone.startswith('+91'):
            phone = phone[3:]
        elif phone.startswith('91'):
            phone = phone[2:]
        if phone.isdigit() and len(phone) == 10 and 6000000000 <= int(phone) <= 9999999999:
            return True
    return False

df['isValidMobile'] = df['phoneNumber'].apply(is_valid_phone_number)
valid_phone_count = df['isValidMobile'].sum()

# Task 8: Pearson Correlation
correlation_data = df.dropna(subset=['age', 'num_medicines'])
pearson_correlation = correlation_data['age'].corr(correlation_data['num_medicines']).round(2)

# Print Results
print(f"Missing Values: {missing_values_percentage.tolist()}")
print(f"Female Percentage: {female_percentage}")
print(f"Adult Count: {adult_count}")
print(f"Average Medicines: {avg_medicines}")
print(f"3rd Most Prescribed Medicine: {third_most_prescribed}")
print(f"Active, Inactive Percentages: {active_percentage}, {inactive_percentage}")
print(f"Valid Phone Numbers: {valid_phone_count}")
print(f"Pearson Correlation: {pearson_correlation}")
