import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
home_dir="RestructuredData/"
people_file=home_dir+"people_list_serialized_rs.pkl"
workplaces_file=home_dir+"workplaces_list_serialized_rs.pkl"
school_file=home_dir+"schools_list_serialized_rs.pkl"
people_data=pickle.load(open(people_file, 'rb'))
households_file=home_dir+"households_list_serialized_rs.pkl"
households_data=pickle.load(open(households_file,'rb'))
income_groups=['<25k','25-50k', '50-75k', '75-100k','100-125k','125-150k','150k+']
student_M, student_F, employee_M, employee_F=0,0,0,0
ages_employees, ages_students={},{}
ages_employees=defaultdict(lambda: 0,ages_employees)
ages_students=defaultdict(lambda: 0,ages_students)
race_codes=['White','Black or African American','American Indian', 'Alaska Native', 'America Indian and/or Alaska native', 'Asian', 
            'Native Hawaiian or Other Pacific Islander','Other', 'Mixed']
employees_by_race={r:0 for r in race_codes}
students_by_race={r:0 for r in race_codes}
people_by_race={r:0 for r in race_codes}
for row in people_data:
    person=people_data[row]
    if person.work_id:
        if person.sex==0:
            employee_M+=1
        else:
            employee_F+=1
        ages_employees[person.age]+=1
        employees_by_race[race_codes[person.race-1]]+=1
    if person.school_id:
        if person.sex==0:
            student_M+=1
        else:
            student_F+=1
        ages_students[person.age]+=1
        students_by_race[race_codes[person.race-1]]+=1
    people_by_race[race_codes[person.race-1]]+=1
# print("Employee distribution by race:\n",employees_by_race)
# print("Student distribution by race:\n",students_by_race)
f, (ax1, ax2, ax3)=plt.subplots(3,1)
plt.subplots_adjust(hspace=0.6)
plt.figure(figsize=(12,12))
student_ages=sorted(list(ages_students.keys()))
employee_ages=sorted(list(ages_employees.keys()))
yticks=range(min(ages_students.values()), max(ages_students.values()), 200) 
ax1.plot(student_ages, [ages_students[key] for key in student_ages])
ax1.set_title("Student Age Distribution")
#ax1.set_yticks(yticks)
yticks=range(min(ages_employees.values()), max(ages_employees.values()), 200) 
ax2.plot(employee_ages, [ages_employees[key] for key in employee_ages], 'orange')
ax2.set_title("Employee Age Distribution")
#ax2.set_yticks(yticks)
x_vals=["Student\nFemale", "Student\nMale", "Employee\nMale", "Employee\nFemale"]
y_vals=[student_M, student_F, employee_M, employee_F]
#ax3.bar(x_vals, y_vals)
ax3.bar(x_vals, y_vals, color=['blue','orange','blue','orange'])
ax3.set_title("Gender Distribution")
#plt.show()

f,(ax1,ax2,ax3)=plt.subplots(3,1)
plt.subplots_adjust(hspace=0.6)
for key in race_codes:
    if students_by_race[key] !=0:
        ax1.bar(key+" Student",students_by_race[key],width=1)
    if employees_by_race[key] !=0:
        ax2.bar(key+"Employee",employees_by_race[key], width=1)
    if people_by_race[key] !=0:
        ax3.bar(key,people_by_race[key], label=key, width=1)
ax1.set_title("Student")
ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax2.set_title("Employees")
ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax3.set_title("Population")
ax3.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.suptitle("Population distribution by race showing students and employees")
f.legend()
plt.show()
households_by_income={i:0 for i in income_groups}
for row in households_data:
    if households_data[row].hh_income<=25000:
        households_by_income[income_groups[0]]+=1
    elif 25000<households_data[row].hh_income<=50000:
        households_by_income[income_groups[1]]+=1
    elif 50000<households_data[row].hh_income<=75000:
        households_by_income[income_groups[2]]+=1
    elif 75000<households_data[row].hh_income<=100000:
        households_by_income[income_groups[3]]+=1
    elif 100000<households_data[row].hh_income<125000:
        households_by_income[income_groups[4]]+=1
    elif 125000<households_data[row].hh_income<150000:
        households_by_income[income_groups[5]]+=1
    else:
        households_by_income[income_groups[6]]+=1
sns.set(style="whitegrid")
ax=sns.barplot(income_groups, [households_by_income[i] for i in income_groups])
plt.title("Household Income distribution")
plt.xticks(rotation=45)
plt.show()