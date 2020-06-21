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
#workplace_data=pickle.load(open(workplaces_file,'rb'))
#school_data=pickle.load(open(school_file, 'rb'))
student_M, student_F, employee_M, employee_F=0,0,0,0
ages_employees, ages_students={},{}
ages_employees=defaultdict(lambda: 0,ages_employees)
ages_students=defaultdict(lambda: 0,ages_students)
for row in people_data:
    person=people_data[row]
    if person.work_id:
        if person.sex==0:
            employee_M+=1
        else:
            employee_F+=1
        ages_employees[person.age]+=1
    if person.school_id:
        if person.sex==0:
            student_M+=1
        else:
            student_F+=1
        ages_students[person.age]+=1
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
plt.show()
