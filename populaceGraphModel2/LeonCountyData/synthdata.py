convertSex=lambda x: 1 if x=='F' else 0 #Sex: M=0, F=1
convertCol=lambda y: None if y.decode('UTF-8')=="X" else int(y.decode('UTF-8'))  #Other columns: X=None

class Person:
    def __init__(self,attrs: list):
        '''
            Initialize Person objects with attributes from array
        '''
        self.sp_id=int(attrs[0].decode('UTF-8'))
        self.sp_hh_id=int(attrs[1].decode('UTF-8'))
        self.age=int(attrs[2].decode('UTF-8'))
        self.sex=convertSex(attrs[3].decode('UTF-8'))
        self.race=int(attrs[4].decode('UTF-8'))
        self.relate=int(attrs[5].decode('UTF-8'))
        self.school_id=convertCol(attrs[6])
        self.work_id=convertCol(attrs[7])
        self.comorbidities={'Hypertension': False,'Obesity': False, 'Lung disease':False,'Diabetes': False,'Heart disease':False,
        'MaskUsage': False, 'Other': False}

class Workplace():
    def __init__(self, attrs:list):
        '''
            Initialize Workplace objects with attributes from array
        '''
        self.sp_id=int(attrs[0].decode('UTF-8'))
        self.latitude=float(attrs[1].decode('UTF-8'))
        self.longitude=float(attrs[2].decode('UTF-8'))
        self.others={'zipcode': None, 'members_count': 0}

class Household:
    def __init__(self, attrs: list):
        '''
            Initialize Household objects with attributes from array
        '''
        print("household, attrs= ", attrs)
        self.sp_id=int(attrs[0].decode('UTF-8'))
        self.stcotrbg=int(attrs[1].decode('UTF-8'))
        self.hh_race=int(attrs[2].decode('UTF-8'))
        self.hh_income=int(attrs[3].decode('UTF-8'))
        self.latitude=float(attrs[4].decode('UTF-8'))
        self.longitude=float(attrs[5].decode('UTF-8'))
        self.others={'zipcode': None, 'members_count': 0}

class School():
    def __init__(self, attrs:list):
        '''
            Initialize School objects with attributes from array
        '''
        self.sp_id=int(attrs[0].decode('UTF-8'))
        self.stco=int(attrs[1].decode('UTF-8'))
        self.latitude=float(attrs[2].decode('UTF-8'))
        self.longitude=float(attrs[3].decode('UTF-8'))
        self.others={'zipcode': None, 'members_count': 0}
