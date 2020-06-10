class Household:
	def __init__(self, attrs: list):
		'''
            Initialize Household objects with attributes from array
        '''
		self.sp_id=int(attrs[0].decode('UTF-8'))
		self.stcotrbg=int(attrs[1].decode('UTF-8'))
		self.hh_race=int(attrs[2].decode('UTF-8'))
		self.hh_income=int(attrs[3].decode('UTF-8'))
		self.latitude=float(attrs[4].decode('UTF-8'))
		self.longitude=float(attrs[5].decode('UTF-8'))