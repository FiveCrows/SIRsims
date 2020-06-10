class Workplace():
	def __init__(self, attrs:list):
		'''
            Initialize Person objects with attributes from array
        '''
		self.sp_id=int(attrs[0].decode('UTF-8'))
		self.latitude=float(attrs[1].decode('UTF-8'))
		self.longitude=float(attrs[2].decode('UTF-8'))
