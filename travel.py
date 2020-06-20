class Travel:
	def __init__(self,attrs: list):
		'''
            Initialize travel objects with attributes from array
        '''
		self.home=attrs[0]
		self.work=attrs[1] #None if student or unemployed
		self.school=attrs[2] #None if not student