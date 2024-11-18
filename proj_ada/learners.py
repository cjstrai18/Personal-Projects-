class strong: 
    def __init__(self): 
        self.ensemble = [] 
        
    def pred(self, dp): 
        output = 0
        for i in self.ensemble: 
            output += i.partition(dp)*i.weight
        if output > 0: 
            return 1.0
        if output < 0: 
            return -1.0 
        
class weak: 
    def __init__(self, mark, direct): 
        self.marker = mark
        self.weight = 0  
        self.dir = direct
        
    def partition(self, dp): 
        x = self.marker[0]
        y = self.marker[1]
        val = dp[x][y]
        if self.dir == 'pos': 
            return 1.0 if val == 1.0 else -1.0
        if self.dir == 'neg': 
            return -1.0 if val == 1.0 else 1.0
