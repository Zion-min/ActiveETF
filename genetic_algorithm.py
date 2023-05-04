import numpy as np

class GeneticAlgorithm:
    def __init__(self, popsize, mutation_rate, elite, rM, rA, pricedata, t, target_corr = 0.7, verbose=True):
        #hyperparameters
        self.n_stock = len(pricedata.columns)
        self.popsize = popsize
        self.mrate = mutation_rate
        self.elite = elite
        self.pick = 20
 
        self.verbose= verbose
        
        #vars for calc D
        self.rM = rM
        self.rA = rA
        self.std_a = rA.std()
        self.corr_am = np.corrcoef(rA, rM)[0][1]
        self.pricedata = pricedata
        self.tickers = pricedata.columns
        self.t = t
        self.target_corr = target_corr
        
        #vars
        self.generation = 0
        self.population = None
        self.rB = None
        self.fitness_fnc = None
        self.avg_fit = []
        self.selected = None
        self.next_gen = []

        self.fitness_by_gen = []
        
    def init_population(self):
        self.generation = 1
        self.population = [np.zeros(self.n_stock) for i in range(self.popsize)]
        self.population = list(map(self.assign_1, self.population))
        
    def assign_1(self, li):
        idx = np.random.choice(range(len(li)), self.pick, replace=False)
        li[idx] = 1
        return li
    
    #동일비중임. ga2에서는 mean하면안되고 w랑 내적해야함.
    def calc_returns(self, b):
        return np.array(self.pricedata[self.tickers[np.where(b)[0]]]).mean(axis=1)
    
    #전체 풀에 대해서 각 염색체(포트폴리오)마다 r_B값을 계산
    def calc_rB(self):
        self.rB = list(map(self.calc_returns, self.population))
        
    #1개의 염색체에 대해서 fitness 값을 리턴
    def fitness_function(self, rb):
        std_b = rb.std()
        corr_ABM = np.corrcoef([rb, self.rA, self.rM])
        corr_ab = corr_ABM[0][1]
        corr_bm = corr_ABM[0][2]
        return (self.t*self.std_a)**2 * (self.corr_am**2 - self.target_corr**2) +  2 * self.t * (1-self.t)*self.std_a * std_b * (self.corr_am * corr_bm - self.target_corr * corr_ab) + (1- self.t)**2 * std_b**2 * (corr_bm**2 - self.target_corr **2)
    
    def calc_fitnessfnc(self):
        self.fitness_fnc = list(map(self.fitness_function, self.rB))
        self.fitness_by_gen.append(self.fitness_fnc) #boxplot할때 필요
        self.avg_fit.append(np.mean(self.fitness_fnc)) #plot할때 필요

    def selection(self, nothing):
        d = np.array(self.fitness_fnc, dtype='float64')
        #min_max = (d-d.min())/(d.max()-d.min())
        min_max = d - d.min()
        weight = min_max.cumsum()/min_max.sum()
        weight = weight / weight[-1] #0~1나눠진거에서 왜 한번더 1로 나누는 걸까? -> 가끔 0.99999998 나오는 경우가 있어서 
        r = np.random.rand(2) 
        a_idx = np.where(weight > r[0])[0][0]
        b_idx = np.where(weight > r[1])[0][0] #probablitic selection
        return a_idx, b_idx
    
    def select_once(self):
        self.selected = list(map(self.selection, range(int(self.popsize/2)))) #한번에 두개를 뽑으니 popsize/2 -> self.selected에 index저장

    def find_index(self, elite):
        return np.where(self.fitness_fnc==elite)[0]
        
    def elite_append(self):
        elite_list = sorted(self.fitness_fnc,reverse=True)[:self.elite] #ftness 를 20개로 sort

        idx = list(map(tuple,np.array(list(map(self.find_index, elite_list))).reshape(2,-1))) #find_index는 해당하는 elite와 ftnss가 같은 index를 뽑아주고, selected에 넣기 좋게 바꿈
        self.selected[-int(self.elite/2):] = idx #글고 맨뒤에 추가해줌
    
    def crossover_and_mutate(self, parents):
        dad_idx   = np.where(self.population[parents[0]])[0]
        mom_idx = np.where(self.population[parents[1]])[0]

        u, c = np.unique(np.concatenate([dad_idx, mom_idx]), return_counts=True) #return_counts => 개수 새는 거. u=중복 없는 값들, c=그 값의 개수
        dup = u[c>1] #값 개수가 여러개 있는 수를 dup에 따로 저장
        idxs = np.setdiff1d(u, dup) #u-dup 차집합
  
        c1_idx, c2_idx = np.random.choice(idxs, (2, int(len(idxs)/2)), replace=False) #중복이 안되는 걸 반으로 나눠서 각각 할당해주고, 그담에 중복되는 걸 넣어준다
        c1_idx = np.append(c1_idx, dup) 
        c2_idx = np.append(c2_idx, dup)
        
        c1 = np.zeros(self.n_stock)
        c2 = np.zeros(self.n_stock)

        c1[c1_idx] = 1
        c2[c2_idx] = 1

        if np.random.rand() < self.mrate:
            one_to_zero = np.random.choice(c1_idx)
            zero_to_one = np.random.choice(np.setdiff1d(range(self.n_stock), c1_idx))
            c1[one_to_zero] = 0
            c1[zero_to_one] = 1

            one_to_zero = np.random.choice(c2_idx)
            zero_to_one = np.random.choice(np.setdiff1d(range(self.n_stock), c2_idx))
            c2[one_to_zero] = 0
            c2[zero_to_one] = 1
        return c1, c2

    def crossover(self):
        self.next_gen = list(sum(list(map(self.crossover_and_mutate, self.selected)), ()))
    
    def set_next_gen(self):
        self.generation += 1
        self.population = self.next_gen
        self.rB = None
        self.selected = None
        self.next_gen = None

    def step(self):
        self.fitness_fnc = None

        self.calc_rB()
        self.calc_fitnessfnc()
        self.select_once()
        self.crossover()
        self.elite_append()
        self.set_next_gen()
        
    def run(self):
        while True:
            self.step()
            if self.verbose:
                print(f"Gen{self.generation} : {round(self.avg_fit[-1], 10)}")
                if self.generation % 10 == 0:
                    print('------------------')
            if self.generation >= 60:
                break
