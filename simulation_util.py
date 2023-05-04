import time
import numpy as np
import pandas as pd
import copy
import pickle
import matplotlib.pyplot as plt
from scipy import optimize
import time
plt.style.use('seaborn')

from genetic_algorithm import *
from autoencoder import *



class Simulation:
    def __init__(self, tamt, closeprice, cap, indices, lookback, AElookback, AElr, AEepoch):
        self.tamt = tamt
        self.closeprice = closeprice
        self.cap = self.cap
        self.indices = indices
        self.lookback = lookback
        self.AElookback = AElookback
        self.AElr = AElr
        self.AEepoch = AEepoch
        _, self.names = self.closeprice_.iloc[7:9].values
    
    def make_A(selfl, tamt, cap, qtr, n_choice=20):
        tamt = copy.deepcopy(tamt) # 이거 안하면 글로벌 필드에 영향 줌
        tamt = tamt.dropna(axis=1)
        tamt = tamt.mean()#당일 평균거래량
        threshold = tamt.quantile(qtr)
        amt_satis_tickers = tamt[tamt > threshold].index # 거래량 기준 만족하는 티커들
        
        print(f"거래대금으로 줄인 개수 {len(amt_satis_tickers)}")
        cap = copy.deepcopy(cap)[amt_satis_tickers].dropna(axis=1)
        cap = cap.mean()
        threshold = 200_000
        tickers = cap[cap > threshold].index
        print("시총으로 한번 더",len(tickers))
        chosen_tickers = np.random.choice(tickers, n_choice, replace=False)
        #추가적으로 비중고려해주기
        weight = np.random.rand(n_choice)
        weight = weight/sum(weight)
        
        return weight, chosen_tickers, tickers

    #최적화 함수
    def obj_func(self, w, price_finB, t, target_corr, rA, rM):
        rB = (price_finB * w).sum(axis=1)
        std_a = rA.std()
        std_b = rB.std()
        corrs = np.corrcoef([rB, rA, rM])
        corr_am = corrs[1][2]
        corr_bm = corrs[0][2]
        corr_ab = corrs[0][1]

        D = (t*std_a)**2 * (corr_am**2 - target_corr**2) +  2 * t * (1-t)*std_a * std_b * (corr_am * corr_bm - target_corr * corr_ab) + (1- t)**2 * std_b**2 * (corr_bm**2 - target_corr **2)
        return -D * 10000

    def const1(self, w, *args):
        return 1 - sum(w)

    def weight_allocate(self, B, universe, t, target_corr, rA, rM):
        tickers = universe.columns[np.where(B)[0]]
        price_finB = universe[tickers] #에 있는 종목을 가격데이터로 
        n = price_finB.shape[1] #20개의 종목
        #목적함수, 동일비중 (1,20)초기값, 제약조건[eqcons = 0이 되는 제약, 요소별 최소, 최대값 제약], fulloutput=False로 하면 그냥 그 최종 weight? 암튼 그거만 나옴.. , iprint는 중간중간에 결과 보고 해주는거임
        res = optimize.fmin_slsqp(self.obj_func, np.array([1/n for i in range(n)]), args=(price_finB, t, target_corr, rA, rM) , eqcons=[self.const1,], bounds=[(0.01,0.1)]*n, full_output=True, iprint=False)
        return res[0]#, res[1] / -10000

    def simulate(self, t, M, target_corr = 0.7, AE=True):
        st = time.time()
        t = t 
        M = M
        corrs_top_200_timeseries = []
        corrs_am = [] 
        idx_timeseries = []    
        rCs_timeseries = []
        rA_timeseries = []
        rM_timeseries = []
        a = 0
        N = 0

        for i in range(181, len(self.closeprice)-60, 60):
            print(self.closeprice.index[i].strftime(format='%Y-%m-%d'))

            idx_timeseries.append(self.closeprice.index[i].strftime(format='%Y-%m-%d'))
            daily_ret = self.closeprice.iloc[i-self.AElookback-1 : i].dropna(axis=1).pct_change(1).iloc[1:]
            universe = copy.deepcopy(daily_ret)
            tickers = universe.columns #해당 유니버스에 대한 티커
            backward_index = self.indices[M].iloc[i-self.lookback-1:i].pct_change(1).iloc[1:]

            
            #A생성, 전체 유니버스에서 A에 들어가는 종목 제외.
            A_w, A_tkr, tkr = self.make_A(self.tamt.iloc[i-self.AElookback : i], self.cap.iloc[i-self.AElookback : i], qtr=0.1) # 현재시점은 관측 불가능. 즉 i-1
            back_rA = (universe[A_tkr].iloc[-self.lookback:] * A_w).sum(axis=1)
            tickers_without_A = np.setdiff1d(tkr, A_tkr) # 티커에서 A에 포함된 티커를 빼줌.
            #shirinked_tickers =tickers_without_A
            
            #오토인코더 학습 및 축소된 유니버스 생성 -> 티커
            if AE:
                AE = AutoEncoder(self.AElr, self.AEepoch)
                shrinked_tickers = AE.shrink_universe(universe[tickers_without_A],
                                                verbose=False)
            else:
                #이 코드 될까? 유닛테스트 안해봄.
                shrinked_tickers = np.random.choice(tkr, int(len(tkr) /2), replace=False)

            #축소된 유니버스로 GA 학습 => B 도출
            GA = GeneticAlgorithm(popsize=1000, mutation_rate=.10, elite=50, 
                rM=backward_index, 
                rA=back_rA, 
                pricedata=universe[shrinked_tickers].iloc[-60:],
                t= 0.3, verbose=False, #여기서 등장하는 t
                target_corr = target_corr)
            GA.init_population()
            
            GA.run()
            plt.figure(figsize=(20,9))
            plt.boxplot(GA.fitness_by_gen)
            plt.plot(range(1,len(GA.avg_fit)+1), GA.avg_fit)
            plt.axhline(0.0, c='red', alpha=1, linewidth=1, linestyle='--')
            plt.xlabel('Gen')
            plt.ylabel("Fitnees")
            plt.show()
            
            top200 = list(map(lambda x : x[0], sorted(zip(GA.population, GA.fitness_fnc), key=lambda x : x[1], reverse=True)[:200]))
            top200_tickers = list(map(lambda x : shrinked_tickers[np.where(x)], top200))
            forward_window = self.closeprice.iloc[i-1:i+self.lookback].pct_change(1).iloc[1:]
            rA = (forward_window[A_tkr] * A_w).sum(axis=1)
            forward_index = self.indices[M].iloc[i-1:i+self.lookback].pct_change(1).iloc[1:]
            n = 100
            target_corr = GA.target_corr
            B_weights = list(map(self.weight_allocate, top200, [universe[shrinked_tickers].iloc[-self.lookback:]]*n, [t]*n, [target_corr]*n, [rA]*n, [forward_index]*n))

            rCs = list(map(        lambda x,w : t * rA + (1-t) * (forward_window[x] * w).sum(axis=1),        top200_tickers, B_weights    ))
            corrs = list(map(        lambda rC : np.corrcoef(rC, forward_index)[0][1],        rCs    ))
            
            corrs_am.append(np.corrcoef(rA, forward_index)[0][1])
            corrs_top_200_timeseries.append(corrs)
            C_tickers = np.union1d(A_tkr, top200_tickers)
            print("C포트폴리오[0] : ")
            print(f"A : {self.names[list(map(lambda x: np.where(x == tickers)[0][0], A_tkr))]}")
            print(f"B : {self.names[list(map(lambda x: np.where(x == tickers)[0][0], top200_tickers[0]))]}")
            print("")
            print(f"과거 A 코릴{round(np.corrcoef(back_rA, backward_index)[0][1], 4)}")
            print(f"미래 A 코릴 : {round(np.corrcoef(rA, forward_index)[0][1], 4)}")
            print(f"미래 B 코릴(평균값) : {round(np.mean(corrs),4)}")
            print(f"미래 C 코릴(평균값) : {round(np.mean(list(map(lambda x : np.corrcoef(x, forward_index)[0][1],rCs))),4)}")
            a_added = len(np.array(corrs)[np.array(corrs) > .7])
            a += a_added
            N += 100
            print(f"100개의 C 중 corr > .7인거 개수 : {a_added / 100}")
            
            if any(list(map(lambda x : forward_window[x].isna().any().any(), top200_tickers))):
                print("NA값 존재")
                break
            else:
                for i in (rCs[:-1]):
                    plt.plot((1+i).cumprod()*100, c='dodgerblue', alpha=.2)
                plt.plot((1+i).cumprod()*100, c='dodgerblue', alpha=.8, label='C')
                plt.plot((rA+1).cumprod()*100, c='red', linewidth=5, label='A', alpha=.7)
                plt.plot((forward_index+1).cumprod()*100, c='orange', linewidth=5, label='M')
                plt.ylabel("NAV")
                plt.legend()
                plt.show()
            rCs_timeseries.append(rCs)
            rA_timeseries.append(rA)
            rM_timeseries.append(forward_index)

            print("- - - - - - - - - - - - - - - \n\n\n")

        #시계열 박스플롯
        plt.boxplot(corrs_top_200_timeseries, labels=idx_timeseries)
        plt.axhline(0.7,c='r', linestyle='dashed')
        plt.show()

        print(f"걸린 시간 : {round((time.time() - st)/60, 2)}분")
        print("="*100)
        print("="*100)
        print("="*100)

        return {"timeseries" : idx_timeseries,
                "corrs_by_time" : corrs_top_200_timeseries,
                "corrs_am" : corrs_am,
                "phat" : a/N,
                "n" : N,
                "rCs" : rCs_timeseries,
                "rA" : rA_timeseries,
                "rM" : rM_timeseries}

    def save(data, t, M):
        with open(f'./result/분석결과_버퍼_{M}_{t}.pkl', 'wb') as f:
            pickle.dump(data, f)

    def save1(data, t, M):
        with open(f'./result/분석결과_디폴트_{M}_{t}.pkl', 'wb') as f:
            pickle.dump(data, f)

    def boxplot(x, dataset, line=None, horizontal_level = 0.7, box_interval = 2, box_length= 0.2, figsize=(10,7)):
        plt.figure(figsize=figsize)
        args1 = {"c":'black', "linewidth":.5}
        args2 = {"c":'orange', "linewidth":2}
        box_interval = box_interval
        box_length = box_length
        for i, data in enumerate(dataset):
            i = i * box_interval
            q1 = np.quantile(data, .25)
            q2 = np.quantile(data, .5)
            q3 = np.quantile(data, .75)
            iqr = q3 - q1
            head = 1.5 * iqr + q3
            tail = q1 - 1.5 * iqr
            maxhead = max(filter(lambda x : x <= head, data))
            mintail = min(filter(lambda x : x >= tail, data))
            undertail = list(filter(lambda x : x < tail, data))
            overhead = list(filter(lambda x : x > head, data))
            #Horizontal line
            plt.plot([i-box_length, i+box_length], [q1, q1], **args1)
            plt.plot([i-box_length, i+box_length], [q3, q3], **args1)
            plt.plot([i-box_length, i+box_length], [q2, q2], **args2)
            plt.plot([i-box_length, i+box_length], [maxhead, maxhead], **args1)
            plt.plot([i-box_length, i+box_length], [mintail, mintail], **args1)
            #Vertical line
            plt.plot([i-box_length, i-box_length], [q1, q3], **args1)
            plt.plot([i+box_length, i+box_length], [q1, q3], **args1)
            plt.plot([i, i], [q3, maxhead], **args1)
            plt.plot([i, i], [q1, mintail], **args1)
            plt.scatter([i]*len(overhead), overhead, c='black', s=10)
            plt.scatter([i]*len(undertail), undertail, c='black', s=10)
        
        if line:
            plt.plot([i*box_interval for i in range(len(line))], line, c='gray', label='A')
        if horizontal_level:
            plt.axhline(horizontal_level, c='red', linestyle='dashed')
        plt.legend()
        plt.xticks([i * box_interval for i in range(len(x))], x, rotation=90)
        plt.ylabel("Corr")
        plt.show()

    def report(self, M, t):
        with open(f'./result/분석결과_디폴트_{M}_{t}.pkl', 'rb') as f:
            res = pickle.load(f)
        idx = res['timeseries']
        corrs_by_time = res['corrs_by_time']
        corrs_am = res['corrs_am']
        phat = res['phat']
        n = res['n']
        rCs = res['rCs']
        rA = res['rA']
        rM = res['rM']
        
        self.boxplot(x=idx, dataset = corrs_by_time , line=corrs_am)

        print(f"phat = {phat}, n = {n}")

        #res = report('KRX 반도체', 0.5) #결과보는 코드

    def report(self, M, t, plotting=False):
        with open(f'./result/분석결과_버퍼_{M}_{t}.pkl', 'rb') as f:
            res = pickle.load(f)
        idx = res['timeseries']
        corrs_by_time = res['corrs_by_time']
        corrs_am = res['corrs_am']
        phat = res['phat']
        n = res['n']
        rCs = res['rCs']
        rA = res['rA']
        rM = res['rM']

        if plotting:
            for rcs, ra, rm, corrs in zip(rCs, rA, rM, corrs_by_time):
                for i in rcs[:-1]:
                    plt.plot((1+i).cumprod()*100, c='dodgerblue', alpha=.2)
                plt.plot((1+rcs[-1]).cumprod()*100, c='dodgerblue', alpha=.8, label='C')
                plt.plot((ra+1).cumprod()*100, c='red', linewidth=5, label='A', alpha=.7)
                plt.plot((rm+1).cumprod()*100, c='orange', linewidth=5, label='M')
                plt.ylabel("NAV")
                plt.legend()
                plt.show()
                print("평균 코릴 ", np.mean(corrs))

        
        self.boxplot(x=idx, dataset = corrs_by_time , line=corrs_am)

        print(f"phat = {phat}, n = {n}")