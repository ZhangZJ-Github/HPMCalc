import par_parser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    filepath="E:\\11-11\\5\\Genac20G100keV_1_20231105_221817_23_Acc01-5-1-HiQ-witness.par"
    par = par_parser.PAR(filepath)
    particles_energy_title=' ALL PARTICLES @AXES(X1,KE)-#2 $$$PLANE_X1_AND_KE_AT_X0=  0.000'
    n=50   #按z坐标均匀划分n段
    df=pd.DataFrame()
    zmean=list()
    emean=list()  #每段求平均值
    for i in range(10):
        df=df._append(par.phasespaces[particles_energy_title][-i-1]['data'])
    df = df.sort_values(by=0) #排序
    df=df.dropna(axis=0, how='any') #去掉nan
    df.columns = ['z', 'e']
    zmin=min(df['z'])
    zmax=max(df['z'])
    zlength=zmax-zmin
    for i in range(n):
        left=zmin+i*zlength/n
        right=zmin+(i+1)*zlength/n
        zmean.append(np.mean((df.query('z>=@left&z<=@right'))['z']))
        emean.append(np.mean((df.query('z>=@left&z<=@right'))['e']))

    # for i in range(n):
    #     zmean.append(np.mean(df[i*len(df)//n:(i+1)*len(df)//n][0]))
    #     emean.append(np.mean(df[i * len(df) // n:(i + 1) * len(df) // n][1]))
    #     emax.append(max(df[i * len(df) // n:(i + 1) * len(df) // n][1]))
    #plt.scatter(zmean, emean)
    #plt.scatter(zmean, emax)
    #plt.show()
    acc_gradient=(emean[n-1]-emean[int(n*2/5)])/(zmean[n-1]-zmean[int(n*2/5)])




