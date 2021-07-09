import time
import tkinter
from tkinter import filedialog, scrolledtext
import bl
import easygui
zong=1
root1 = tkinter.Tk()
root1.geometry('660x680')
root1.wm_title('Visual Interface')

def Button_command():
    Filepath = filedialog.askopenfilename()  # 获得选择好的文件

    t1 = tkinter.StringVar()
    t1.set(Filepath)
    entry = tkinter.Entry(root1, textvariable=t1).place(x=80, y=15)

    print(t1.get())  # 这里是提取的数据表格
    q = t1.get()
    bl.lujing=t1.get()
    if (q != 0):
        bl.datashuru = 1

varName1 = tkinter.StringVar()
varName1.set('')


varName2 = tkinter.StringVar()
varName2.set('')
label0 = tkinter.Label(root1, text='Enter the data table：')
label0.place(x=10, y=10)
labelName = tkinter.Label(root1, text=' Please enter y:',justify=tkinter.RIGHT, width=80)
labelName.place(x=15, y=100,width=85, height=20)
labelName = tkinter.Label(root1, text=' Please enter x:', justify=tkinter.RIGHT, width=80)
labelName.place(x=15, y=140, width=85, height=20)


# 控制右半区域
varNamey1 = tkinter.StringVar()
varNamey1.set('')
entryNamey1 = tkinter.Entry(root1, width=100,textvariable=varNamey1)#此处为规划后X值的输入窗口
entryNamey1.place(x=340, y=125, width=180)
def loginy():
    namey1 = entryName.get()
    bl.yy=namey1
    namey2='~'
    namey3=entryNamey1.get()
    bl.yx = namey3

    namey=namey1+namey2+namey3
    print(namey)
    bl.bianliangy=namey

    print(bl.bianliangy)
    if(bl.bianliangy!='~'):
        easygui.msgbox('Variable input successful', title='Helpful hints')
    if(bl.bianliangy == '~'):
        easygui.msgbox('Please enter data')






varName = tkinter.StringVar()
varName.set('')
btn = tkinter.Button(root1, text='...', width=2, height=1, command=Button_command).place(x=280, y=10)

t2 = tkinter.Entry(root1, width=20).place(x=140, y=15)

entryName = tkinter.Entry(root1, width=100,textvariable=varName1)#此处为Y值的输入窗口
entryName.place(x=120, y=105, width=120)

entryNamex = tkinter.Entry(root1, width=100,textvariable=varName2)#此处为x值的输入窗口
entryNamex.place(x=120, y=145, width=120, height=20)

def login():
    name1 = entryName.get()
    bl.y=name1
    name2='~'
    name3=entryNamex.get()
    bl.x = name3

    name=name1+name2+name3
    print(name)
    bl.bianliang=name
    print('变量的定义输出为：')
    print(bl.bianliang)
    if(bl.bianliang!='~'):
        easygui.msgbox('Variable input successful', title='Hint')
    if(bl.bianliang == '~'):
        easygui.msgbox('Please enter data')

def callback1():  # 定义一个 改变文本的函数 .
    if(bl.datashuru==1):
        easygui.msgbox('Data table uploaded successfully ', title='Hint')
    if(bl.datashuru == 0):
        easygui.msgbox('Please enter data table')

def zuotu():
    print(bl.bianliangy)
    model3 = sm.formula.ols(bl.bianliangy, data=train).fit()
    # 模型回归系数的估计值

    print(model3.params)

    # 异常值检验
    outliers = model3.get_influence()

    # 高杠杆值点（帽子矩阵）
    leverage = outliers.hat_matrix_diag
    # dffits值
    dffits = outliers.dffits[0]
    # 学生化残差
    resid_stu = outliers.resid_studentized_external
    # cook距离
    cook = outliers.cooks_distance[0]

    # 合并各种异常值检验的统计量值
    contat1 = pd.concat([pd.Series(leverage, name='leverage'), pd.Series(dffits, name='dffits'),
                         pd.Series(resid_stu, name='resid_stu'), pd.Series(cook, name='cook')], axis=1)
    # 重设train数据的行索引
    train.index = range(train.shape[0])
    # 将上面的统计量与train数据集合并
    profit_outliers = pd.concat([train, contat1], axis=1)
    profit_outliers.head()

    outliers_ratio = sum(np.where((np.abs(profit_outliers.resid_stu) > 2), 1, 0)) / profit_outliers.shape[0]
    print(outliers_ratio)

    # 挑选出非异常的观测点
    none_outliers = profit_outliers.loc[np.abs(profit_outliers.resid_stu) <= 2,]

    # 应用无异常值的数据集重新建模
    model4 = sm.formula.ols(bl.bianliangy, data=none_outliers).fit()  # 系数关系在此处
    print(model4.params)
    bl.xishu=model4.params
    print('模型概览')
    print(model4.summary())



    # 模型预测
    # model4对测试集的预测
    list1=['1']
    print(bl.yx)
    
    xh=0;
    for j in bl.yx:
        xh=1
        if(j!='+'):
            bl.sum=bl.sum+j;
            print(bl.sum)
        if (j == '+'):
            list1.append(bl.sum)
            bl.sum = ''
    if(xh==1):
        list1.append(bl.sum)

    del (list1[0])
    
   
    print(list1)
    pred4 = model4.predict(exog=test.loc[:, list1])
    plt.figure()
    print('对比预测值和实际值的差异：\n', pd.DataFrame({'Prediction': pred4, 'Real': test.pre}))
    bl.chayi=pd.DataFrame({'Prediction': pred4, 'Real': test.pre})
    # 绘制预测值与实际值的散点图
    print(pred4)
    plt.scatter(x=test.pre, y=pred4)
    # 添加斜率为1，截距项为0的参考线
    plt.plot([test.pre.min(), test.pre.max()], [test.pre.min(), test.pre.max()],
             color='red', linestyle='--')
    # 添加轴标签
    plt.xlabel('Actual value')
    plt.ylabel('Predictive value')
    # 显示图形
    plt.show()
    print(test)

scr = scrolledtext.ScrolledText(root1, width=25, height=15, font=("隶书", 15))  # 滚动文本框（宽，高（这里的高应该是以行数为单位），字体样式）
scr.place(x=20, y=270)  # 滚动文本框在页面的位置
def showdata():
    # scr = scrolledtext.ScrolledText(root1, width=25, height=15, font=("隶书", 15))  # 滚动文本框（宽，高（这里的高应该是以行数为单位），字体样式）
    # scr.place(x=20, y=270)  # 滚动文本框在页面的位置
    scr.insert(tkinter.END, "Comparison of actual value and predicted result：  ")
    scr.insert(tkinter.END, bl.blDataFrame)
    scr.insert(tkinter.END, "     Partial regression coefficient of the model：     ")
    scr.insert(tkinter.END, model.params)

scr1 = scrolledtext.ScrolledText(root1, width=25, height=15, font=("隶书", 15))  # 滚动文本框（宽，高（这里的高应该是以行数为单位），字体样式）
scr1.place(x=350, y=270)  # 滚动文本框在页面的位置
def showdatay():
    # scr = scrolledtext.ScrolledText(root1, width=25, height=15, font=("隶书", 15))  # 滚动文本框（宽，高（这里的高应该是以行数为单位），字体样式）
    # scr.place(x=360, y=270)  # 滚动文本框在页面的位置
    scr1.insert(tkinter.END, "Comparison of actual value and predicted result：  ")
    scr1.insert(tkinter.END, bl.chayi)
    scr1.insert(tkinter.END, "     Partial regression coefficient of the model：     ")
    scr1.insert(tkinter.END, bl.xishu)


   #测试代码段
b1 = tkinter.Button(root1, text='Result', width=10, command=showdata).place(x=140, y=220)

btny = tkinter.Button(root1, text='Ok', width=5,height=1, command=login).place(x=260, y=100)
btnx = tkinter.Button(root1, text='Ok', width=5,height=1, command=login).place(x=260, y=140)


b=tkinter.Button(root1, text='Plot', width=10, command=root1.quit).place(x=40, y=220)#点击开始作图按钮，执行quit命令

btn2 = tkinter.Button(root1, text='Submit', width=10, command=callback1).place(x=10, y=40)

buttonOk = tkinter.Button(root1, text='Submit',width=10,command=login)
buttonOk.place(x=10, y=170)

# 右半区域控制
by3=tkinter.Button(root1, text='Plot', width=10, command=zuotu).place(x=360, y=200)#点击开始作图按钮，执行quit
b1y3= tkinter.Button(root1, text='Result', width=10, command=showdatay).place(x=480, y=200)

def guanxi():
    import matplotlib.pyplot as plt
    import seaborn
    # 绘制散点图矩阵
    list=['1']
    # for i in bl.x:
    #     if(i!='+'):
    #         list.append(i)
    for index in range(len(bl.x) - 1):
        # print(girl_str[index])
        if (bl.x[index] != '+'):
            j = bl.x[index]
            if (bl.x[index + 1] != '+'):
                k = j + bl.x[index + 1]
                list.append(k)
                continue
            list.append(bl.x[index])
        # if (girl_str[len(girl_str)-2] == '+'):
        #     list.append(girl_str[index])
    if (bl.x[len(bl.x) - 2] == '+'):
        list.append(bl.x[index + 1])


    list.append(bl.y)
    del(list[0])
    seaborn.pairplot(Profit.loc[:, list])
    # seaborn.pairplot(Profit.loc[:, ['M', 'V', 'AC', bl.y]])
    # 显示图形
    plt.show()
    zong=2
    # 模型修正

# 右半区的编辑界面
labenew1 = tkinter.Label(root1, text='The optimization model')
labenew1.place(x=340, y=10)
btn2 = tkinter.Button(root1, text='Look at the relationship between y and x', width=40, command=guanxi).place(x=340, y=40)
labenew2 = tkinter.Label(root1, text='Let is take x that has a linear relationship')
labenew2.place(x=340, y=90)


btny2 = tkinter.Button(root1, text='Ok', width=5,height=1, command=loginy).place(x=540, y=120)


root1.mainloop()


print('111111111')
print(bl.lujing)
print(bl.bianliang)
if(zong==1):
    import pandas as pd
    import matplotlib.pyplot as plt
    import sns as sns
    import statsmodels.api as sm
    from sklearn import model_selection
    from sklearn import model_selection
    # 导入数据


    # Profit = pd.read_excel(r'D:\\pre.xlsx')
    Profit = pd.read_excel(bl.lujing)#str即为输入的文件路径
    # 将数据集拆分为训练集和测试集
    train, test = model_selection.train_test_split(Profit, test_size = 0.2, random_state=1234)
    # 根据train数据集建模

    model = sm.formula.ols(bl.bianliang, data = train).fit()#可以输入样本x值
    # model = sm.formula.ols('pre ~  M + V + AC ', data = train).fit()
    print('Partial regression coefficient of the model\n', model.params)
    # 删除test数据集中的Profit变量，用剩下的自变量进行预测
    test_X = test.drop(labels = 'pre', axis = 1)
    pred = model.predict(exog = test_X)
    bl.blDataFrame=pd.DataFrame({'Prediction':pred,'Real':test.pre})#全局变量修饰
    # print('对比预测值和实际值的差异：\n',pd.DataFrame({'Prediction':pred,'Real':test.pre}))


    import numpy as np
    # 计算建模数据中，因变量的均值
    ybar = train.pre.mean()
    # 统计变量个数和观测个数
    p = model.df_model
    n = train.shape[0]
    # 计算回归离差平方和
    RSS = np.sum((model.fittedvalues-ybar) ** 2)
    # 计算误差平方和
    ESS = np.sum(model.resid ** 2)
    # 计算F统计量的值
    F = (RSS/p)/(ESS/(n-p-1))
    print('F统计量的值：',F)
    # 返回模型中的F值
    model.fvalue



    from scipy.stats import f
    # 计算F分布的理论值
    F_Theroy = f.ppf(q=0.95, dfn = p,dfd = n-p-1)
    print('F分布的理论值为：',F_Theroy)
    print(n)
    model.summary()


    # 正态性检验
    # 直方图法
    # 导入第三方模块
    # import scipy.stats as stats
    # import seaborn as sns
    # # 中文和负号的正常显示
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # # 绘制直方图
    # sns.distplot(a = Profit.pre, bins = 10, fit = stats.norm, norm_hist = True,
    #              hist_kws = {'color':'steelblue', 'edgecolor':'black'},
    #              kde_kws = {'color':'black', 'linestyle':'--', 'label':'核密度曲线'},
    #              fit_kws = {'color':'red', 'linestyle':':', 'label':'正态密度曲线'})
    # # 显示图例
    # plt.legend()
    # # 显示图形
    # plt.show()

    plt.scatter(x = test.pre, y = pred)
    # 添加斜率为1，截距项为0的参考线
    # 添加轴标签
    plt.xlabel('Actual value')
    plt.ylabel('Predictive value')
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.plot([test.pre.min(),test.pre.max()],[test.pre.min(),test.pre.max()],
            color = 'red', linestyle = '--')

    # 显示图形
    plt.show()
