import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import sys
import math


print("Initiating Program.")

#데이터를 가공하여 분석하기 쉽도록 만드는 클래스입니다.
class dataGenerator:

    #필요한 애트리뷰트를 미리 선언해 놓겠습니다.
    data = None #클래스의 pandas 데이터 원문을 담는 애트리뷰트입니다.
    original = None

    def __init__(self, fileName):

        try:
            print("Instance Generate Command Detected. Instance Generating...\n 인스턴스 생성 명령이 감지되었습니다. 인스턴스 생성 중...")
            print("\n\nData received. Data reading. Might take time...\n데이터 입수. 데이터 읽는 중입니다. 데이터 크기에 따라 시간이 다소 소요될 수 있습니다...")

            self.data = pd.read_csv(fileName)
            self.original = self.data
        except:

            print("\n\nInitiated Failed. Recommendation: Check whether you forget to enter 'csv' file name as parameter. \n 인스턴스 생성 실패. 제안: 'csv' 파일 이름을 제대"
                  "파라미터로 넣었는지 확인하십시오.")

        else:

            print(self.data)
            print(
                "\n\nData reading Complete. Data type in pandas DataFrame object. Displaying data...\n 데이터 읽기가 완료되었습니다. 데이터 타입은 판다스 데이터프레임 객체입니다. 데이터 개요를 표시하였습니다.")
            print("\n\nInstance Successfully Generated. \n 인스턴스 생성 성공. 생성 프로세스 종료.")

    def dataIndexing(self, columnName):

        self.data.index = self.data.loc[:, columnName]
        print(self.data)
        print("Indexing Complete. \n %s열에 대해 인덱싱이 완료되었습니다." % columnName)


    def dataIndexInitiate(self):

        self.data.index = range(len(self.data.iloc[:, 1]))
        print(self.data)
        print("Index initiate Complete. \n 객체의 data 애트리뷰트의 인덱싱이 초기되었습니다.")




    def dataExtract(self, dataColumnName, extractingElementName):



        try:

            return self.data[:][(self.data[dataColumnName] == extractingElementName)]

        except:

            print("Error.Cannot Extract data. Recommendation: Check if element is in the existing column of your data. \n"
                  "오류. 데이터 추출에 실패했습니다. 제안: 입력하신 칼럼이 존재하는지, 혹은 해당 칼럼 안에 입력하신 데이터가 존재하는지 확인해 주세요.")


    def columnBreaker(self, columnName, index=None, type="numpy"):

        if index == None and type == "pandas":


            return pd.DataFrame(self.data[:][columnName])
            print("column slicing complete. Data type in pandas DataFrame. \n 칼럼 분리에 성공했습니다. 자료형은 판다스 데이터프레임 객체입니다.")
        elif index == None and type == "numpy":

            return pd.DataFrame(self.data[:][columnName]).to_numpy().T[0]
            print("column slicing complete. Data type in Numpy array. \n 칼럼 분리에 성공했습니다. 자료형은 넘파이 객체입니다.")

        elif index != None and type == "pandas":

            index2 = index.tolist()

            return pd.DataFrame(self.data[:][columnName], index=index2)
            print("column slicing complete. Data type in pandas DataFrame with custom index. \n 칼럼 분리에 성공했습니다. 자료형은 인덱스가 붙은 판다스 데이터프레임 객체입니다.")

    def noneCare(self, type):

        if type == None:

            print("Error. Please type 'type' parameter. \n 오류. 어떤 종류의 데이터 정화 작업인지 'type' 파라미터를 통해 알려주십시오.")

        elif type == "delete":

            self.data.dropna()

        elif type == "check":

            result = pd.DataFrame([], index=self.data.colums.tolist())
            print("Searching initiated. \n None 객체를 찾습니다. \n 참고로, 단순히 None의 존재만 파악하실 것이면 판다스 객체는 해당 메소드를 지원합니다.")
            for n in range[len(self.data.index)]:



                for item in self.data.iloc[n][:]:

                    if item == None or item == "None":

                        result.append(self.data[n][:])

                if n % 50 == 0 and n != 0:
                    print("%d rows searched. \n %d번째 행을 분석중입니다.")


            print("Searching Complete. Displaying results...\n 모든 행과 열을 확인했습니다. 결과 출력 중...")

            print(result)


def model(loc, std, max, input, parameter):

    if input >= loc:

        modelResult = (-(1/(std*parameter)) * (input-loc)) + max

    else:

        modelResult = (std * (1 / (parameter)) * (input - loc)) + max

    if modelResult < 0 :

        return 0

    else:

        return  modelResult




def standard(oneDimensionNumpyArray):

    return oneDimensionNumpyArray/oneDimensionNumpyArray.sum()


def dataIndexing(dataset, columnName):

    dataset.index=dataset.loc[:, columnName]
    print(dataset)
    print("Indexing Complete. \n %s열에 대해 인덱싱이 완료되었습니다." %columnName)

    return dataset

def dataIndexInitiate(dataset):

    dataset.index = range(len(dataset.iloc[:, 1]))
    print(dataset)
    print("Index initiate Complete. \n 객체의 data 애트리뷰트의 인덱싱이 초기되었습니다.")
    return dataset


def groupbyToDataframe(series):

    return series.reset_index()


def dataLabeling(dataframe, listOfLocations):

    dataIndexing(dataframe, "CARD_SIDO_NM")

    dataLabel = dataframe.loc[listOfLocations, :]

    dataLabel = dataIndexInitiate(dataLabel)

    return dataLabel

def dataCleaning(dataframe):
    dataframe1 = dataIndexing(dataframe, "REG_YYMM")

    # 코로나 전후의 산업 상태를 추출
    precorona = dataframe1.loc[[1, 2, 3, 4], :]
    print(precorona)
    postcorona = dataframe1.loc[[13, 14, 15, 16], :]
    print(postcorona)

    dataIndexInitiate(precorona)
    dataIndexInitiate(postcorona)

    for n in [1, 2, 3, 4]:
        postcorona.loc[:, "REG_YYMM"][(postcorona.loc[:, "REG_YYMM"] == 12 + n)] = n

    print(postcorona)

    preco = precorona.iloc[:, -1]
    postco = postcorona.iloc[:, -1]
    preco.index = pd.MultiIndex.from_frame(precorona.iloc[:, :-1])
    postco.index = pd.MultiIndex.from_frame(postcorona.iloc[:, :-1])

    print("Post Corona")
    print(postco)
    print("Pre Corona")
    print(preco)

    data2 = (postco - preco).dropna()

    print("Delta data")
    print(data2)

    cleandata = data2.to_frame().reset_index()
    print(cleandata)
    return dataIndexing(cleandata, "REG_YYMM")


def moneyPerUseOptimize(dataframe):


    MPU = dataframe.groupby(["REG_YYMM", "CARD_SIDO_NM", "STD_CLSS_NM"])["Money_per_use"].mean()

    MPU = groupbyToDataframe(MPU)

    print(MPU)

    data2 = dataCleaning(MPU)



    alpha = list()



    for location in data2.loc[:, "CARD_SIDO_NM"].unique():

        for industry in data2.loc[:, "STD_CLSS_NM"].unique():

            checker = data2.loc[(data2["CARD_SIDO_NM"]==location) & (data2["STD_CLSS_NM"] == industry)]


            print("Checker")
            print(checker)

            alphamin = checker.loc[:, "Money_per_use"].min()
            alphaminLoc = 3


            newChecker = list()

            if len(checker.iloc[:, 0]) == 4:

                for num in checker.loc[:, "REG_YYMM"]:

                    for value in -checker.loc[:, "Money_per_use"][checker["REG_YYMM"] == num]:
                        if value <= 0:

                            for n in range(int(-value)):
                                newChecker.append(int((6/num)-1))

                            print("append", int((6/num)-1), -value)

                        else:

                            for n in range(int(value)):
                                newChecker.append(num)

                            print("append", num, value)

                mu, std = norm.fit(newChecker)

                alpha.append([location, industry, mu, std, alphamin])





            elif len(checker.iloc[:, 0]) > 0:

                alpha.append([location, industry, alphaminLoc, 1, alphamin])


            else:

                alpha.append([location, industry, alphaminLoc, 1, 0])


    alpha = pd.DataFrame(alpha, columns= ["CARD_SIDO_NM", "STD_CLSS_NM","alphaLoc", "alphaStand", "alphaMin"])


    print(alpha)

    return alpha

def usePerCumstomerOptimize(dataframe):

    CPU = dataframe.groupby(["REG_YYMM", "CARD_SIDO_NM", "STD_CLSS_NM"])["use_per_cumstomer"].sum()

    CPU = groupbyToDataframe(CPU)

    print(CPU)

    data2 = dataCleaning(CPU)


    muList0 = list()
    stdList0 = list()
    muList1 = list()
    stdList1 = list()
    muList2 = list()
    stdList2 = list()

    customModel = list()

    for location in data2.loc[:, "CARD_SIDO_NM"].unique():

        for industry in data2.loc[:, "STD_CLSS_NM"].unique():

            checker = data2.loc[(data2["CARD_SIDO_NM"] == location) & (data2["STD_CLSS_NM"] == industry)]

            checker.loc[:, "use_per_cumstomer"][(checker["use_per_cumstomer"] > 0)] = -1
            print("Checker")
            print(checker)

            newChecker = list()




            if len(checker.iloc[:, 1]) == 4:

                for num in checker.loc[:, "REG_YYMM"]:

                    for value in -checker.loc[:, "use_per_cumstomer"][checker["REG_YYMM"] == num]:
                        if value <= 0:
                            for n in range(int(-value)):

                                newChecker.append(int((6/num)-1))

                            print("append", int((6/num)-1), -value)

                        else:

                            for n in range(int(value)):
                                newChecker.append(num)

                            print("append", num, value)


                mu, std = norm.fit(newChecker)

               # for use in checker.


                if location in ["대구", "인천", "대전", "울산", "세종", "충북", "전북"]:

                    muList0.append(mu)

                    stdList0.append(std)
                    customModel.append([location, industry, mu, "generalStand0"])

                    print([location, industry, mu, std])

                elif location in ["충남", "전남", "경북", "경남"]:

                    muList1.append(mu)

                    stdList1.append(std)
                    customModel.append([location, industry, mu, "generalStand1"])

                    print([location, industry, mu, std])

                else:
                    muList2.append(mu)

                    stdList2.append(std)
                    customModel.append([location, industry, mu, "generalStand2"])

                    print([location, industry, mu, std])

            else:

                if location in ["대구", "인천", "대전", "울산", "세종", "충북", "전북"]:

                    customModel.append([location, industry, "generalMean0", "generalStand0"])

                elif location in ["충남", "전남", "경북", "경남"]:

                    customModel.append([location, industry, "generalMean1", "generalStand1"])

                else:

                    customModel.append([location, industry, "generalMean2", "generalStand2"])


    modelMean0 = statistics.mean(muList0)
    modelStan0 = statistics.mean(stdList0)
    modelMean1 = statistics.mean(muList1)
    modelStan1 = statistics.mean(stdList1)
    modelMean2 = statistics.mean(muList2)
    modelStan2 = statistics.mean(stdList2)

    model = pd.DataFrame(np.array(customModel),  columns= ["CARD_SIDO_NM", "STD_CLSS_NM","modelMean", "modelStand"])

    model.loc[:,"modelMean"][(model["modelMean"]=="generalMean0")] = modelMean0
    model.loc[:, "modelStand"][(model["modelStand"] == "generalStand0")] = modelStan0
    model.loc[:,"modelMean"][(model["modelMean"]=="generalMean1")] = modelMean1
    model.loc[:, "modelStand"][(model["modelStand"] == "generalStand1")] = modelStan1
    model.loc[:,"modelMean"][(model["modelMean"]=="generalMean2")] = modelMean2
    model.loc[:, "modelStand"][(model["modelStand"] == "generalStand2")] = modelStan2

    return model


def AIparameter(LabeledData):
    alpha3 = moneyPerUseOptimize(LabeledData)
#["CARD_SIDO_NM", "STD_CLSS_NM","alphaLoc", "alphaStand", "alphaMin"]

    model = usePerCumstomerOptimize(LabeledData)

    print(alpha3)
    print(model)
    print("alpha index")
    model["alphaLoc"] = alpha3.iloc[:, -3]
    model["alphaStand"] = alpha3.iloc[:, -2]
    model["alphaMin"] = alpha3.iloc[:, -1]
    print(model)

    return model

def AI(originalData, parameters, forcastingDate, parameter1, parameter2 , parameter3):


    precoronas = originalData.groupby(["CARD_SIDO_NM", "REG_YYMM", "STD_CLSS_NM"])["AMT", "CNT"].sum()
    print("precoronas")
    print(precoronas)
    precoronas = precoronas.reset_index()

    newColumn = {"REG_YYMM": list(), "AMT":list()}


    for n in range(len(parameters.iloc[:, 1])):



        parameter = parameters.iloc[n, :]


        precoronas1 = precoronas.iloc[:, :][(precoronas["REG_YYMM"]== 201900 + forcastingDate)]

        precorona =precoronas1.loc[:, "AMT"].sum()
        preuse = precoronas1.loc[:, "CNT"].sum()
        prealpha = precorona / preuse
 #already defined.
        print("precorona")
        print(precorona)
        print("Parameter")
        print(parameter)
        print("forcastingDate", forcastingDate)
        print("parameter.loc[modelmean]", parameter.loc["modelMean"])
        print("parameter.loc[0, modelStand]", parameter.loc["modelStand"])

        Loc = float(parameter.loc["modelMean"])
        Scale = float(parameter.loc["modelStand"])
        alphaLoc = float(parameter.loc["alphaLoc"])
        alphaStand = float(parameter.loc["alphaStand"])
        alphaMin = float(parameter.loc["alphaMin"])


        if preuse == 0 or alphaMin == 0:

            postcorona = 0

        else:

            alphaModel = norm.pdf(forcastingDate, loc=alphaLoc, scale= alphaStand * parameter2)
            modelPre = norm.pdf(forcastingDate, loc=Loc , scale= Scale * parameter1)
            alphaModelStand = norm.pdf(alphaLoc, loc=alphaLoc, scale=alphaStand * parameter2 )
            ModelStan = norm.pdf(Loc, loc=Loc, scale=Scale * parameter1)





            print("Loc:", Loc)
            print("Scale:", Scale)
            print("AlphaLoc:", alphaLoc)
            print("AlphaStand:", alphaStand)
            print("preuse:", preuse)
            print("alphaModel:", alphaModel)
            print("alphaModelStand:", alphaModelStand)
            print("ModelPre:", modelPre)
            print("ModelStan:", ModelStan)

            postcorona = precorona * (1+ (alphaMin/abs(alphaMin))* ((modelPre *alphaModel) / (alphaModelStand * ModelStan)))

            print("precorona")
            print(precorona)
            print("\n postcorona")
            print(postcorona, "\n\n", "=" * 30)

            if postcorona <= 0:

                sys.exit()








        newColumn["REG_YYMM"].append(202000 + forcastingDate)
        newColumn["AMT"].append(postcorona)



    parameters = parameters.assign(REG_YYMM= newColumn["REG_YYMM"], AMT=newColumn["AMT"])



    return parameters





'''
    sub = prePredictionData
    
    

    for location in sub.loc[:, "CARD_SIDO_NM"].unique():

        precorona1 = originalData.iloc[:, :][
            (originalData["CARD_SIDO_NM"] == location)  & (
                        originalData["REG_YYMM"] == forcastingDate + 201900)]
        preuse1 = originalData.iloc[:,:][
            (originalData["CARD_SIDO_NM"] == location) & (
                        originalData["REG_YYMM"] == forcastingDate + 201900)]

        for industry in sub.loc[:, "STD_CLSS_NM"].unique():


            precorona = precorona1["AMT"][(originalData["STD_CLSS_NM"]==industry)]
            preuse = preuse1["CNT"][(originalData["STD_CLSS_NM"]==industry)]
            precorona = precorona.sum()
            preuse = preuse.sum()
            print("precorona")
            print(precorona)
            print("Parameter")
            print(parameters)

            if precorona != 0:
                parameter = parameters[(parameters["CARD_SIDO_NM"] == location) & (parameters["STD_CLSS_NM"] == industry)]

                if parameter.empty or parameter.loc[:, "alphaMin"].mean() == 0:
                    sub.loc[:, "AMT"][(sub["CARD_SIDO_NM"] == location) & (sub["STD_CLSS_NM"] == industry)] = precorona

                else:

                    print(parameter)


                    parameter.reset_index()
                    print("forcastingDate", forcastingDate)
                    print("parameter.loc[modelmean]", parameter.loc[:, "modelMean"])
                    print("parameter.loc[0, modelStand]", parameter.loc[:, "modelStand"])

                    Loc = float(parameter.loc[:, "modelMean"].mean())
                    Scale = float(parameter.loc[:, "modelStand"].mean())*parameter1
                    alphaLoc = float(parameter.loc[:, "alphaLoc"].mean())
                    alphaStand = float(parameter.loc[:, "alphaStand"].mean())
                    alphaMin = float(parameter.loc[:, "alphaMin"].mean())
                    alphaModel = model(alphaLoc, alphaStand, alphaMin, forcastingDate, parameter2)


                    modelPre = model(Loc, Scale, preuse, forcastingDate, parameter1)
                    #customer = originalData["CSTMR_CNT"][(originalData["CARD_SIDO_NM"]==location) & (originalData["STD_CLSS_NM"]==industry) & (originalData["REG_YYMM"]==forcastingDate+201900)]
                    #customer = customer.sum()



                    print("Loc:", Loc)
                    print("Scale:", Scale)
                    print("Alpha:", alphaModel)
                    print("ModelPre:", modelPre)
                    print("AlphaStand:", alphaStand)
                    print("preuse:", preuse)

                    postcorona = precorona + (modelPre * alphaModel)
                    print("precorona")
                    print(precorona)
                    print("\n postcorona")
                    print(postcorona, "\n\n", "="*30)

                    if pd.isna(postcorona) :

                        print("AI Error. Wrong Parameter.")
                        sys.exit()

                    elif postcorona < 0 :

                        print("AI Error. Wrong Parameter.")
                        sys.exit()



                    else:
                        sub.loc[:, "AMT"][(sub["REG_YYMM"] == 202000 + forcastingDate) & (sub["CARD_SIDO_NM"]==location) & (sub["STD_CLSS_NM"]==industry)] = postcorona



            else:

                sub.loc[:, "AMT"][(sub["REG_YYMM"] == 202000 + forcastingDate) & (sub["CARD_SIDO_NM"] == location) & (sub["STD_CLSS_NM"] == industry)] = precorona




    return sub



'''










#초기 데이터 정화작업


ccdata = dataGenerator("data.csv")

print(ccdata.data)
newdata = pd.read_csv("202004.csv")
ccdata.data = pd.concat([ccdata.data, newdata])

print(ccdata.data)


ccdata.data.iloc[1::5000].to_csv("/Users/gimhyeonjun/PycharmProjects/JejuCC_AI/id.csv")

print(ccdata.dataIndexing("REG_YYMM"))

print(ccdata.dataIndexInitiate())

customer = ccdata.columnBreaker("CSTMR_CNT")

use = ccdata.columnBreaker("CNT")

usedMoney = ccdata.columnBreaker("AMT")

moneyPerCus = usedMoney / customer

moneyPerPurchase = usedMoney/use

cusPerUse = use

cusPerUseStan = cusPerUse / cusPerUse.sum()
cusPerUseSum = cusPerUse.sum()

variableArray = np.array([moneyPerPurchase, moneyPerCus, cusPerUse])




print("Variable Ready. \n 변수들이 준비되었습니다.")

#불필요한 데이터 삭제

ccdataNoMove = ccdata.data.iloc[ : , 0:6]
variable = pd.DataFrame(variableArray.T, columns=["Money_per_use", "Money_per_customer", "use_per_cumstomer"])

print(variable)

ccdataNoMove = ccdataNoMove.join(variable)


print(ccdataNoMove)

ccdata.data = ccdataNoMove


print(ccdata.data)


#날짜 넘버링 데이터 클리닝

for n in range(1, 13):
    ccdata.data.loc[:, "REG_YYMM"][(ccdata.data.loc[:, "REG_YYMM"] == 201900 + n)] = n

for n in range(0, 4):
    ccdata.data.loc[:, "REG_YYMM"][(ccdata.data.loc[:, "REG_YYMM"] == 202001 + n)] = n + 13


print("Year and Month data cleansing Complete. Now in indexed int data. 년월 데이터 정화 완료. 이제는 달을 간격으로 인덱싱된 정수 자료형입니다.")

print(ccdata.data)

#관광 수준에 따른 데이터 레이블링 분류


dataLabel0 = ccdata.data
#dataLabel1 = dataLabeling(ccdata.data, ["충남", "전남", "경북", "경남"])
#dataLabel2 = dataLabeling(ccdata.data, ["제주", "강원", "경기", "서울"])


model0 = AIparameter(dataLabel0)
#model1 = AIparameter(dataLabel1)
#model2 = AIparameter(dataLabel2)

#마지막 파라미터를 높이면전체적으로 확산.
sub1=AI(ccdata.original, model0, 4, 3, 3, 3)
sub=AI(ccdata.original, model0, 7, 3, 3, 3)

sub = sub.append(sub1)

sub = sub.sort_values(by=["REG_YYMM", "CARD_SIDO_NM", "STD_CLSS_NM"])

id = list()
for n in range(len(sub.iloc[:, 1])):
    id.append(n)

sub = sub.assign(id=id)


sub.loc[:, ["id","REG_YYMM","CARD_SIDO_NM","STD_CLSS_NM","AMT"]].to_csv("/Users/gimhyeonjun/PycharmProjects/JejuCC_AI/FinalSubmission3.csv", index=False)


'''

dataLabel3 = dataLabeling(ccdata.data, ["강원"])

model3 = AIparameter(dataLabel3)

AI(ccdata.original, model3, 4, cusPerUseSum)


'''




