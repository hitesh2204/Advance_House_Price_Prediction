import requests
from app.schema import HouseData

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv ()

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

prompt = PromptTemplate(
    template = """
                Our house price prediction model predicted a price of ₹{prediction}.
                Explain this prediction to the user in simple language.
                """,
    input_variables=['prediction']
)

parser = StrOutputParser()

house_data = HouseData(
    MSSubClass=60,
    MSZoning="RL",
    LotFrontage=65,
    LotArea=8450,
    Street="Pave",
    Alley="NA",
    LotShape="Reg",
    LandContour="Lvl",
    Utilities="AllPub",
    LotConfig="Inside",
    LandSlope="Gtl",
    Neighborhood="CollgCr",
    Condition1="Norm",
    Condition2="Norm",
    BldgType="1Fam",
    HouseStyle="2Story",
    OverallQual=7,
    OverallCond=5,
    YearBuilt=2003,
    YearRemodAdd=2003,
    RoofStyle="Gable",
    RoofMatl="CompShg",
    Exterior1st="VinylSd",
    Exterior2nd="VinylSd",
    MasVnrType="BrkFace",
    MasVnrArea=196,
    ExterQual="Gd",
    ExterCond="TA",
    Foundation="PConc",
    BsmtQual="Gd",
    BsmtCond="TA",
    BsmtExposure="No",
    BsmtFinType1="GLQ",
    BsmtFinSF1=706,
    BsmtFinType2="Unf",
    BsmtFinSF2=0,
    BsmtUnfSF=150,
    TotalBsmtSF=856,
    Heating="GasA",
    HeatingQC="Ex",
    CentralAir="Y",
    Electrical="SBrkr",
    FirstFlrSF=856,
    SecondFlrSF=750,
    LowQualFinSF=0,
    GrLivArea=1710,
    BsmtFullBath=1,
    BsmtHalfBath=0,
    FullBath=2,
    HalfBath=3,
    BedroomAbvGr=3,
    KitchenAbvGr=1,
    KitchenQual="Gd",
    TotRmsAbvGrd=8,
    Functional="Typ",
    Fireplaces=0,
    FireplaceQu="NA",
    GarageType="Attchd",
    GarageYrBlt=2003,
    GarageFinish="RFn",
    GarageCars=2,
    GarageArea=548,
    GarageQual="TA",
    GarageCond="TA",
    PavedDrive="Y",
    WoodDeckSF=0,
    OpenPorchSF=50,
    EnclosedPorch=0,
    ThreeSsnPorch=0,
    ScreenPorch=0,
    PoolArea=0,
    PoolQC="NA",
    Fence="NA",
    MiscFeature="NA",
    MiscVal=0,
    MoSold=2,
    YrSold=2008,
    SaleType="WD",
    SaleCondition="Normal"
)

payload = house_data.model_dump(by_alias=True)

response = requests.post("http://127.0.0.1:8000/predict",
              json = payload)

prediction = response.json()['prediction']

chain = prompt | llm | parser

result = chain.invoke({'prediction':prediction})
print(result)

