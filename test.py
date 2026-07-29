import pandas as pd

@app.post("/predict")
def predict(data: HouseData):
    return {
        'message':'api working'
    }
try:
        logger.info("Loading the input data")

        df = pd.DataFrame([data.model_dump(by_alias=True)])

        logger.info("User data successfully loaded")

        preprocess_data = preprocessor.transform(df)

        logger.info("Successfully preprocess data")

        prediction = model.predict(preprocess_data)

        logger.info("Prediction done successfully")

        return {
            "prediction": float(prediction[0])
        }
    except Exception as e:

        logger.error(str(CustomException(e,sys)))
        raise e



def load_preprocessor():
    with open("artifacts//preprocessor.pkl",'rb') as file:
        preprocessor_obj = pickle.load(file)
    return preprocessor_obj

def load_model():
    with open("artifacts//model.pkl","rb") as file:
        model_obj = pickle.load(file)
    return model_obj