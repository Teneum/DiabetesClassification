from model import Model

model = Model()
model.load_model(filename='1')
print(model.predict(array=[8,125,96,0,0,0,0.232,54]))