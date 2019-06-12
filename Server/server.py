from flask import Flask
from flask import request
from flask_cors import CORS,cross_origin
import os
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
@app.route("/test.py",methods=['GET'])  # consider to use more elegant URL in your JS
@cross_origin()
def get_x():
    #x = 2
    #print ('Called')
    if request.args['move']=='1':
        os.system("sudo python3 /home/pi/Desktop/FinalProject/FinalTest/RealTimeClassifier.py //home/pi/Desktop/FinalProject/S1-Delsys-15Class/T_T1.csv")
    if request.args['move']=='2':
        os.system("sudo python3 /home/pi/Desktop/FinalProject/FinalTest/RealTimeClassifier.py //home/pi/Desktop/FinalProject/S1-Delsys-15Class/I_I1.csv")
    if request.args['move']=='3':
        os.system("sudo python3 /home/pi/Desktop/FinalProject/FinalTest/RealTimeClassifier.py //home/pi/Desktop/FinalProject/S1-Delsys-15Class/M_M1.csv")
    if request.args['move']=='4':
        os.system("sudo python3 /home/pi/Desktop/FinalProject/FinalTest/RealTimeClassifier.py //home/pi/Desktop/FinalProject/S1-Delsys-15Class/L_L1.csv")
    if request.args['move']=='5':
        os.system("sudo python3 /home/pi/Desktop/FinalProject/FinalTest/RealTimeClassifier.py //home/pi/Desktop/FinalProject/S1-Delsys-15Class/R_R1.csv")
    if request.args['move']=='6':
        os.system("sudo sh /home/pi/Desktop/FinalProject/FinalTest/clean_hand.sh")    
        #print ('Done')
    return "hello"

if __name__ == "__main__":
    # here is starting of the development HTTP server
    
    app.run()
