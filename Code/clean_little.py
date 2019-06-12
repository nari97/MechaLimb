import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setup(6, GPIO.OUT)
        
pwm_servo = GPIO.PWM(6, 50)
pwm_servo.start(13)
i=2
while True:
    pwm_servo.ChangeDutyCycle(2)
    i=i+1
    time.sleep(0.05)
    if(i>13):
        break