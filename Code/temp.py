import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

GPIO.setup(27, GPIO.OUT)
GPIO.setup(6,GPIO.OUT)
GPIO.setup(22,GPIO.OUT)
pwm_servo = GPIO.PWM(27, 50)
pwm_servo.start(13)
pwm0 = GPIO.PWM(6,50)
pwm0.start(13)
pwm1 = GPIO.PWM(22,50)
pwm1.start(2)
i=13
j = 2
while True:
    pwm_servo.ChangeDutyCycle(i)

    i=i-1

    time.sleep(0.05)
    pwm0.ChangeDutyCycle(j)
    j=j+1
    pwm1.ChangeDutyCycle(j)
    time.sleep(0.05)
    if(i<2):
        break

