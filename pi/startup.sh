#!/bin/bash
cd /home/noah/Desktop/ENGG3112/ENGG3112/pi
source venv/bin/activate
sudo pigpiod
sleep 1
sudo venv/bin/python leds.py &
python main.py
